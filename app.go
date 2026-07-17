package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"embed"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	wailsruntime "github.com/wailsapp/wails/v2/pkg/runtime"
)

//go:embed workers/*.py
var workerFiles embed.FS

type App struct {
	ctx  context.Context
	mu   sync.Mutex
	jobs []Job
}
type Stage struct {
	ID     string `json:"id"`
	Name   string `json:"name"`
	Status string `json:"status"`
	Detail string `json:"detail"`
}
type Artifact struct {
	Stage   string         `json:"stage"`
	Kind    string         `json:"kind"`
	Path    string         `json:"path"`
	Metrics map[string]any `json:"metrics,omitempty"`
}
type LogEntry struct {
	Time    string `json:"time"`
	Stage   string `json:"stage"`
	Level   string `json:"level"`
	Message string `json:"message"`
}
type Job struct {
	ID        string     `json:"id"`
	Name      string     `json:"name"`
	Created   string     `json:"created"`
	Status    string     `json:"status"`
	Progress  int        `json:"progress"`
	Image     string     `json:"image,omitempty"`
	ImageHash string     `json:"imageHash,omitempty"`
	Workspace string     `json:"workspace,omitempty"`
	Stages    []Stage    `json:"stages"`
	Artifacts []Artifact `json:"artifacts,omitempty"`
	Logs      []LogEntry `json:"logs,omitempty"`
}
type SystemInfo struct {
	Platform  string `json:"platform"`
	Workspace string `json:"workspace"`
	Jobs      int    `json:"jobs"`
	Python    bool   `json:"python"`
}
type workerEvent struct {
	Type     string         `json:"type"`
	Kind     string         `json:"kind"`
	Path     string         `json:"path"`
	Message  string         `json:"message"`
	Progress float64        `json:"progress"`
	Metrics  map[string]any `json:"metrics"`
}

func NewApp() *App                         { return &App{jobs: []Job{}} }
func (a *App) startup(ctx context.Context) { a.ctx = ctx; a.load() }
func (a *App) rootPath() string            { d, _ := os.UserConfigDir(); return filepath.Join(d, "SpriteEngine") }
func (a *App) dataPath() string            { return filepath.Join(a.rootPath(), "jobs.json") }
func (a *App) load() {
	b, e := os.ReadFile(a.dataPath())
	if e == nil {
		_ = json.Unmarshal(b, &a.jobs)
	}
}
func (a *App) save() {
	_ = os.MkdirAll(filepath.Dir(a.dataPath()), 0755)
	b, _ := json.MarshalIndent(a.jobs, "", "  ")
	_ = os.WriteFile(a.dataPath(), b, 0644)
}
func pipeline() []Stage {
	return []Stage{{"prepare", "Image cleanup", "ready", "Background removal & subject validation"}, {"reconstruct", "3D reconstruction", "queued", "Multi-view diffusion → textured mesh"}, {"retopo", "Mesh cleanup", "queued", "Retopology, UV & material pass"}, {"rig", "Auto rig", "queued", "Skeleton, skin weights & deformation check"}, {"motion", "Animation", "queued", "Motion generation and retargeting"}, {"export", "Export", "queued", "GLB / FBX / USDZ package"}}
}
func newID() string {
	return fmt.Sprintf("%s-%d", time.Now().Format("20060102-150405"), time.Now().UnixNano()%1000000)
}
func (a *App) createJobLocked(name string) Job {
	id := newID()
	ws := filepath.Join(a.rootPath(), "projects", id)
	j := Job{ID: id, Name: name, Created: time.Now().Format(time.RFC3339), Status: "draft", Workspace: ws, Stages: pipeline(), Artifacts: []Artifact{}, Logs: []LogEntry{}}
	a.jobs = append([]Job{j}, a.jobs...)
	_ = os.MkdirAll(ws, 0755)
	a.save()
	return j
}
func (a *App) CreateJob(name string) (Job, error) {
	if strings.TrimSpace(name) == "" {
		return Job{}, errors.New("name is required")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.createJobLocked(strings.TrimSpace(name)), nil
}
func (a *App) ListJobs() []Job {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]Job, len(a.jobs))
	copy(out, a.jobs)
	return out
}
func (a *App) ImportReference() (Job, error) {
	if a.ctx == nil {
		return Job{}, errors.New("application is not ready")
	}
	path, e := wailsruntime.OpenFileDialog(a.ctx, wailsruntime.OpenDialogOptions{Title: "Import character reference", Filters: []wailsruntime.FileFilter{{DisplayName: "Images (*.png;*.jpg;*.jpeg;*.webp)", Pattern: "*.png;*.jpg;*.jpeg;*.webp"}}})
	if e != nil || path == "" {
		if e != nil {
			return Job{}, e
		}
		return Job{}, errors.New("cancelled")
	}
	return a.importPath(path)
}
func (a *App) importPath(path string) (Job, error) {
	in, e := os.Open(path)
	if e != nil {
		return Job{}, e
	}
	defer in.Close()
	h := sha256.New()
	if _, e = io.Copy(h, in); e != nil {
		return Job{}, e
	}
	hash := hex.EncodeToString(h.Sum(nil))
	if _, e = in.Seek(0, 0); e != nil {
		return Job{}, e
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	j := a.createJobLocked(strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)))
	ext := strings.ToLower(filepath.Ext(path))
	dst := filepath.Join(j.Workspace, "source"+ext)
	out, e := os.Create(dst)
	if e != nil {
		return Job{}, e
	}
	_, ce := io.Copy(out, in)
	xe := out.Close()
	if ce != nil {
		return Job{}, ce
	}
	if xe != nil {
		return Job{}, xe
	}
	for i := range a.jobs {
		if a.jobs[i].ID == j.ID {
			a.jobs[i].Image = dst
			a.jobs[i].ImageHash = hash
			a.jobs[i].Status = "ready"
			a.jobs[i].Stages[0].Detail = "Reference secured · SHA-256 " + hash[:8]
			j = a.jobs[i]
			break
		}
	}
	a.save()
	return j, nil
}

func (a *App) workerPath() (string, error) {
	files := []string{"baseline_worker.py", "procedural_character.py"}
	dir := filepath.Join(a.rootPath(), "runtime")
	if e := os.MkdirAll(dir, 0755); e != nil {
		return "", e
	}
	for _, name := range files {
		data, e := workerFiles.ReadFile("workers/" + name)
		if e != nil {
			return "", e
		}
		if e = os.WriteFile(filepath.Join(dir, name), data, 0755); e != nil {
			return "", e
		}
	}
	return filepath.Join(dir, "baseline_worker.py"), nil
}
func nextStage(j Job) (int, bool) {
	for i, s := range j.Stages {
		if s.Status == "ready" {
			return i, true
		}
	}
	return 0, false
}
func (a *App) RunNextStage(id string) (Job, error) {
	a.mu.Lock()
	idx := -1
	stageIndex := -1
	var snapshot Job
	for i := range a.jobs {
		if a.jobs[i].ID == id {
			idx = i
			s, ok := nextStage(a.jobs[i])
			if !ok {
				a.mu.Unlock()
				return a.jobs[i], errors.New("no stage is ready")
			}
			stageIndex = s
			a.jobs[i].Stages[s].Status = "running"
			a.jobs[i].Status = "processing"
			a.jobs[i].Logs = append(a.jobs[i].Logs, LogEntry{time.Now().Format(time.RFC3339), a.jobs[i].Stages[s].ID, "info", "Stage started"})
			snapshot = a.jobs[i]
			a.save()
			break
		}
	}
	a.mu.Unlock()
	if idx < 0 {
		return Job{}, errors.New("job not found")
	}
	worker, e := a.workerPath()
	if e != nil {
		return a.failStage(idx, stageIndex, e)
	}
	input := snapshot.Image
	if stageIndex > 0 && len(snapshot.Artifacts) > 0 {
		input = snapshot.Artifacts[len(snapshot.Artifacts)-1].Path
	}
	req := map[string]any{"type": "run", "jobId": id, "stage": snapshot.Stages[stageIndex].ID, "workspace": snapshot.Workspace, "input": input, "adapter": "local-baseline", "options": map[string]any{"targetTriangles": 42000}}
	payload, _ := json.Marshal(req)
	cmd := exec.Command("python3", worker)
	cmd.Stdin = bytes.NewReader(append(payload, '\n'))
	stdout, e := cmd.StdoutPipe()
	if e != nil {
		return a.failStage(idx, stageIndex, e)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if e = cmd.Start(); e != nil {
		return a.failStage(idx, stageIndex, e)
	}
	events := []workerEvent{}
	scan := bufio.NewScanner(stdout)
	for scan.Scan() {
		var ev workerEvent
		if json.Unmarshal(scan.Bytes(), &ev) == nil {
			events = append(events, ev)
			if a.ctx != nil {
				wailsruntime.EventsEmit(a.ctx, "worker:event", ev)
			}
		}
	}
	e = cmd.Wait()
	if e != nil {
		return a.failStage(idx, stageIndex, fmt.Errorf("worker failed: %v %s", e, stderr.String()))
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, ev := range events {
		if ev.Type == "error" {
			a.jobs[idx].Stages[stageIndex].Status = "failed"
			return a.jobs[idx], errors.New(ev.Message)
		}
		if ev.Type == "artifact" {
			a.jobs[idx].Artifacts = append(a.jobs[idx].Artifacts, Artifact{snapshot.Stages[stageIndex].ID, ev.Kind, ev.Path, ev.Metrics})
		}
		if ev.Message != "" {
			a.jobs[idx].Logs = append(a.jobs[idx].Logs, LogEntry{time.Now().Format(time.RFC3339), snapshot.Stages[stageIndex].ID, "info", ev.Message})
		}
	}
	a.jobs[idx].Stages[stageIndex].Status = "done"
	a.jobs[idx].Stages[stageIndex].Detail = "Completed · local-baseline"
	if stageIndex+1 < len(a.jobs[idx].Stages) {
		a.jobs[idx].Stages[stageIndex+1].Status = "ready"
	} else {
		a.jobs[idx].Status = "complete"
	}
	a.jobs[idx].Progress = (stageIndex + 1) * 100 / len(a.jobs[idx].Stages)
	a.save()
	return a.jobs[idx], nil
}
func (a *App) failStage(i, s int, cause error) (Job, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.jobs[i].Stages[s].Status = "failed"
	a.jobs[i].Status = "failed"
	a.jobs[i].Logs = append(a.jobs[i].Logs, LogEntry{time.Now().Format(time.RFC3339), a.jobs[i].Stages[s].ID, "error", cause.Error()})
	a.save()
	return a.jobs[i], cause
}

func (a *App) ReadArtifact(path string) (string, error) {
	a.mu.Lock()
	allowed := false
	for _, j := range a.jobs {
		for _, artifact := range j.Artifacts {
			if artifact.Path == path && strings.HasSuffix(strings.ToLower(path), ".glb") {
				allowed = true
				break
			}
		}
	}
	a.mu.Unlock()
	if !allowed {
		return "", errors.New("artifact is not registered")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if len(data) > 100*1024*1024 {
		return "", errors.New("artifact exceeds preview limit")
	}
	if len(data) < 4 || string(data[:4]) != "glTF" {
		return "", errors.New("artifact is not a GLB")
	}
	return "data:model/gltf-binary;base64," + base64.StdEncoding.EncodeToString(data), nil
}

func (a *App) RunAllStages(id string) (Job, error) {
	for {
		a.mu.Lock()
		var current Job
		found := false
		for _, j := range a.jobs {
			if j.ID == id {
				current, found = j, true
				break
			}
		}
		a.mu.Unlock()
		if !found {
			return Job{}, errors.New("job not found")
		}
		if current.Status == "complete" {
			return current, nil
		}
		if _, ok := nextStage(current); !ok {
			return current, errors.New("pipeline has no ready stage")
		}
		updated, err := a.RunNextStage(id)
		if err != nil {
			return updated, err
		}
	}
}

// AdvanceJob is retained for older generated bindings and delegates to the real worker pipeline.
func (a *App) AdvanceJob(id string) (Job, error) { return a.RunNextStage(id) }
func (a *App) OpenWorkspace(id string) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, j := range a.jobs {
		if j.ID == id {
			if j.Workspace == "" {
				return errors.New("workspace unavailable")
			}
			if e := os.MkdirAll(j.Workspace, 0755); e != nil {
				return e
			}
			wailsruntime.BrowserOpenURL(a.ctx, "file://"+j.Workspace)
			return nil
		}
	}
	return errors.New("job not found")
}
func (a *App) OpenExternal(url string) error {
	if a.ctx == nil {
		return errors.New("application is not ready")
	}
	allowed := []string{"https://itch.io/", "https://quaternius.com/", "https://www.mixamo.com/", "https://www.reallusion.com/", "https://github.com/"}
	for _, prefix := range allowed {
		if strings.HasPrefix(url, prefix) {
			wailsruntime.BrowserOpenURL(a.ctx, url)
			return nil
		}
	}
	return errors.New("external URL is not allowed")
}
func (a *App) SystemInfo() SystemInfo {
	_, e := exec.LookPath("python3")
	return SystemInfo{runtime.GOOS + "/" + runtime.GOARCH + " · Wails · Three.js", a.rootPath(), len(a.jobs), e == nil}
}
