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
	"sort"
	"strings"
	"sync"
	"time"

	wailsruntime "github.com/wailsapp/wails/v2/pkg/runtime"
)

//go:embed workers/*.py tools/matrix/validate_character.py
var workerFiles embed.FS

type App struct {
	ctx     context.Context
	mu      sync.Mutex
	jobs    []Job
	deleted map[string]bool // 이번 세션에서 삭제한 job ID — save 병합 시 부활 방지
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

func NewApp() *App                         { return &App{jobs: []Job{}, deleted: map[string]bool{}} }
func (a *App) startup(ctx context.Context) { a.ctx = ctx; a.load() }

// SPRITEENGINE_CONFIG_DIR isolates config (jobs.json, runpod.json) in tests:
// on macOS os.UserConfigDir ignores XDG_CONFIG_HOME, so without this seam the
// test suite reads the developer's real RunPod credentials and hits the live
// GPU endpoint.
func (a *App) rootPath() string {
	if dir := strings.TrimSpace(os.Getenv("SPRITEENGINE_CONFIG_DIR")); dir != "" {
		return dir
	}
	d, _ := os.UserConfigDir()
	return filepath.Join(d, "SpriteEngine")
}
func (a *App) dataPath() string { return filepath.Join(a.rootPath(), "jobs.json") }
func (a *App) load() {
	b, e := os.ReadFile(a.dataPath())
	if e == nil {
		_ = json.Unmarshal(b, &a.jobs)
	}
	// No worker survives an app restart: a job persisted as processing/running
	// is a zombie and would block DeleteJob/ResetStage forever.
	changed := false
	for i := range a.jobs {
		for s := range a.jobs[i].Stages {
			if a.jobs[i].Stages[s].Status == "running" {
				a.jobs[i].Stages[s].Status = "failed"
				changed = true
			}
		}
		if a.jobs[i].Status == "processing" {
			a.jobs[i].Status = "failed"
			a.jobs[i].Logs = append(a.jobs[i].Logs, LogEntry{time.Now().Format(time.RFC3339), "system", "error", "Stage interrupted by app shutdown"})
			changed = true
		}
		// 과거 버전이 stage 성공 후 status를 processing으로 남긴 채 저장한 job:
		// 실패한 stage가 없고 다음 ready stage가 있으면 실행 가능한 상태로 복구한다.
		if a.jobs[i].Status == "failed" {
			hasFailed := false
			for _, s := range a.jobs[i].Stages {
				if s.Status == "failed" {
					hasFailed = true
					break
				}
			}
			if !hasFailed {
				if _, ok := nextStage(a.jobs[i]); ok {
					a.jobs[i].Status = "ready"
					changed = true
				}
			}
		}
	}
	if changed {
		a.save()
	}
}
// mergeDiskJobsLocked는 외부 도구(tools/matrix/run_character.py 등)가 앱
// 실행 중 jobs.json에 직접 등록해 메모리에는 없는 job을 목록에 병합한다.
// 이번 세션에서 명시적으로 삭제한 job은 부활시키지 않는다.
// a.mu를 잡은 상태에서 호출해야 한다.
func (a *App) mergeDiskJobsLocked() {
	b, err := os.ReadFile(a.dataPath())
	if err != nil {
		return
	}
	var disk []Job
	if json.Unmarshal(b, &disk) != nil {
		return
	}
	known := make(map[string]bool, len(a.jobs))
	for _, j := range a.jobs {
		known[j.ID] = true
	}
	merged := false
	for _, j := range disk {
		if j.ID != "" && !known[j.ID] && !a.deleted[j.ID] {
			a.jobs = append(a.jobs, j)
			merged = true
		}
	}
	if merged {
		// UI는 최신 job이 위에 오길 기대한다 — ID 접두사가
		// "20060102-150405" 타임스탬프라 문자열 내림차순 = 최신순.
		sort.SliceStable(a.jobs, func(x, y int) bool { return a.jobs[x].ID > a.jobs[y].ID })
	}
}

func (a *App) save() {
	_ = os.MkdirAll(filepath.Dir(a.dataPath()), 0755)
	// 메모리 목록으로 통째로 덮어쓰면 외부 등록 job이 유실되므로 병합 후 저장.
	a.mergeDiskJobsLocked()
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
	// 목록 조회 시에도 병합 — 외부 등록 프로젝트가 앱 재시작 없이
	// 목록 새로고침만으로 나타난다.
	a.mergeDiskJobsLocked()
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
	// validate_character.py(렌더링 정상성 게이트)도 함께 추출해야 임베드
	// 실행 환경에서 워커가 rig/motion 결과를 실측 검증할 수 있다.
	files := map[string]string{
		"workers/baseline_worker.py":         "baseline_worker.py",
		"workers/procedural_character.py":    "procedural_character.py",
		"tools/matrix/validate_character.py": "validate_character.py",
	}
	dir := filepath.Join(a.rootPath(), "runtime")
	if e := os.MkdirAll(dir, 0755); e != nil {
		return "", e
	}
	for src, name := range files {
		data, e := workerFiles.ReadFile(src)
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
// emitJobUpdate는 stage 상태가 바뀔 때마다 frontend에 최신 Job을 push한다.
// RunAllStages처럼 오래 걸리는 호출 중에도 UI가 실시간으로 갱신되도록 한다.
func (a *App) emitJobUpdate(j Job) {
	if a.ctx != nil {
		wailsruntime.EventsEmit(a.ctx, "job:update", j)
	}
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
	a.emitJobUpdate(snapshot)
	input := snapshot.Image
	if stageIndex > 0 && len(snapshot.Artifacts) > 0 {
		input = snapshot.Artifacts[len(snapshot.Artifacts)-1].Path
	}
	if snapshot.Stages[stageIndex].ID == "reconstruct" && a.GetRunPodConfig().Configured {
		artifact, remoteErr := a.runPodReconstruct(input, snapshot.Workspace)
		if remoteErr != nil {
			return a.failStage(idx, stageIndex, remoteErr)
		}
		a.mu.Lock()
		a.jobs[idx].Artifacts = append(a.jobs[idx].Artifacts, artifact)
		a.jobs[idx].Stages[stageIndex].Status = "done"
		a.jobs[idx].Stages[stageIndex].Detail = "Completed · RunPod Hunyuan3D-2.1"
		if stageIndex+1 < len(a.jobs[idx].Stages) {
			a.jobs[idx].Stages[stageIndex+1].Status = "ready"
			a.jobs[idx].Status = "ready" // processing으로 남으면 delete/reset이 영원히 거부됨
		} else {
			a.jobs[idx].Status = "complete"
		}
		a.jobs[idx].Progress = (stageIndex + 1) * 100 / len(a.jobs[idx].Stages)
		a.jobs[idx].Logs = append(a.jobs[idx].Logs, LogEntry{time.Now().Format(time.RFC3339), "reconstruct", "info", "Real mesh generated by RunPod Hunyuan3D-2.1"})
		a.save()
		updated := a.jobs[idx]
		a.mu.Unlock()
		a.emitJobUpdate(updated)
		return updated, nil
	}
	// motion 단계도 RunPod이 설정돼 있으면 HY-Motion으로 실제 스켈레탈 클립을
	// 생성한다 (artifact/log 등록은 RunPodGenerateMotion이 수행).
	if snapshot.Stages[stageIndex].ID == "motion" && a.GetRunPodConfig().Configured {
		result, remoteErr := a.RunPodGenerateMotion(id, defaultMotionPrompts())
		if remoteErr != nil {
			return a.failStage(idx, stageIndex, remoteErr)
		}
		a.mu.Lock()
		a.jobs[idx].Stages[stageIndex].Status = "done"
		a.jobs[idx].Stages[stageIndex].Detail = fmt.Sprintf("Completed · RunPod HY-Motion (%d clips)", result.Clips)
		if stageIndex+1 < len(a.jobs[idx].Stages) {
			a.jobs[idx].Stages[stageIndex+1].Status = "ready"
			a.jobs[idx].Status = "ready" // processing으로 남으면 delete/reset이 영원히 거부됨
		} else {
			a.jobs[idx].Status = "complete"
		}
		a.jobs[idx].Progress = (stageIndex + 1) * 100 / len(a.jobs[idx].Stages)
		a.save()
		updated := a.jobs[idx]
		a.mu.Unlock()
		a.emitJobUpdate(updated)
		return updated, nil
	}
	worker, e := a.workerPath()
	if e != nil {
		return a.failStage(idx, stageIndex, e)
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
		// 워커는 실패 사유를 stdout의 error 이벤트로 보낸 뒤 exit 1 한다 —
		// exit code보다 그 메시지(예: 렌더링 정상성 게이트 사유)를 로그에 남긴다.
		for _, ev := range events {
			if ev.Type == "error" && ev.Message != "" {
				return a.failStage(idx, stageIndex, errors.New(ev.Message))
			}
		}
		return a.failStage(idx, stageIndex, fmt.Errorf("worker failed: %v %s", e, stderr.String()))
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, ev := range events {
		if ev.Type == "error" {
			a.jobs[idx].Stages[stageIndex].Status = "failed"
			a.jobs[idx].Status = "failed"
			a.save()
			a.emitJobUpdate(a.jobs[idx])
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
		a.jobs[idx].Status = "ready" // processing으로 남으면 delete/reset이 영원히 거부됨
	} else {
		a.jobs[idx].Status = "complete"
	}
	a.jobs[idx].Progress = (stageIndex + 1) * 100 / len(a.jobs[idx].Stages)
	a.save()
	a.emitJobUpdate(a.jobs[idx])
	return a.jobs[idx], nil
}
func (a *App) failStage(i, s int, cause error) (Job, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.jobs[i].Stages[s].Status = "failed"
	a.jobs[i].Status = "failed"
	a.jobs[i].Logs = append(a.jobs[i].Logs, LogEntry{time.Now().Format(time.RFC3339), a.jobs[i].Stages[s].ID, "error", cause.Error()})
	a.save()
	a.emitJobUpdate(a.jobs[i])
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
