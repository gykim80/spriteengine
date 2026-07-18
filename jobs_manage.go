package main

import (
	"encoding/base64"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"time"

	wailsruntime "github.com/wailsapp/wails/v2/pkg/runtime"
)

// jobIsRunning reports whether a job currently has an in-flight stage. Delete
// and reset are refused in that state because RunNextStage keeps a slice index
// across an unlock and mutating the slice underneath it would corrupt state.
func jobIsRunning(j Job) bool {
	if j.Status == "processing" {
		return true
	}
	for _, s := range j.Stages {
		if s.Status == "running" {
			return true
		}
	}
	return false
}

func (a *App) findJobLocked(id string) int {
	for i := range a.jobs {
		if a.jobs[i].ID == id {
			return i
		}
	}
	return -1
}

// insideProjectsRoot guards filesystem removal: only paths strictly inside
// <configRoot>/projects may ever be deleted by the app.
func (a *App) insideProjectsRoot(path string) bool {
	if strings.TrimSpace(path) == "" {
		return false
	}
	root := filepath.Clean(filepath.Join(a.rootPath(), "projects"))
	clean := filepath.Clean(path)
	return clean != root && strings.HasPrefix(clean, root+string(filepath.Separator))
}

// DeleteJob removes the job record and, when the workspace is safely contained
// in the projects root, its files. Returns the updated job list.
func (a *App) DeleteJob(id string) ([]Job, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	i := a.findJobLocked(id)
	if i < 0 {
		return nil, errors.New("job not found")
	}
	if jobIsRunning(a.jobs[i]) {
		return nil, errors.New("job is processing; wait for the stage to finish")
	}
	workspace := a.jobs[i].Workspace
	a.jobs = append(a.jobs[:i], a.jobs[i+1:]...)
	a.save()
	if a.insideProjectsRoot(workspace) {
		_ = os.RemoveAll(workspace)
	}
	out := make([]Job, len(a.jobs))
	copy(out, a.jobs)
	return out, nil
}

// RenameJob updates the display name of a project.
func (a *App) RenameJob(id, name string) (Job, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return Job{}, errors.New("name is required")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	i := a.findJobLocked(id)
	if i < 0 {
		return Job{}, errors.New("job not found")
	}
	a.jobs[i].Name = name
	a.save()
	return a.jobs[i], nil
}

// ResetStage rewinds the pipeline to the given stage: the stage becomes ready,
// every downstream stage returns to queued, and their artifacts are removed.
func (a *App) ResetStage(id, stageID string) (Job, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	i := a.findJobLocked(id)
	if i < 0 {
		return Job{}, errors.New("job not found")
	}
	if jobIsRunning(a.jobs[i]) {
		return Job{}, errors.New("job is processing; wait for the stage to finish")
	}
	si := -1
	for s := range a.jobs[i].Stages {
		if a.jobs[i].Stages[s].ID == stageID {
			si = s
			break
		}
	}
	if si < 0 {
		return Job{}, errors.New("stage not found")
	}
	template := pipeline()
	reset := map[string]bool{}
	for s := si; s < len(a.jobs[i].Stages); s++ {
		if s == si {
			a.jobs[i].Stages[s].Status = "ready"
		} else {
			a.jobs[i].Stages[s].Status = "queued"
		}
		a.jobs[i].Stages[s].Detail = template[s].Detail
		reset[a.jobs[i].Stages[s].ID] = true
	}
	if si == 0 && a.jobs[i].ImageHash != "" {
		a.jobs[i].Stages[0].Detail = "Reference secured · SHA-256 " + a.jobs[i].ImageHash[:8]
	}
	kept := a.jobs[i].Artifacts[:0]
	for _, artifact := range a.jobs[i].Artifacts {
		if !reset[artifact.Stage] {
			kept = append(kept, artifact)
		}
	}
	a.jobs[i].Artifacts = kept
	if a.insideProjectsRoot(a.jobs[i].Workspace) {
		for stage := range reset {
			_ = os.RemoveAll(filepath.Join(a.jobs[i].Workspace, stage))
		}
	}
	a.jobs[i].Progress = si * 100 / len(a.jobs[i].Stages)
	if a.jobs[i].Image == "" {
		a.jobs[i].Status = "draft"
	} else {
		a.jobs[i].Status = "ready"
	}
	a.jobs[i].Logs = append(a.jobs[i].Logs, LogEntry{time.Now().Format(time.RFC3339), stageID, "info", "Stage reset · downstream invalidated"})
	a.save()
	return a.jobs[i], nil
}

// exportGLBToPath copies the newest GLB artifact of the job to dst after
// validating the glTF magic. Split from ExportFinalGLB so tests can run it
// without a native save dialog.
func (a *App) exportGLBToPath(id, dst string) (string, error) {
	a.mu.Lock()
	i := a.findJobLocked(id)
	if i < 0 {
		a.mu.Unlock()
		return "", errors.New("job not found")
	}
	src := ""
	for k := len(a.jobs[i].Artifacts) - 1; k >= 0; k-- {
		if strings.HasSuffix(strings.ToLower(a.jobs[i].Artifacts[k].Path), ".glb") {
			src = a.jobs[i].Artifacts[k].Path
			break
		}
	}
	a.mu.Unlock()
	if src == "" {
		return "", errors.New("no GLB artifact yet; run the pipeline first")
	}
	data, err := os.ReadFile(src)
	if err != nil {
		return "", err
	}
	if len(data) < 4 || string(data[:4]) != "glTF" {
		return "", errors.New("artifact is not a GLB")
	}
	if err := os.WriteFile(dst, data, 0644); err != nil {
		return "", err
	}
	return dst, nil
}

// ExportFinalGLB asks the user for a destination and writes the newest GLB.
func (a *App) ExportFinalGLB(id string) (string, error) {
	if a.ctx == nil {
		return "", errors.New("application is not ready")
	}
	a.mu.Lock()
	i := a.findJobLocked(id)
	name := "character"
	if i >= 0 && strings.TrimSpace(a.jobs[i].Name) != "" {
		name = a.jobs[i].Name
	}
	a.mu.Unlock()
	if i < 0 {
		return "", errors.New("job not found")
	}
	dst, err := wailsruntime.SaveFileDialog(a.ctx, wailsruntime.SaveDialogOptions{Title: "Export GLB", DefaultFilename: name + ".glb", Filters: []wailsruntime.FileFilter{{DisplayName: "glTF Binary (*.glb)", Pattern: "*.glb"}}})
	if err != nil {
		return "", err
	}
	if dst == "" {
		return "", errors.New("cancelled")
	}
	return a.exportGLBToPath(id, dst)
}

// ReadJobImage serves the imported reference image as a data URI for card
// thumbnails. Only the registered job image path is readable.
func (a *App) ReadJobImage(id string) (string, error) {
	a.mu.Lock()
	i := a.findJobLocked(id)
	path := ""
	if i >= 0 {
		path = a.jobs[i].Image
	}
	a.mu.Unlock()
	if i < 0 {
		return "", errors.New("job not found")
	}
	if path == "" {
		return "", errors.New("job has no reference image")
	}
	mime := ""
	switch strings.ToLower(filepath.Ext(path)) {
	case ".png":
		mime = "image/png"
	case ".jpg", ".jpeg":
		mime = "image/jpeg"
	case ".webp":
		mime = "image/webp"
	default:
		return "", errors.New("unsupported image type")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if len(data) > 20*1024*1024 {
		return "", errors.New("image exceeds preview limit")
	}
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data), nil
}
