package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestLiveRunPodReconstruct exercises the exact production path (config load,
// /run submission, /status polling, GLB validation) against the real RunPod
// endpoint. It costs GPU time, so it only runs when explicitly requested:
//
//	SPRITEENGINE_LIVE_E2E=1 go test -run TestLiveRunPodReconstruct -timeout 30m
func TestLiveRunPodReconstruct(t *testing.T) {
	if os.Getenv("SPRITEENGINE_LIVE_E2E") != "1" {
		t.Skip("set SPRITEENGINE_LIVE_E2E=1 to run the live RunPod E2E test")
	}
	a := NewApp()
	if !a.GetRunPodConfig().Configured {
		t.Fatal("RunPod is not configured on this machine")
	}

	// A simple high-contrast subject on white, centered — enough for the
	// preprocessor to find a foreground object.
	size := 256
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			dx, dy := x-size/2, y-size/2
			if dx*dx+dy*dy < (size/3)*(size/3) {
				img.Set(x, y, color.RGBA{40, 90, 200, 255})
			} else {
				img.Set(x, y, color.RGBA{255, 255, 255, 255})
			}
		}
	}
	workspace := t.TempDir()
	input := filepath.Join(workspace, "input.png")
	f, err := os.Create(input)
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
	f.Close()

	artifact, err := a.runPodReconstruct(input, workspace)
	if err != nil {
		t.Fatalf("live reconstruction failed: %v", err)
	}
	data, err := os.ReadFile(artifact.Path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) < 1024 || string(data[:4]) != "glTF" {
		t.Fatalf("artifact is not a GLB (%d bytes)", len(data))
	}
	t.Logf("live GLB: %s (%d bytes, metrics=%v)", artifact.Path, len(data), artifact.Metrics)
}

// runWorkerStage는 embedded baseline worker의 한 stage를 서브프로세스로 실행해
// artifact 경로를 돌려준다 (bakeMotionGLB와 동일한 JSON Lines 프로토콜).
func runWorkerStage(t *testing.T, worker, stage, workspace, input string) string {
	t.Helper()
	req, _ := json.Marshal(map[string]any{"type": "run", "jobId": "live-e2e", "stage": stage,
		"workspace": workspace, "input": input, "adapter": "local-baseline"})
	cmd := exec.Command("python3", worker)
	cmd.Stdin = bytes.NewReader(append(req, '\n'))
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	path := ""
	scan := bufio.NewScanner(stdout)
	for scan.Scan() {
		var ev workerEvent
		if json.Unmarshal(scan.Bytes(), &ev) != nil {
			continue
		}
		if ev.Type == "error" {
			t.Fatalf("%s stage error: %s", stage, ev.Message)
		}
		if ev.Type == "artifact" {
			path = ev.Path
		}
	}
	if err := cmd.Wait(); err != nil {
		t.Fatalf("%s worker failed: %v %s", stage, err, stderr.String())
	}
	if path == "" {
		t.Fatalf("%s produced no artifact", stage)
	}
	return path
}

// TestLiveRunPodGenerateMotion exercises the production natural-language motion
// path end to end: config load → HY-Motion /run+poll → hy_motion.json →
// baseline worker SMPL retarget bake → artifact registration. jobs.json is
// isolated in a temp config dir; only the real runpod.json credentials are
// copied in, so the developer's project list is never touched.
//
//	SPRITEENGINE_LIVE_E2E=1 go test -run TestLiveRunPodGenerateMotion -timeout 30m
func TestLiveRunPodGenerateMotion(t *testing.T) {
	if os.Getenv("SPRITEENGINE_LIVE_E2E") != "1" {
		t.Skip("set SPRITEENGINE_LIVE_E2E=1 to run the live RunPod E2E test")
	}
	userCfg, err := os.UserConfigDir()
	if err != nil {
		t.Fatal(err)
	}
	creds, err := os.ReadFile(filepath.Join(userCfg, "SpriteEngine", "runpod.json"))
	if err != nil {
		t.Skipf("RunPod is not configured on this machine: %v", err)
	}
	cfgDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(cfgDir, "runpod.json"), creds, 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("SPRITEENGINE_CONFIG_DIR", cfgDir)

	a := NewApp()
	if !a.GetRunPodConfig().Configured {
		t.Fatal("RunPod config did not load from the isolated dir")
	}
	job, err := a.CreateJob("motion-live-e2e")
	if err != nil {
		t.Fatal(err)
	}
	worker, err := a.workerPath()
	if err != nil {
		t.Fatal(err)
	}
	// 로컬(procedural) 메시 → retopo → auto-rig로 스킨 있는 GLB를 준비한다.
	mesh := runWorkerStage(t, worker, "reconstruct", job.Workspace, "")
	clean := runWorkerStage(t, worker, "retopo", job.Workspace, mesh)
	rigged := runWorkerStage(t, worker, "rig", job.Workspace, clean)
	a.mu.Lock()
	for i := range a.jobs {
		if a.jobs[i].ID == job.ID {
			a.jobs[i].Artifacts = append(a.jobs[i].Artifacts, Artifact{Stage: "rig", Kind: "rigged-model", Path: rigged})
		}
	}
	a.mu.Unlock()

	result, err := a.RunPodGenerateMotion(job.ID, []MotionPrompt{
		{ID: "wave", Text: "a person waves hello with the right hand", Duration: 4},
	})
	if err != nil {
		t.Fatalf("live motion generation failed: %v", err)
	}
	if result.Clips < 1 {
		t.Fatalf("expected at least 1 baked clip, got %d (errors=%v)", result.Clips, result.Errors)
	}
	data, err := os.ReadFile(result.Path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) < 1024 || string(data[:4]) != "glTF" {
		t.Fatalf("baked artifact is not a GLB (%d bytes)", len(data))
	}
	found := false
	for _, j := range a.ListJobs() {
		if j.ID != job.ID {
			continue
		}
		for _, art := range j.Artifacts {
			if art.Stage == "motion" && art.Path == result.Path {
				found = true
			}
		}
	}
	if !found {
		t.Fatal("motion artifact was not registered on the job")
	}
	t.Logf("live motion GLB: %s (%d bytes, clips=%d, model=%s)", result.Path, len(data), result.Clips, result.Model)
}
