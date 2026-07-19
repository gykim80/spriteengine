package main

import (
	"bufio"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// isolateConfig keeps tests away from the real user config and RunPod
// credentials so the offline pipeline path is always exercised.
func isolateConfig(t *testing.T) {
	t.Helper()
	t.Setenv("SPRITEENGINE_CONFIG_DIR", t.TempDir())
	t.Setenv("RUNPOD_ENDPOINT_ID", "")
	t.Setenv("RUNPOD_API_KEY", "")
}

func TestPipelineShape(t *testing.T) {
	stages := pipeline()
	if len(stages) != 6 || stages[0].Status != "ready" || stages[5].ID != "export" {
		t.Fatalf("unexpected pipeline: %#v", stages)
	}
}

func TestImportPathCopiesAndHashes(t *testing.T) {
	isolateConfig(t)
	source := filepath.Join(t.TempDir(), "hero.png")
	pngHeader := append([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"), []byte("\x00\x00\x00\x10\x00\x00\x00\x20")...)
	if err := os.WriteFile(source, pngHeader, 0644); err != nil {
		t.Fatal(err)
	}
	a := NewApp()
	job, err := a.importPath(source)
	if err != nil {
		t.Fatal(err)
	}
	if job.ImageHash == "" || job.Status != "ready" {
		t.Fatalf("missing provenance: %#v", job)
	}
	if _, err := os.Stat(job.Image); err != nil {
		t.Fatalf("image was not copied: %v", err)
	}
}

func TestRunNextStagePersistsArtifact(t *testing.T) {
	isolateConfig(t)
	source := filepath.Join(t.TempDir(), "hero.png")
	pngHeader := append([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"), []byte("\x00\x00\x00\x10\x00\x00\x00\x20")...)
	if err := os.WriteFile(source, pngHeader, 0644); err != nil {
		t.Fatal(err)
	}
	a := NewApp()
	job, err := a.importPath(source)
	if err != nil {
		t.Fatal(err)
	}
	job, err = a.RunNextStage(job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if job.Progress != 16 || job.Stages[0].Status != "done" || job.Stages[1].Status != "ready" {
		t.Fatalf("bad transition: %#v", job)
	}
	if len(job.Artifacts) != 1 || job.Artifacts[0].Kind != "reference" {
		t.Fatalf("artifact not persisted: %#v", job.Artifacts)
	}
	if _, err := os.Stat(job.Artifacts[0].Path); err != nil {
		t.Fatalf("artifact missing: %v", err)
	}
}

// TestSaveMergesExternallyRegisteredJobs: 앱 실행 중 외부 도구
// (tools/matrix/run_character.py 등)가 jobs.json에 직접 등록한 job이
// 앱의 다음 save에서 유실되지 않고 병합되어야 한다.
// 회귀: matrix2-dog2가 앱 내 프로젝트 생성 시 덮어쓰기로 사라진 버그.
func TestSaveMergesExternallyRegisteredJobs(t *testing.T) {
	isolateConfig(t)
	a := NewApp()
	inApp, err := a.CreateJob("in-app")
	if err != nil {
		t.Fatal(err)
	}

	// 외부 도구가 디스크의 jobs.json에 직접 항목을 추가한 상황을 재현
	var disk []Job
	b, err := os.ReadFile(a.dataPath())
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &disk); err != nil {
		t.Fatal(err)
	}
	external := Job{ID: "20990101-000000-000001", Name: "external-dog", Status: "done", Stages: pipeline()}
	disk = append([]Job{external}, disk...)
	b, _ = json.MarshalIndent(disk, "", "  ")
	if err := os.WriteFile(a.dataPath(), b, 0644); err != nil {
		t.Fatal(err)
	}

	// 앱이 다시 저장해도(새 job 생성) 외부 항목이 살아남아야 한다
	if _, err := a.CreateJob("second-in-app"); err != nil {
		t.Fatal(err)
	}
	b, _ = os.ReadFile(a.dataPath())
	var after []Job
	if err := json.Unmarshal(b, &after); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, j := range after {
		if j.ID == external.ID {
			found = true
		}
	}
	if !found {
		t.Fatalf("externally registered job was clobbered by save: %d jobs", len(after))
	}
	// 최신순(ID 내림차순) 정렬 유지 — 외부 항목(2099년)이 맨 위여야 한다
	if after[0].ID != external.ID {
		t.Fatalf("expected newest-first ordering, got %s first", after[0].ID)
	}

	// 삭제한 job은 병합으로 부활하면 안 된다
	if _, err := a.DeleteJob(external.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := a.RenameJob(inApp.ID, "renamed"); err != nil {
		t.Fatal(err)
	}
	b, _ = os.ReadFile(a.dataPath())
	var final []Job
	if err := json.Unmarshal(b, &final); err != nil {
		t.Fatal(err)
	}
	for _, j := range final {
		if j.ID == external.ID {
			t.Fatal("deleted job was resurrected by save merge")
		}
	}
}

// TestListJobsPicksUpExternallyRegisteredJobs: 외부 도구가 등록한 job이
// 앱 재시작 없이 ListJobs(프로젝트 목록 새로고침)만으로 나타나야 한다.
func TestListJobsPicksUpExternallyRegisteredJobs(t *testing.T) {
	isolateConfig(t)
	a := NewApp()
	if _, err := a.CreateJob("in-app"); err != nil {
		t.Fatal(err)
	}
	var disk []Job
	b, err := os.ReadFile(a.dataPath())
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &disk); err != nil {
		t.Fatal(err)
	}
	external := Job{ID: "20990101-000000-000002", Name: "external-live", Status: "done", Stages: pipeline()}
	disk = append([]Job{external}, disk...)
	b, _ = json.MarshalIndent(disk, "", "  ")
	if err := os.WriteFile(a.dataPath(), b, 0644); err != nil {
		t.Fatal(err)
	}
	jobs := a.ListJobs()
	if len(jobs) != 2 || jobs[0].ID != external.ID {
		t.Fatalf("external job not visible via ListJobs: %#v", jobs)
	}
}

func TestReadArtifactRejectsUnregisteredPath(t *testing.T) {
	a := NewApp()
	if _, err := a.ReadArtifact(filepath.Join(t.TempDir(), "unknown.glb")); err == nil {
		t.Fatal("expected unregistered artifact rejection")
	}
}

func TestCompleteOfflinePipelineProducesAnimatedGLB(t *testing.T) {
	isolateConfig(t)
	source := filepath.Join(t.TempDir(), "hero.png")
	pngHeader := append([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"), []byte("\x00\x00\x00\x10\x00\x00\x00\x20")...)
	if err := os.WriteFile(source, pngHeader, 0644); err != nil {
		t.Fatal(err)
	}
	a := NewApp()
	job, err := a.importPath(source)
	if err != nil {
		t.Fatal(err)
	}
	for range pipeline() {
		job, err = a.RunNextStage(job.ID)
		if err != nil {
			t.Fatal(err)
		}
	}
	if job.Status != "complete" || job.Progress != 100 {
		t.Fatalf("pipeline incomplete: %#v", job)
	}
	last := job.Artifacts[len(job.Artifacts)-1]
	if last.Kind != "package" || filepath.Ext(last.Path) != ".glb" {
		t.Fatalf("missing final GLB: %#v", last)
	}
	data, err := os.ReadFile(last.Path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) < 20 || string(data[:4]) != "glTF" {
		t.Fatal("final artifact is not GLB")
	}
}

func TestBaselineWorkerProtocol(t *testing.T) {
	workspace := t.TempDir()
	source := filepath.Join(workspace, "input.png")
	pngHeader := append([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"), []byte("\x00\x00\x00\x10\x00\x00\x00\x20")...)
	if err := os.WriteFile(source, pngHeader, 0644); err != nil {
		t.Fatal(err)
	}
	request, _ := json.Marshal(map[string]any{"type": "run", "jobId": "test", "stage": "prepare", "workspace": workspace, "input": source})
	cmd := exec.Command("python3", filepath.Join("workers", "baseline_worker.py"))
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	_, _ = stdin.Write(append(request, '\n'))
	_ = stdin.Close()
	scanner := bufio.NewScanner(stdout)
	foundDone := false
	for scanner.Scan() {
		var msg map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &msg); err != nil {
			t.Fatal(err)
		}
		if msg["type"] == "done" {
			foundDone = true
		}
	}
	if err := cmd.Wait(); err != nil {
		t.Fatal(err)
	}
	if !foundDone {
		t.Fatal("worker emitted no done event")
	}
}
