package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func importFixtureJob(t *testing.T, a *App) Job {
	t.Helper()
	source := filepath.Join(t.TempDir(), "hero.png")
	pngHeader := append([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"), []byte("\x00\x00\x00\x10\x00\x00\x00\x20")...)
	if err := os.WriteFile(source, pngHeader, 0644); err != nil {
		t.Fatal(err)
	}
	job, err := a.importPath(source)
	if err != nil {
		t.Fatal(err)
	}
	return job
}

func TestDeleteJobRemovesRecordAndWorkspace(t *testing.T) {
	isolateConfig(t)
	a := NewApp()
	job := importFixtureJob(t, a)
	if _, err := os.Stat(job.Workspace); err != nil {
		t.Fatalf("workspace missing before delete: %v", err)
	}
	jobs, err := a.DeleteJob(job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 0 {
		t.Fatalf("job record not removed: %#v", jobs)
	}
	if _, err := os.Stat(job.Workspace); !os.IsNotExist(err) {
		t.Fatalf("workspace should be deleted, got: %v", err)
	}
	// Persistence check: a fresh App must not resurrect the job.
	b := NewApp()
	b.load()
	if len(b.jobs) != 0 {
		t.Fatalf("deleted job persisted: %#v", b.jobs)
	}
}

func TestDeleteJobRefusesOutsideProjectsRoot(t *testing.T) {
	isolateConfig(t)
	a := NewApp()
	job := importFixtureJob(t, a)
	outside := t.TempDir()
	marker := filepath.Join(outside, "keep.txt")
	if err := os.WriteFile(marker, []byte("keep"), 0644); err != nil {
		t.Fatal(err)
	}
	a.mu.Lock()
	a.jobs[a.findJobLocked(job.ID)].Workspace = outside
	a.mu.Unlock()
	if _, err := a.DeleteJob(job.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatalf("files outside projects root must survive delete: %v", err)
	}
}

func TestRenameJobPersists(t *testing.T) {
	isolateConfig(t)
	a := NewApp()
	job := importFixtureJob(t, a)
	if _, err := a.RenameJob(job.ID, "  "); err == nil {
		t.Fatal("expected empty-name rejection")
	}
	updated, err := a.RenameJob(job.ID, "  Knight v2 ")
	if err != nil {
		t.Fatal(err)
	}
	if updated.Name != "Knight v2" {
		t.Fatalf("name not trimmed/updated: %q", updated.Name)
	}
	b := NewApp()
	b.load()
	if b.jobs[0].Name != "Knight v2" {
		t.Fatalf("rename not persisted: %q", b.jobs[0].Name)
	}
}

func TestResetStageClearsDownstream(t *testing.T) {
	isolateConfig(t)
	a := NewApp()
	job := importFixtureJob(t, a)
	var err error
	for range pipeline() {
		if job, err = a.RunNextStage(job.ID); err != nil {
			t.Fatal(err)
		}
	}
	if job.Status != "complete" {
		t.Fatalf("fixture pipeline incomplete: %#v", job)
	}
	job, err = a.ResetStage(job.ID, "prepare")
	if err != nil {
		t.Fatal(err)
	}
	if job.Status != "ready" || job.Progress != 0 {
		t.Fatalf("bad reset state: status=%q progress=%d", job.Status, job.Progress)
	}
	if job.Stages[0].Status != "ready" {
		t.Fatalf("stage 0 should be ready: %#v", job.Stages[0])
	}
	for _, s := range job.Stages[1:] {
		if s.Status != "queued" {
			t.Fatalf("downstream stage not queued: %#v", s)
		}
	}
	if len(job.Artifacts) != 0 {
		t.Fatalf("artifacts should be cleared: %#v", job.Artifacts)
	}
	if _, err := os.Stat(filepath.Join(job.Workspace, "reconstruct")); !os.IsNotExist(err) {
		t.Fatalf("stage dir should be removed, got: %v", err)
	}
	// Reference image must survive a full reset.
	if _, err := os.Stat(job.Image); err != nil {
		t.Fatalf("reference image lost on reset: %v", err)
	}
}

func TestResetStageMidPipeline(t *testing.T) {
	isolateConfig(t)
	a := NewApp()
	job := importFixtureJob(t, a)
	var err error
	for range pipeline() {
		if job, err = a.RunNextStage(job.ID); err != nil {
			t.Fatal(err)
		}
	}
	job, err = a.ResetStage(job.ID, "rig")
	if err != nil {
		t.Fatal(err)
	}
	if job.Progress != 50 {
		t.Fatalf("progress should rewind to 50: %d", job.Progress)
	}
	for i, want := range []string{"done", "done", "done", "ready", "queued", "queued"} {
		if job.Stages[i].Status != want {
			t.Fatalf("stage %d = %q, want %q", i, job.Stages[i].Status, want)
		}
	}
	for _, artifact := range job.Artifacts {
		if artifact.Stage == "rig" || artifact.Stage == "motion" || artifact.Stage == "export" {
			t.Fatalf("downstream artifact survived reset: %#v", artifact)
		}
	}
	// Upstream artifacts must survive so the pipeline can resume.
	if len(job.Artifacts) != 3 {
		t.Fatalf("upstream artifacts lost: %#v", job.Artifacts)
	}
	if job, err = a.RunNextStage(job.ID); err != nil {
		t.Fatalf("pipeline cannot resume after reset: %v", err)
	}
	if job.Stages[3].Status != "done" {
		t.Fatalf("rig did not rerun: %#v", job.Stages[3])
	}
}

func TestExportGLBToPath(t *testing.T) {
	isolateConfig(t)
	a := NewApp()
	job := importFixtureJob(t, a)
	var err error
	for range pipeline() {
		if job, err = a.RunNextStage(job.ID); err != nil {
			t.Fatal(err)
		}
	}
	dst := filepath.Join(t.TempDir(), "out.glb")
	got, err := a.exportGLBToPath(job.ID, dst)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(got)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) < 20 || string(data[:4]) != "glTF" {
		t.Fatal("exported file is not a GLB")
	}
}

func TestExportGLBRequiresArtifact(t *testing.T) {
	isolateConfig(t)
	a := NewApp()
	job := importFixtureJob(t, a)
	if _, err := a.exportGLBToPath(job.ID, filepath.Join(t.TempDir(), "out.glb")); err == nil {
		t.Fatal("expected export rejection without GLB artifact")
	}
}

func TestImportReferenceDataCreatesReadyJob(t *testing.T) {
	isolateConfig(t)
	a := NewApp()
	png := append([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"), []byte("\x00\x00\x00\x10\x00\x00\x00\x20")...)
	job, err := a.ImportReferenceData("dropped hero.png", base64.StdEncoding.EncodeToString(png))
	if err != nil {
		t.Fatal(err)
	}
	if job.Name != "dropped hero" {
		t.Fatalf("name = %q", job.Name)
	}
	if job.Status != "ready" || job.ImageHash == "" {
		t.Fatalf("job not ready with provenance: %+v", job)
	}
	data, err := os.ReadFile(job.Image)
	if err != nil || len(data) != len(png) {
		t.Fatalf("workspace copy mismatch: %v len=%d", err, len(data))
	}
	sum := sha256.Sum256(png)
	if job.ImageHash != hex.EncodeToString(sum[:]) {
		t.Fatalf("hash mismatch: %s", job.ImageHash)
	}
	if !a.insideProjectsRoot(job.Workspace) {
		t.Fatalf("workspace outside projects root: %s", job.Workspace)
	}
}

func TestImportReferenceDataRejectsBadInput(t *testing.T) {
	isolateConfig(t)
	a := NewApp()
	if _, err := a.ImportReferenceData("evil.exe", base64.StdEncoding.EncodeToString([]byte("x"))); err == nil {
		t.Fatal("expected extension rejection")
	}
	if _, err := a.ImportReferenceData("a.png", "%%%not-base64%%%"); err == nil {
		t.Fatal("expected base64 rejection")
	}
	if _, err := a.ImportReferenceData("a.png", ""); err == nil {
		t.Fatal("expected empty payload rejection")
	}
	if len(a.ListJobs()) != 0 {
		t.Fatalf("rejected imports must not leave jobs: %#v", a.ListJobs())
	}
}

func TestRunNextStageLeavesJobManageable(t *testing.T) {
	isolateConfig(t)
	a := NewApp()
	job := importFixtureJob(t, a)
	updated, err := a.RunNextStage(job.ID)
	if err != nil {
		t.Fatal(err)
	}
	// 중간 stage 성공 후 job은 processing에 남으면 안 된다 (delete/reset 영구 거부 버그).
	if updated.Status != "ready" {
		t.Fatalf("status after mid-pipeline stage = %q, want ready", updated.Status)
	}
	if _, err := a.ResetStage(job.ID, "prepare"); err != nil {
		t.Fatalf("reset refused after stage success: %v", err)
	}
	if _, err := a.DeleteJob(job.ID); err != nil {
		t.Fatalf("delete refused after stage success: %v", err)
	}
}

func TestLoadRepairsStaleFailedJob(t *testing.T) {
	isolateConfig(t)
	a := NewApp()
	job := importFixtureJob(t, a)
	if _, err := a.RunNextStage(job.ID); err != nil {
		t.Fatal(err)
	}
	// 과거 버전처럼 stage는 정상(done/ready)인데 status만 failed로 저장된 상태를 재현
	a.mu.Lock()
	a.jobs[a.findJobLocked(job.ID)].Status = "failed"
	a.save()
	a.mu.Unlock()
	b := NewApp()
	b.load()
	i := b.findJobLocked(job.ID)
	if i < 0 {
		t.Fatal("job missing after reload")
	}
	if b.jobs[i].Status != "ready" {
		t.Fatalf("stale failed job not repaired: %q", b.jobs[i].Status)
	}
}
