package main

import (
	"bufio"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestPipelineShape(t *testing.T) {
	stages := pipeline()
	if len(stages) != 6 || stages[0].Status != "ready" || stages[5].ID != "export" {
		t.Fatalf("unexpected pipeline: %#v", stages)
	}
}

func TestImportPathCopiesAndHashes(t *testing.T) {
	config := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", config)
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
	config := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", config)
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

func TestReadArtifactRejectsUnregisteredPath(t *testing.T) {
	a := NewApp()
	if _, err := a.ReadArtifact(filepath.Join(t.TempDir(), "unknown.glb")); err == nil {
		t.Fatal("expected unregistered artifact rejection")
	}
}

func TestCompleteOfflinePipelineProducesAnimatedGLB(t *testing.T) {
	config := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", config)
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
	preview, err := a.ReadArtifact(last.Path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(preview, "data:model/gltf-binary;base64,") {
		t.Fatal("invalid preview data URL")
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
