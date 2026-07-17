package main

import (
	"image"
	"image/color"
	"image/png"
	"os"
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
