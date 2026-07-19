package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunPodAuthenticationHeader(t *testing.T) {
	var gotAuth, gotPath string
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth, gotPath = r.Header.Get("Authorization"), r.URL.Path
		_, _ = io.WriteString(w, `{"workers":{"idle":1}}`)
	}))
	defer srv.Close()
	c := &runPodClient{baseURL: srv.URL, endpointID: "endpoint-1", apiKey: "secret-token", http: srv.Client()}
	if _, _, err := c.request(context.Background(), http.MethodGet, "health", nil); err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer secret-token" {
		t.Fatalf("unexpected auth header %q", gotAuth)
	}
	if gotPath != "/endpoint-1/health" {
		t.Fatalf("unexpected path %q", gotPath)
	}
}

func TestRunPod401IsActionableAndDoesNotLeakKey(t *testing.T) {
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":"bad key"}`)
	}))
	defer srv.Close()
	c := &runPodClient{baseURL: srv.URL, endpointID: "endpoint", apiKey: "do-not-leak", http: srv.Client()}
	_, _, err := c.request(context.Background(), http.MethodGet, "health", nil)
	if err == nil || !strings.Contains(err.Error(), "401") || !strings.Contains(err.Error(), "새 API key") {
		t.Fatalf("expected 401 error, got %v", err)
	}
	if strings.Contains(err.Error(), c.apiKey) {
		t.Fatal("credential leaked in error")
	}
}

func TestNormalizeRunPodAPIKey(t *testing.T) {
	cases := map[string]string{
		" secret-token ":                "secret-token",
		"Bearer secret-token":           "secret-token",
		"RUNPOD_API_KEY='secret-token'": "secret-token",
		`RUNPOD_API_KEY="secret-token"`: "secret-token",
		"Bearer RUNPOD_SECRET":          "RUNPOD_SECRET",
		"••••••••":                      "",
		"********":                      "",
	}
	for input, want := range cases {
		if got := normalizeRunPodAPIKey(input); got != want {
			t.Fatalf("normalize %q = %q, want %q", input, got, want)
		}
	}
}

func TestSaveAndTestDoesNotPersistRejectedKey(t *testing.T) {
	t.Setenv("RUNPOD_API_KEY", "")
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", home)
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()
	a := NewApp()
	_, err := a.SaveAndTestRunPodConfig("endpoint", "rejected-key", srv.URL)
	if err == nil {
		t.Fatal("expected authentication failure")
	}
	if data, readErr := os.ReadFile(a.runPodConfigPath()); readErr == nil && strings.Contains(string(data), "rejected-key") {
		t.Fatal("rejected credential was persisted")
	}
}

func TestSaveRunPodConfigPermissionsAndRedaction(t *testing.T) {
	t.Setenv("RUNPOD_API_KEY", "")
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", home)
	a := NewApp()
	cfg, err := a.SaveRunPodConfig("endpoint", "private-key", "https://api.runpod.ai/v2")
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.Configured || cfg.KeySource != "secure config" {
		t.Fatalf("unexpected public config: %+v", cfg)
	}
	info, err := os.Stat(a.runPodConfigPath())
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm()&0077 != 0 {
		t.Fatalf("credential file permissions too broad: %o", info.Mode().Perm())
	}
	data, _ := os.ReadFile(a.runPodConfigPath())
	if !strings.Contains(string(data), "private-key") {
		t.Fatal("key was not persisted")
	}
	public, _ := os.ReadFile(a.dataPath())
	if strings.Contains(string(public), "private-key") {
		t.Fatal("key leaked into project manifest")
	}
}

// TestRunNextStageMotionRoutesToHYMotion: RunPod이 설정돼 있으면 파이프라인
// motion 단계가 로컬 passthrough 대신 HY-Motion 경로(모의 서버 → hy_motion.json
// → SMPL 리타겟 베이킹)를 타고, 기본 클립 세트가 애니메이션으로 등록되는지 검증.
func TestRunNextStageMotionRoutesToHYMotion(t *testing.T) {
	cfgDir := t.TempDir()
	t.Setenv("SPRITEENGINE_CONFIG_DIR", cfgDir)
	t.Setenv("RUNPOD_ENDPOINT_ID", "")
	t.Setenv("RUNPOD_API_KEY", "")

	a := NewApp()
	source := filepath.Join(t.TempDir(), "hero.png")
	pngHeader := append([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"), []byte("\x00\x00\x00\x10\x00\x00\x00\x20")...)
	if err := os.WriteFile(source, pngHeader, 0644); err != nil {
		t.Fatal(err)
	}
	job, err := a.importPath(source)
	if err != nil {
		t.Fatal(err)
	}
	// RunPod 미설정 상태로 prepare→reconstruct→retopo→rig까지 오프라인 실행.
	for i := 0; i < 4; i++ {
		if job, err = a.RunNextStage(job.ID); err != nil {
			t.Fatal(err)
		}
	}
	if job.Stages[3].Status != "done" {
		t.Fatalf("rig stage did not finish: %#v", job.Stages)
	}

	// 모의 HY-Motion 서버: /run이 요청된 프롬프트마다 identity 쿼터니언
	// 3프레임 모션을 담아 즉시 COMPLETED로 응답한다.
	joints := []string{
		"Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
		"L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
		"R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
		"L_Wrist", "R_Wrist",
	}
	var gotPrompts []MotionPrompt
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/run") {
			http.NotFound(w, r)
			return
		}
		var req struct {
			Input struct {
				Task    string         `json:"task"`
				Prompts []MotionPrompt `json:"prompts"`
			} `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Input.Task != "motion" {
			t.Errorf("unexpected RunPod payload: task=%q err=%v", req.Input.Task, err)
		}
		gotPrompts = req.Input.Prompts
		frame := make([][]float64, len(joints))
		for i := range frame {
			frame[i] = []float64{0, 0, 0, 1}
		}
		quats := [][][]float64{frame, frame, frame}
		trans := [][]float64{{0, 0.9, 0}, {0, 0.95, 0}, {0, 0.9, 0}}
		motions := make([]map[string]any, 0, len(req.Input.Prompts))
		for _, p := range req.Input.Prompts {
			motions = append(motions, map[string]any{
				"id": p.ID, "text": p.Text, "fps": 30,
				"joints": joints, "quats": quats, "trans": trans,
			})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "mock-motion", "status": "COMPLETED",
			"output": map[string]any{"model": "HY-Motion-1.0-Lite", "motions": motions, "errors": map[string]string{}},
		})
	}))
	defer srv.Close()
	creds := fmt.Sprintf(`{"endpointId":"ep-test","apiKey":"rpa_mock_key_1234","baseUrl":%q}`, srv.URL)
	if err := os.WriteFile(filepath.Join(cfgDir, "runpod.json"), []byte(creds), 0600); err != nil {
		t.Fatal(err)
	}

	job, err = a.RunNextStage(job.ID) // motion 단계
	if err != nil {
		t.Fatal(err)
	}
	if len(gotPrompts) != len(defaultMotionPrompts()) {
		t.Fatalf("expected default prompt set, got %d prompts", len(gotPrompts))
	}
	if job.Stages[4].Status != "done" || !strings.Contains(job.Stages[4].Detail, "HY-Motion") {
		t.Fatalf("motion stage did not route to HY-Motion: %#v", job.Stages[4])
	}
	last := job.Artifacts[len(job.Artifacts)-1]
	if last.Stage != "motion" || last.Metrics["adapter"] != "hy-motion-retarget" {
		t.Fatalf("expected hy-motion-retarget artifact, got %#v", last)
	}
	if clips, _ := last.Metrics["animations"].(float64); int(clips) != len(defaultMotionPrompts()) {
		t.Fatalf("expected %d baked clips, got %v", len(defaultMotionPrompts()), last.Metrics["animations"])
	}
	if job.Stages[5].Status != "ready" {
		t.Fatalf("export stage should be ready: %#v", job.Stages[5])
	}
}

// 30개 프롬프트는 RunPod handler의 job당 20개 제한을 넘으므로 15+15 두 배치로
// 나뉘어 전송되고, 응답이 하나의 hy_motion.json으로 병합돼 30클립이 베이킹돼야 한다.
func TestRunPodGenerateMotionBatchesOver20(t *testing.T) {
	cfgDir := t.TempDir()
	t.Setenv("SPRITEENGINE_CONFIG_DIR", cfgDir)
	t.Setenv("RUNPOD_ENDPOINT_ID", "")
	t.Setenv("RUNPOD_API_KEY", "")

	a := NewApp()
	source := filepath.Join(t.TempDir(), "hero.png")
	pngHeader := append([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"), []byte("\x00\x00\x00\x10\x00\x00\x00\x20")...)
	if err := os.WriteFile(source, pngHeader, 0644); err != nil {
		t.Fatal(err)
	}
	job, err := a.importPath(source)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4; i++ { // prepare→reconstruct→retopo→rig 오프라인 실행
		if job, err = a.RunNextStage(job.ID); err != nil {
			t.Fatal(err)
		}
	}

	joints := []string{
		"Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
		"L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
		"R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
		"L_Wrist", "R_Wrist",
	}
	var batchSizes []int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/run") {
			http.NotFound(w, r)
			return
		}
		var req struct {
			Input struct {
				Prompts []MotionPrompt `json:"prompts"`
			} `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode payload: %v", err)
		}
		if len(req.Input.Prompts) > 20 {
			t.Errorf("batch exceeds handler limit: %d prompts", len(req.Input.Prompts))
		}
		// 워커의 렌더링 정상성 게이트(deformation 검사)는 항등/중복 클립을
		// 실패시키므로, 클립마다 상이한 소각도 L_Shoulder 회전을 넣어 최소한의
		// 유효 모션을 만든다 (배칭 검증이라는 테스트 의도는 그대로).
		// base는 전역 클립 인덱스 기반이어야 배치 간(clip01 vs clip16) 중복이 없다.
		offset := 0
		for _, s := range batchSizes {
			offset += s
		}
		batchSizes = append(batchSizes, len(req.Input.Prompts))
		trans := [][]float64{{0, 0.9, 0}, {0, 0.95, 0}, {0, 0.9, 0}}
		motions := make([]map[string]any, 0, len(req.Input.Prompts))
		for pi, p := range req.Input.Prompts {
			base := 0.05 + 0.01*float64(offset+pi)
			quats := make([][][]float64, 3)
			for f := 0; f < 3; f++ {
				frame := make([][]float64, len(joints))
				for j := range frame {
					frame[j] = []float64{0, 0, 0, 1}
				}
				half := base * float64(f) / 2
				frame[16] = []float64{math.Sin(half), 0, 0, math.Cos(half)} // L_Shoulder
				quats[f] = frame
			}
			motions = append(motions, map[string]any{
				"id": p.ID, "text": p.Text, "fps": 30,
				"joints": joints, "quats": quats, "trans": trans,
			})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "mock-motion", "status": "COMPLETED",
			"output": map[string]any{"model": "HY-Motion-1.0-Lite", "motions": motions, "errors": map[string]string{}},
		})
	}))
	defer srv.Close()
	creds := fmt.Sprintf(`{"endpointId":"ep-test","apiKey":"rpa_mock_key_1234","baseUrl":%q}`, srv.URL)
	if err := os.WriteFile(filepath.Join(cfgDir, "runpod.json"), []byte(creds), 0600); err != nil {
		t.Fatal(err)
	}

	prompts := make([]MotionPrompt, 30)
	for i := range prompts {
		prompts[i] = MotionPrompt{ID: fmt.Sprintf("clip%02d", i+1),
			Text: fmt.Sprintf("motion variant %d", i+1), Duration: 4}
	}
	result, err := a.RunPodGenerateMotion(job.ID, prompts)
	if err != nil {
		t.Fatal(err)
	}
	if len(batchSizes) != 2 || batchSizes[0] != 15 || batchSizes[1] != 15 {
		t.Fatalf("expected balanced 15+15 batches, got %v", batchSizes)
	}
	if result.Clips != 30 {
		t.Fatalf("expected 30 baked clips, got %d", result.Clips)
	}
	// 병합된 hy_motion.json에 30개 모션이 순서대로 들어있는지 확인
	raw, err := os.ReadFile(filepath.Join(job.Workspace, "motion", "hy_motion.json"))
	if err != nil {
		t.Fatal(err)
	}
	var mergedFile struct {
		Model   string `json:"model"`
		Motions []struct {
			ID string `json:"id"`
		} `json:"motions"`
	}
	if err := json.Unmarshal(raw, &mergedFile); err != nil {
		t.Fatal(err)
	}
	if mergedFile.Model != "HY-Motion-1.0-Lite" || len(mergedFile.Motions) != 30 {
		t.Fatalf("merged hy_motion.json invalid: model=%q motions=%d", mergedFile.Model, len(mergedFile.Motions))
	}
	if mergedFile.Motions[0].ID != "clip01" || mergedFile.Motions[29].ID != "clip30" {
		t.Fatalf("merged motion order broken: first=%s last=%s", mergedFile.Motions[0].ID, mergedFile.Motions[29].ID)
	}
}

func TestChunkMotionPromptsBalance(t *testing.T) {
	mk := func(n int) []MotionPrompt {
		out := make([]MotionPrompt, n)
		for i := range out {
			out[i] = MotionPrompt{ID: fmt.Sprintf("p%d", i)}
		}
		return out
	}
	cases := []struct {
		n    int
		want []int
	}{
		{5, []int{5}}, {20, []int{20}}, {21, []int{11, 10}},
		{30, []int{15, 15}}, {45, []int{15, 15, 15}}, {60, []int{20, 20, 20}},
	}
	for _, c := range cases {
		got := chunkMotionPrompts(mk(c.n), 20)
		sizes := make([]int, len(got))
		total := 0
		for i, b := range got {
			sizes[i] = len(b)
			total += len(b)
		}
		if total != c.n || fmt.Sprint(sizes) != fmt.Sprint(c.want) {
			t.Errorf("n=%d: got %v want %v", c.n, sizes, c.want)
		}
	}
}
