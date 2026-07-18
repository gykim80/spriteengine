package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// RunPodConfig is safe to return to the UI: the API key is never serialized.
type RunPodConfig struct {
	EndpointID string `json:"endpointId"`
	BaseURL    string `json:"baseUrl"`
	Configured bool   `json:"configured"`
	KeySource  string `json:"keySource"`
}

type runPodDiskConfig struct {
	EndpointID string `json:"endpointId"`
	BaseURL    string `json:"baseUrl"`
	APIKey     string `json:"apiKey,omitempty"`
}

type RunPodStatus struct {
	OK         bool   `json:"ok"`
	EndpointID string `json:"endpointId"`
	Message    string `json:"message"`
	Workers    int    `json:"workers"`
}

func normalizeRunPodAPIKey(raw string) string {
	key := strings.TrimSpace(raw)
	// Reject masked values shown by dashboards/password managers. Sending these
	// otherwise produces a confusing 401 even though the field looks populated.
	if strings.Contains(key, "••••") || strings.Contains(key, "****") {
		return ""
	}
	// Accept common copy/paste forms without sending a malformed double-Bearer header.
	if i := strings.Index(key, "="); strings.HasPrefix(strings.ToUpper(key), "RUNPOD_API_KEY=") && i >= 0 {
		key = strings.TrimSpace(key[i+1:])
	}
	key = strings.Trim(key, "\"'")
	if strings.HasPrefix(strings.ToLower(key), "bearer ") {
		key = strings.TrimSpace(key[len("bearer "):])
	}
	return key
}

type runPodClient struct {
	baseURL    string
	endpointID string
	apiKey     string
	http       *http.Client
}

func normalizeRunPodBaseURL(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		raw = "https://api.runpod.ai/v2"
	}
	u, err := url.Parse(raw)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return "", errors.New("RunPod base URL is invalid")
	}
	if u.Scheme != "https" && u.Hostname() != "127.0.0.1" && u.Hostname() != "localhost" {
		return "", errors.New("RunPod base URL must use HTTPS")
	}
	return strings.TrimRight(raw, "/"), nil
}

func (a *App) runPodConfigPath() string { return filepath.Join(a.rootPath(), "runpod.json") }

func (a *App) loadRunPodDiskConfig() (runPodDiskConfig, error) {
	var cfg runPodDiskConfig
	data, err := os.ReadFile(a.runPodConfigPath())
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return cfg, err
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, err
	}
	return cfg, nil
}

func (a *App) runPodClient() (*runPodClient, error) {
	cfg, err := a.loadRunPodDiskConfig()
	if err != nil {
		return nil, err
	}
	key := normalizeRunPodAPIKey(cfg.APIKey)
	if key == "" {
		key = normalizeRunPodAPIKey(os.Getenv("RUNPOD_API_KEY"))
	}
	endpoint := strings.TrimSpace(os.Getenv("RUNPOD_ENDPOINT_ID"))
	if endpoint == "" {
		endpoint = strings.TrimSpace(cfg.EndpointID)
	}
	if endpoint == "" {
		return nil, errors.New("RunPod endpoint ID is not configured")
	}
	if key == "" {
		return nil, errors.New("RunPod API key is not configured")
	}
	base, err := normalizeRunPodBaseURL(cfg.BaseURL)
	if err != nil {
		return nil, err
	}
	return &runPodClient{baseURL: base, endpointID: endpoint, apiKey: key, http: &http.Client{Timeout: 30 * time.Second}}, nil
}

func (a *App) GetRunPodConfig() RunPodConfig {
	cfg, _ := a.loadRunPodDiskConfig()
	endpoint := cfg.EndpointID
	if env := strings.TrimSpace(os.Getenv("RUNPOD_ENDPOINT_ID")); env != "" {
		endpoint = env
	}
	keySource := "none"
	configured := normalizeRunPodAPIKey(cfg.APIKey) != ""
	if configured {
		keySource = "secure config"
	} else if normalizeRunPodAPIKey(os.Getenv("RUNPOD_API_KEY")) != "" {
		configured, keySource = true, "environment"
	}
	base, _ := normalizeRunPodBaseURL(cfg.BaseURL)
	return RunPodConfig{EndpointID: endpoint, BaseURL: base, Configured: configured && endpoint != "", KeySource: keySource}
}

// SaveRunPodConfig stores credentials only in the Go backend config with owner-only permissions.
// Passing an empty key preserves the existing key.
func (a *App) SaveRunPodConfig(endpointID, apiKey, baseURL string) (RunPodConfig, error) {
	endpointID = strings.TrimSpace(endpointID)
	if endpointID == "" || strings.ContainsAny(endpointID, "/?# ") {
		return RunPodConfig{}, errors.New("valid RunPod endpoint ID is required")
	}
	base, err := normalizeRunPodBaseURL(baseURL)
	if err != nil {
		return RunPodConfig{}, err
	}
	old, _ := a.loadRunPodDiskConfig()
	if strings.TrimSpace(apiKey) == "" {
		apiKey = old.APIKey
	}
	if normalizeRunPodAPIKey(apiKey) == "" && normalizeRunPodAPIKey(os.Getenv("RUNPOD_API_KEY")) == "" {
		return RunPodConfig{}, errors.New("RunPod API key is required")
	}
	cfg := runPodDiskConfig{EndpointID: endpointID, BaseURL: base, APIKey: normalizeRunPodAPIKey(apiKey)}
	if err := os.MkdirAll(a.rootPath(), 0700); err != nil {
		return RunPodConfig{}, err
	}
	data, _ := json.MarshalIndent(cfg, "", "  ")
	if err := os.WriteFile(a.runPodConfigPath(), data, 0600); err != nil {
		return RunPodConfig{}, err
	}
	return a.GetRunPodConfig(), nil
}

// SaveAndTestRunPodConfig verifies new credentials first, then persists them atomically.
// A failed 401 never replaces the last known credential.
func (a *App) SaveAndTestRunPodConfig(endpointID, apiKey, baseURL string) (RunPodStatus, error) {
	endpointID = strings.TrimSpace(endpointID)
	if endpointID == "" || strings.ContainsAny(endpointID, "/?# ") {
		return RunPodStatus{}, errors.New("valid RunPod endpoint ID is required")
	}
	base, err := normalizeRunPodBaseURL(baseURL)
	if err != nil {
		return RunPodStatus{}, err
	}
	old, _ := a.loadRunPodDiskConfig()
	key := normalizeRunPodAPIKey(apiKey)
	if key == "" {
		key = normalizeRunPodAPIKey(old.APIKey)
	}
	if key == "" {
		key = normalizeRunPodAPIKey(os.Getenv("RUNPOD_API_KEY"))
	}
	if key == "" {
		return RunPodStatus{}, errors.New("RunPod API key is required")
	}
	client := &runPodClient{baseURL: base, endpointID: endpointID, apiKey: key, http: &http.Client{Timeout: 30 * time.Second}}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	data, _, err := client.request(ctx, http.MethodGet, "health", nil)
	if err != nil {
		return RunPodStatus{EndpointID: endpointID, Message: err.Error()}, err
	}
	var health struct {
		Workers struct {
			Idle    int `json:"idle"`
			Ready   int `json:"ready"`
			Running int `json:"running"`
		} `json:"workers"`
	}
	if err := json.Unmarshal(data, &health); err != nil {
		return RunPodStatus{}, errors.New("RunPod returned an invalid health response")
	}
	if err := os.MkdirAll(a.rootPath(), 0700); err != nil {
		return RunPodStatus{}, err
	}
	cfg := runPodDiskConfig{EndpointID: endpointID, BaseURL: base, APIKey: key}
	encoded, _ := json.MarshalIndent(cfg, "", "  ")
	tmp := a.runPodConfigPath() + ".tmp"
	if err := os.WriteFile(tmp, encoded, 0600); err != nil {
		return RunPodStatus{}, err
	}
	if err := os.Rename(tmp, a.runPodConfigPath()); err != nil {
		_ = os.Remove(tmp)
		return RunPodStatus{}, err
	}
	workers := health.Workers.Idle + health.Workers.Ready + health.Workers.Running
	return RunPodStatus{OK: true, EndpointID: endpointID, Workers: workers, Message: fmt.Sprintf("RunPod 인증 및 저장 완료 · 사용 가능 worker %d", workers)}, nil
}

func (c *runPodClient) request(ctx context.Context, method, suffix string, body any) ([]byte, int, error) {
	var reader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, 0, err
		}
		reader = bytes.NewReader(data)
	}
	endpoint := fmt.Sprintf("%s/%s/%s", c.baseURL, url.PathEscape(c.endpointID), strings.TrimLeft(suffix, "/"))
	req, err := http.NewRequestWithContext(ctx, method, endpoint, reader)
	if err != nil {
		return nil, 0, err
	}
	// RunPod requires exactly the Bearer scheme; query-string keys are deliberately avoided.
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 128<<20))
	if err != nil {
		return nil, resp.StatusCode, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		if resp.StatusCode == http.StatusUnauthorized {
			return data, resp.StatusCode, errors.New("RunPod 인증 실패(401): RunPod Settings > API Keys에서 새 API key를 생성해 실제 값을 다시 저장하세요. Endpoint ID나 가려진 key(****)를 API key 칸에 넣으면 인증되지 않습니다")
		}
		if resp.StatusCode == http.StatusForbidden {
			return data, resp.StatusCode, errors.New("RunPod 권한 거부(403): 이 API key에 Serverless endpoint 실행 권한이 없습니다")
		}
		return data, resp.StatusCode, fmt.Errorf("RunPod API returned %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	return data, resp.StatusCode, nil
}

func (a *App) TestRunPod() (RunPodStatus, error) {
	client, err := a.runPodClient()
	if err != nil {
		return RunPodStatus{}, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	// /health is endpoint-scoped and validates both endpoint ID and credentials without starting a paid job.
	data, _, err := client.request(ctx, http.MethodGet, "health", nil)
	if err != nil {
		return RunPodStatus{EndpointID: client.endpointID, Message: err.Error()}, err
	}
	var health struct {
		Workers struct {
			Idle    int `json:"idle"`
			Ready   int `json:"ready"`
			Running int `json:"running"`
		} `json:"workers"`
	}
	if err := json.Unmarshal(data, &health); err != nil {
		return RunPodStatus{}, errors.New("RunPod returned an invalid health response")
	}
	workers := health.Workers.Idle + health.Workers.Ready + health.Workers.Running
	message := fmt.Sprintf("RunPod 인증 완료 · 사용 가능 worker %d", workers)
	return RunPodStatus{OK: true, EndpointID: client.endpointID, Message: message, Workers: workers}, nil
}

// runPodJobResponse는 task 종류(reconstruct/motion)에 따라 output 스키마가
// 다르므로 원본 JSON을 보존하고, 각 호출부에서 자신에게 맞는 구조체로 푼다.
type runPodJobResponse struct {
	ID     string          `json:"id"`
	Status string          `json:"status"`
	Error  any             `json:"error"`
	Output json.RawMessage `json:"output"`
}

func runPodError(value any) string {
	if value == nil {
		return ""
	}
	if text, ok := value.(string); ok {
		return text
	}
	data, _ := json.Marshal(value)
	return string(data)
}

// runPodSync submits the job asynchronously (/run returns the job ID immediately)
// and polls /status until it settles. /runsync is deliberately avoided: RunPod holds
// that connection across cold starts (~50s measured), which trips the 30s HTTP
// client timeout before the wait window ends.
func (c *runPodClient) runPodSync(ctx context.Context, payload any) (runPodJobResponse, error) {
	data, _, err := c.request(ctx, http.MethodPost, "run", payload)
	if err != nil {
		return runPodJobResponse{}, err
	}
	var response runPodJobResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return response, errors.New("RunPod returned an invalid generation response")
	}
	for response.Status == "IN_QUEUE" || response.Status == "IN_PROGRESS" {
		if response.ID == "" {
			return response, errors.New("RunPod job response is missing its ID")
		}
		select {
		case <-ctx.Done():
			return response, fmt.Errorf("RunPod generation timeout: %w", ctx.Err())
		case <-time.After(2 * time.Second):
		}
		data, _, err = c.request(ctx, http.MethodGet, "status/"+url.PathEscape(response.ID), nil)
		if err != nil {
			return response, err
		}
		if err := json.Unmarshal(data, &response); err != nil {
			return response, errors.New("RunPod returned an invalid job status")
		}
	}
	if response.Status != "COMPLETED" {
		detail := runPodError(response.Error)
		if detail == "" {
			detail = "job ended with status " + response.Status
		}
		return response, fmt.Errorf("RunPod job failed: %s", detail)
	}
	return response, nil
}

func (a *App) runPodReconstruct(imagePath, workspace string) (Artifact, error) {
	client, err := a.runPodClient()
	if err != nil {
		return Artifact{}, err
	}
	image, err := os.ReadFile(imagePath)
	if err != nil {
		return Artifact{}, fmt.Errorf("read reconstruction input: %w", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()
	// octree_resolution/face_count/texture_resolution keep the GLB under RunPod's
	// ~20MB response limit (above it the COMPLETED output is silently dropped).
	// texture=true는 hy3dpaint 멀티뷰 PBR 텍스처(뒷면 포함)를 요청한다 — 미지원
	// handler(구버전)는 이 필드를 무시하고, paint 실패 시 shape GLB로 폴백된다.
	payload := map[string]any{"input": map[string]any{
		"image": base64.StdEncoding.EncodeToString(image), "seed": 1234,
		"steps": 30, "guidance_scale": 5.0,
		"octree_resolution": 256, "face_count": 40000,
		"texture": true, "max_num_view": 6, "texture_resolution": 512,
	}}
	response, err := client.runPodSync(ctx, payload)
	if err != nil {
		return Artifact{}, err
	}
	var out struct {
		GLB          string `json:"glb_base64"`
		Model        string `json:"model"`
		Error        string `json:"error"`
		Textured     bool   `json:"textured"`
		TextureError string `json:"texture_error"`
	}
	if len(response.Output) > 0 {
		_ = json.Unmarshal(response.Output, &out)
	}
	if out.GLB == "" {
		if out.Error != "" {
			return Artifact{}, fmt.Errorf("RunPod reconstruction failed: %s", out.Error)
		}
		return Artifact{}, errors.New("RunPod completed without a GLB output")
	}
	glb, err := base64.StdEncoding.DecodeString(out.GLB)
	if err != nil || len(glb) < 4 || string(glb[:4]) != "glTF" {
		return Artifact{}, errors.New("RunPod output is not a valid GLB")
	}
	dir := filepath.Join(workspace, "reconstruct")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return Artifact{}, err
	}
	path := filepath.Join(dir, "hunyuan3d21.glb")
	if err := os.WriteFile(path, glb, 0644); err != nil {
		return Artifact{}, err
	}
	metrics := map[string]any{
		"adapter": "runpod-hunyuan3d21", "model": out.Model,
		"bytes": len(glb), "previewOnly": false,
		"textured": out.Textured,
	}
	// paint 단계 실패는 job 실패가 아니다 — shape GLB는 유효하므로 사유만 남긴다.
	if out.TextureError != "" {
		metrics["textureError"] = out.TextureError
	}
	return Artifact{Stage: "reconstruct", Kind: "mesh", Path: path, Metrics: metrics}, nil
}

// MotionPrompt는 자연어 텍스트 하나가 HY-Motion 클립 하나로 변환되는 요청 단위다.
type MotionPrompt struct {
	ID       string  `json:"id"`
	Text     string  `json:"text"`
	Duration float64 `json:"duration"`
}

type MotionGenerateResult struct {
	Path   string            `json:"path"`   // 애니메이션이 베이킹된 GLB 경로
	Clips  int               `json:"clips"`  // 베이킹된 클립 수
	Model  string            `json:"model"`  // ex) HY-Motion-1.0-Lite
	Errors map[string]string `json:"errors"` // 프롬프트별 부분 실패 사유
}

// defaultMotionPrompts는 파이프라인 motion 단계가 RunPod HY-Motion으로 자동
// 실행될 때 생성하는 기본 게임 클립 세트다 (한 job으로 함께 생성된다).
func defaultMotionPrompts() []MotionPrompt {
	return []MotionPrompt{
		{ID: "idle", Text: "a person stands still, breathing calmly and shifting weight slightly", Duration: 4},
		{ID: "walk", Text: "a person walks forward casually", Duration: 5},
		{ID: "run", Text: "a person runs forward quickly", Duration: 5},
		{ID: "jump", Text: "a person jumps high in place", Duration: 4},
		{ID: "wave", Text: "a person waves hello with the right hand", Duration: 4},
	}
}

// RunPodGenerateMotion은 자연어 프롬프트들을 RunPod HY-Motion-1.0으로 보내
// SMPL 모션(JSON)을 받은 뒤, 로컬 baseline worker의 motion 단계로 rigged GLB에
// 리타겟·베이킹한다. 결과 GLB는 artifact로 등록된다.
func (a *App) RunPodGenerateMotion(jobID string, prompts []MotionPrompt) (MotionGenerateResult, error) {
	cleaned := make([]MotionPrompt, 0, len(prompts))
	for i, p := range prompts {
		p.Text = strings.TrimSpace(p.Text)
		if p.Text == "" {
			continue
		}
		if strings.TrimSpace(p.ID) == "" {
			p.ID = fmt.Sprintf("motion%d", i+1)
		}
		if p.Duration <= 0 {
			p.Duration = 5
		}
		cleaned = append(cleaned, p)
	}
	if len(cleaned) == 0 {
		return MotionGenerateResult{}, errors.New("모션 프롬프트 텍스트를 입력해 주세요")
	}
	if len(cleaned) > 20 {
		return MotionGenerateResult{}, errors.New("프롬프트는 한 번에 최대 20개까지 가능합니다")
	}
	client, err := a.runPodClient()
	if err != nil {
		return MotionGenerateResult{}, err
	}
	a.mu.Lock()
	var job *Job
	for i := range a.jobs {
		if a.jobs[i].ID == jobID {
			job = &a.jobs[i]
			break
		}
	}
	if job == nil {
		a.mu.Unlock()
		return MotionGenerateResult{}, errors.New("job not found")
	}
	// 리타겟 대상은 스킨이 있는 rig 단계 GLB가 우선; 없으면 최신 GLB로 폴백
	// (스킨 검증은 baseline worker의 motion 단계가 수행한다).
	riggedGLB := ""
	for _, artifact := range job.Artifacts {
		if strings.HasSuffix(strings.ToLower(artifact.Path), ".glb") {
			if artifact.Stage == "rig" {
				riggedGLB = artifact.Path
			} else if riggedGLB == "" {
				riggedGLB = artifact.Path
			}
		}
	}
	workspace := job.Workspace
	a.mu.Unlock()
	if riggedGLB == "" {
		return MotionGenerateResult{}, errors.New("리깅된 GLB가 없습니다 — 파이프라인을 rig 단계까지 먼저 실행해 주세요")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()
	payload := map[string]any{"input": map[string]any{
		"task": "motion", "prompts": cleaned, "seed": 42, "cfg_scale": 5.0,
	}}
	response, err := client.runPodSync(ctx, payload)
	if err != nil {
		return MotionGenerateResult{}, err
	}
	var out struct {
		Model   string            `json:"model"`
		Error   string            `json:"error"`
		Errors  map[string]string `json:"errors"`
		Motions []struct {
			ID string `json:"id"`
		} `json:"motions"`
	}
	if len(response.Output) == 0 || json.Unmarshal(response.Output, &out) != nil {
		return MotionGenerateResult{}, errors.New("RunPod motion 응답을 해석할 수 없습니다")
	}
	if out.Error != "" {
		return MotionGenerateResult{}, fmt.Errorf("RunPod motion 생성 실패: %s", out.Error)
	}
	if len(out.Motions) == 0 {
		return MotionGenerateResult{}, errors.New("RunPod이 모션을 생성하지 못했습니다")
	}
	motionDir := filepath.Join(workspace, "motion")
	if err := os.MkdirAll(motionDir, 0755); err != nil {
		return MotionGenerateResult{}, err
	}
	motionJSON := filepath.Join(motionDir, "hy_motion.json")
	if err := os.WriteFile(motionJSON, response.Output, 0644); err != nil {
		return MotionGenerateResult{}, err
	}

	artifact, err := a.bakeMotionGLB(jobID, workspace, riggedGLB)
	if err != nil {
		return MotionGenerateResult{}, err
	}
	clips := 0
	if n, ok := artifact.Metrics["animations"].(float64); ok {
		clips = int(n)
	}
	a.mu.Lock()
	for i := range a.jobs {
		if a.jobs[i].ID == jobID {
			a.jobs[i].Artifacts = append(a.jobs[i].Artifacts, artifact)
			a.jobs[i].Logs = append(a.jobs[i].Logs, LogEntry{time.Now().Format(time.RFC3339), "motion", "info",
				fmt.Sprintf("HY-Motion: %d개 클립을 GLB 애니메이션으로 베이킹 (%s)", clips, out.Model)})
			a.save()
			a.emitJobUpdate(a.jobs[i])
			break
		}
	}
	a.mu.Unlock()
	return MotionGenerateResult{Path: artifact.Path, Clips: clips, Model: out.Model, Errors: out.Errors}, nil
}

// bakeMotionGLB는 baseline worker의 motion 단계를 서브프로세스로 실행해
// workspace/motion/hy_motion.json의 SMPL 모션을 rigged GLB에 베이킹한다.
func (a *App) bakeMotionGLB(jobID, workspace, riggedGLB string) (Artifact, error) {
	worker, err := a.workerPath()
	if err != nil {
		return Artifact{}, err
	}
	req := map[string]any{"type": "run", "jobId": jobID, "stage": "motion",
		"workspace": workspace, "input": riggedGLB, "adapter": "hy-motion-retarget"}
	payload, _ := json.Marshal(req)
	cmd := exec.Command("python3", worker)
	cmd.Stdin = bytes.NewReader(append(payload, '\n'))
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return Artifact{}, err
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return Artifact{}, err
	}
	var baked *Artifact
	var workerErr error
	scan := bufio.NewScanner(stdout)
	for scan.Scan() {
		var ev workerEvent
		if json.Unmarshal(scan.Bytes(), &ev) != nil {
			continue
		}
		if ev.Type == "error" {
			workerErr = errors.New(ev.Message)
		}
		if ev.Type == "artifact" {
			baked = &Artifact{Stage: "motion", Kind: ev.Kind, Path: ev.Path, Metrics: ev.Metrics}
		}
	}
	if err := cmd.Wait(); err != nil {
		return Artifact{}, fmt.Errorf("motion bake worker failed: %v %s", err, stderr.String())
	}
	if workerErr != nil {
		return Artifact{}, workerErr
	}
	if baked == nil {
		return Artifact{}, errors.New("motion bake produced no artifact")
	}
	// passthrough(local-baseline) 결과가 아닌, 실제 리타겟 베이킹인지 확인한다.
	if adapter, _ := baked.Metrics["adapter"].(string); adapter != "hy-motion-retarget" {
		return Artifact{}, errors.New("모션 베이킹에는 스킨이 포함된 리깅 GLB가 필요합니다 — rig 단계를 다시 실행해 주세요")
	}
	return *baked, nil
}

// ClearRunPodConfig removes a stale key so the app can explicitly fall back to environment credentials.
func (a *App) ClearRunPodConfig() (RunPodConfig, error) {
	if err := os.Remove(a.runPodConfigPath()); err != nil && !os.IsNotExist(err) {
		return RunPodConfig{}, err
	}
	return a.GetRunPodConfig(), nil
}
