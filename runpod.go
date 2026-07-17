package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
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
	data, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return nil, resp.StatusCode, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		if resp.StatusCode == http.StatusUnauthorized {
			return data, resp.StatusCode, errors.New("RunPod authentication failed (401): replace the API key in Settings or RUNPOD_API_KEY")
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

// ClearRunPodConfig removes a stale key so the app can explicitly fall back to environment credentials.
func (a *App) ClearRunPodConfig() (RunPodConfig, error) {
	if err := os.Remove(a.runPodConfigPath()); err != nil && !os.IsNotExist(err) {
		return RunPodConfig{}, err
	}
	return a.GetRunPodConfig(), nil
}
