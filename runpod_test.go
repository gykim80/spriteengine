package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
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
