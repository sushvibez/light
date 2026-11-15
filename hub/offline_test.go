package hub

import (
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

type noNetwork struct{ t *testing.T }

func (n noNetwork) RoundTrip(r *http.Request) (*http.Response, error) {
	n.t.Errorf("offline mode made a request: %s %s", r.Method, r.URL)
	return nil, errors.New("no network in offline mode")
}

func writeFile(t *testing.T, p, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// HF_HUB_OFFLINE: Download, DownloadModelSafetensors, DownloadTo and
// ListFiles serve the cache (refs → snapshots/<sha>, a SHA revision, the
// legacy snapshots/<revision> layout) and never touch the network.
func TestOfflineServesCacheOnly(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HF_HOME", home)
	t.Setenv("HF_ENDPOINT", "http://offline.invalid")
	t.Setenv("HF_HUB_OFFLINE", "1")
	old := httpClient
	httpClient = &http.Client{Transport: noNetwork{t}}
	t.Cleanup(func() { httpClient = old })

	cache := filepath.Join(home, "hub")
	sharded := filepath.Join(cache, "models--org--sharded")
	writeFile(t, filepath.Join(sharded, "refs", "main"), shaA)
	snap := filepath.Join(sharded, "snapshots", shaA)
	writeFile(t, filepath.Join(snap, "config.json"), "{}")
	writeFile(t, filepath.Join(snap, "model.safetensors.index.json"), `{"weight_map": {"a": "s1.safetensors", "b": "sub/s2.safetensors"}}`)
	writeFile(t, filepath.Join(snap, "s1.safetensors"), "one")
	writeFile(t, filepath.Join(snap, "sub", "s2.safetensors"), "two")
	writeFile(t, filepath.Join(snap, "big.bin.part"), "partial")
	legacy := filepath.Join(cache, "models--org--legacy", "snapshots", "v1")
	writeFile(t, filepath.Join(legacy, "model.safetensors"), "weights")

	if p, err := Download("org/sharded", "main", "config.json"); err != nil || p != filepath.Join(snap, "config.json") {
		t.Errorf("refs hit: %s %v", p, err)
	}
	if p, err := Download("org/sharded", shaA, "config.json"); err != nil || p != filepath.Join(snap, "config.json") {
		t.Errorf("sha hit: %s %v", p, err)
	}
	if p, err := Download("org/legacy", "v1", "model.safetensors"); err != nil || p != filepath.Join(legacy, "model.safetensors") {
		t.Errorf("legacy hit: %s %v", p, err)
	}
	paths, err := DownloadModelSafetensors("org/sharded", "main")
	if err != nil || len(paths) != 2 {
		t.Errorf("sharded from cache: %v %v", paths, err)
	}
	if paths, err := DownloadModelSafetensors("org/legacy", "v1"); err != nil || len(paths) != 1 {
		t.Errorf("single from legacy cache: %v %v", paths, err)
	}

	files, err := ListFiles("org/sharded", "main")
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, f := range files {
		names = append(names, f.Name)
	}
	sort.Strings(names)
	want := []string{"config.json", "model.safetensors.index.json", "s1.safetensors", "sub/s2.safetensors"}
	if len(names) != len(want) {
		t.Fatalf("ListFiles %v, want %v", names, want)
	}
	for i := range want {
		if names[i] != want[i] {
			t.Errorf("ListFiles %v, want %v", names, want)
			break
		}
	}

	// Misses are errors wrapping ErrOffline, with no request made.
	misses := map[string]error{}
	_, misses["Download refs miss"] = Download("org/sharded", "main", "missing.json")
	_, misses["Download sha miss"] = Download("org/sharded", shaB, "config.json")
	_, misses["Download unknown repo"] = Download("org/unknown", "main", "config.json")
	_, misses["DownloadModelSafetensors"] = DownloadModelSafetensors("org/unknown", "main")
	_, misses["ListFiles"] = ListFiles("org/unknown", "main")
	misses["DownloadTo"] = DownloadTo("org/unknown", "main", "config.json", filepath.Join(t.TempDir(), "c.json"), nil)
	for name, err := range misses {
		if !errors.Is(err, ErrOffline) {
			t.Errorf("%s: err = %v, want ErrOffline", name, err)
		}
	}
}

func TestOfflineEnvValues(t *testing.T) {
	for v, want := range map[string]bool{"1": true, "true": true, "TRUE": true, "yes": true, "on": true, "0": false, "": false, "false": false} {
		t.Setenv("HF_HUB_OFFLINE", v)
		if got := Offline(); got != want {
			t.Errorf("HF_HUB_OFFLINE=%q: %v, want %v", v, got, want)
		}
	}
}
