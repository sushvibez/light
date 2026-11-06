package hub

import (
	"os"
	"path/filepath"
	"testing"
)

// TestDownloadModelSafetensors_HFPythonCache verifies that we recognise the
// official HuggingFace Python cache layout (`refs/<rev>` + `snapshots/<sha>/`)
// and return the existing shard paths without making any network calls. The
// test only runs when the user actually has Qwen2.5-3B-Instruct cached.
func TestDownloadModelSafetensors_HFPythonCache(t *testing.T) {
	cache, err := CacheDir()
	if err != nil {
		t.Skipf("no cache dir: %v", err)
	}
	repoDir := filepath.Join(cache, "models--Qwen--Qwen2.5-3B-Instruct")
	if _, err := os.Stat(filepath.Join(repoDir, "refs", "main")); err != nil {
		t.Skipf("Qwen2.5-3B-Instruct not cached locally: %v", err)
	}
	paths, err := DownloadModelSafetensors("Qwen/Qwen2.5-3B-Instruct", "main")
	if err != nil {
		t.Fatalf("DownloadModelSafetensors: %v", err)
	}
	if len(paths) != 2 {
		t.Fatalf("expected 2 shards, got %d (%v)", len(paths), paths)
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("shard %s missing: %v", p, err)
		}
	}
	t.Logf("resolved %d shards: %v", len(paths), paths)
}
