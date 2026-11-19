package diffusion

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"light/hub"
)

// Snapshot returns a local directory holding a diffusers pipeline: repo
// itself when it is a directory, else the Hub snapshot of repo with every
// component folder named in model_index.json downloaded (weights in
// safetensors only; root-level single-file checkpoints are skipped).
func Snapshot(repo string) (string, error) {
	return SnapshotWithProgress(repo, nil)
}

// DownloadProgress reports a component file's download (done of total
// bytes; total 0 if unknown). Files already cached are not reported.
type DownloadProgress func(file string, done, total int64)

// SnapshotWithProgress is Snapshot reporting each file as it downloads.
func SnapshotWithProgress(repo string, progress DownloadProgress) (string, error) {
	if st, err := os.Stat(repo); err == nil && st.IsDir() {
		return repo, nil
	}
	idx, err := hub.Download(repo, "main", "model_index.json")
	if err != nil {
		return "", fmt.Errorf("diffusion: %s: %w", repo, err)
	}
	raw, err := os.ReadFile(idx)
	if err != nil {
		return "", err
	}
	var index map[string]any
	if err := json.Unmarshal(raw, &index); err != nil {
		return "", fmt.Errorf("diffusion: model_index.json: %w", err)
	}
	files, err := hub.ListFiles(repo, "main")
	if err != nil {
		return "", err
	}
	safetensors := map[string]bool{} // component folders that ship safetensors
	for _, f := range files {
		if dir, _, ok := strings.Cut(f.Name, "/"); ok && strings.HasSuffix(f.Name, ".safetensors") {
			safetensors[dir] = true
		}
	}
	for _, f := range files {
		dir, _, ok := strings.Cut(f.Name, "/")
		if !ok {
			continue // root files: README, single-file checkpoints
		}
		if _, isComponent := index[dir]; !isComponent {
			continue
		}
		if safetensors[dir] && (strings.HasSuffix(f.Name, ".bin") || strings.HasSuffix(f.Name, ".pt")) {
			continue
		}
		if strings.HasSuffix(f.Name, ".msgpack") || strings.HasSuffix(f.Name, ".h5") || strings.HasSuffix(f.Name, ".onnx") {
			continue
		}
		// Large files over flaky connections: retry, resuming the partial file.
		var derr error
		for attempt := range 4 {
			if attempt > 0 {
				time.Sleep(time.Duration(attempt) * 2 * time.Second)
			}
			var p hub.Progress
			if progress != nil {
				name := f.Name
				p = func(done, total int64) { progress(name, done, total) }
			}
			if _, derr = hub.DownloadWithProgress(repo, "main", f.Name, p); derr == nil {
				break
			}
		}
		if derr != nil {
			return "", fmt.Errorf("diffusion: %s/%s: %w", repo, f.Name, derr)
		}
	}
	return filepath.Dir(idx), nil
}
