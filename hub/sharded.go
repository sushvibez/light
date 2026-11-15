package hub

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// indexJSON mirrors HuggingFace's `model.safetensors.index.json` layout.
type indexJSON struct {
	Metadata  map[string]any    `json:"metadata"`
	WeightMap map[string]string `json:"weight_map"` // tensor name → shard filename
}

// DownloadModelSafetensors fetches a model's safetensors weights, transparently
// handling both single-file and sharded layouts. Returns the list of local
// paths to feed into `safetensors.LoadFiles` (in load order).
//
// Resolution rules, matching the HuggingFace Python client:
//
//  1. Try `model.safetensors`; if the repo ships one, use it.
//  2. Otherwise download `model.safetensors.index.json` and every shard it
//     names (`model-00001-of-NNNNN.safetensors`, …).
//  3. Otherwise fail with both probe errors so the user sees the layout.
//
// Each underlying file is cached on disk via `Download`.
func DownloadModelSafetensors(repo, revision string) ([]string, error) {
	if revision == "" {
		revision = "main"
	}
	// Try the single-file form first. On a 404 we fall through to the sharded
	// layout; other errors (auth, network) bubble up.
	p, sha, err := downloadAt(repo, revision, "model.safetensors", nil)
	if err == nil {
		return []string{p}, nil
	}
	if !isHTTPNotFound(err) {
		return nil, err
	}
	if sha != "" {
		revision = sha // the probe's commit: the whole load stays on it
	}
	// Sharded layout: index.json names every shard.
	// Shards are then fetched from the commit the index came from, so a
	// branch moving mid-load cannot mix commits.
	var idxPath string
	idxPath, sha, err = downloadAt(repo, revision, "model.safetensors.index.json", nil)
	if err != nil {
		return nil, fmt.Errorf("DownloadModelSafetensors: neither single nor sharded weights found for %s: %w", repo, err)
	}
	raw, err := os.ReadFile(idxPath)
	if err != nil {
		return nil, err
	}
	var idx indexJSON
	if err := json.Unmarshal(raw, &idx); err != nil {
		return nil, fmt.Errorf("DownloadModelSafetensors: parse index: %w", err)
	}
	if len(idx.WeightMap) == 0 {
		return nil, fmt.Errorf("DownloadModelSafetensors: empty weight_map in %s", repo)
	}
	uniq := make(map[string]struct{}, 8)
	for _, shard := range idx.WeightMap {
		// Shard names are checkpoint-controlled: reject traversal before
		// fetching anything.
		if _, err := artifactPath(shard); err != nil {
			return nil, fmt.Errorf("DownloadModelSafetensors: %s index: %w", repo, err)
		}
		uniq[shard] = struct{}{}
	}
	if sha != "" {
		revision = sha
	}
	shards := make([]string, 0, len(uniq))
	for s := range uniq {
		shards = append(shards, s)
	}
	sort.Strings(shards) // deterministic load order
	paths := make([]string, len(shards))
	for i, shard := range shards {
		p, _, err := downloadAt(repo, revision, shard, nil)
		if err != nil {
			return nil, fmt.Errorf("DownloadModelSafetensors: shard %s: %w", shard, err)
		}
		paths[i] = p
	}
	return paths, nil
}

// isHTTPNotFound returns true when err is a 404-shaped Download error. We
// detect by string match since Download reports HTTP errors via fmt.Errorf;
// good enough for the single-vs-sharded probe.
func isHTTPNotFound(err error) bool {
	if err == nil {
		return false
	}
	return strings.Contains(err.Error(), "HTTP 404")
}
