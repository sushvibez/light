package hub

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// DefaultEndpoint is the public HuggingFace Hub URL. Override via the
// `HF_ENDPOINT` environment variable to point at a mirror.
const DefaultEndpoint = "https://huggingface.co"

// Endpoint returns the configured HF endpoint (env override > default).
func Endpoint() string {
	if v := os.Getenv("HF_ENDPOINT"); v != "" {
		return strings.TrimRight(v, "/")
	}
	return DefaultEndpoint
}

// CacheDir returns the local cache root: `$HF_HOME/hub` or
// `$XDG_CACHE_HOME/huggingface/hub`, falling back to `~/.cache/huggingface/hub`.
func CacheDir() (string, error) {
	if hf := os.Getenv("HF_HOME"); hf != "" {
		return filepath.Join(hf, "hub"), nil
	}
	if xdg := os.Getenv("XDG_CACHE_HOME"); xdg != "" {
		return filepath.Join(xdg, "huggingface", "hub"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".cache", "huggingface", "hub"), nil
}

// Download fetches `<endpoint>/<repo>/resolve/<revision>/<filename>` to the
// local cache and returns the path to the cached file.
//
// The cache mirrors the official `huggingface_hub` Python client's layout —
// `models--<org>--<name>/snapshots/<commit-sha>/<filename>` plus
// `refs/<revision>` containing the SHA — so files we download are
// transparently picked up by Diffusers / Transformers without any extra
// configuration. Existing files in our older `snapshots/main/...` layout
// are still honored as a fallback.
func Download(repo, revision, filename string) (string, error) {
	return download(repo, revision, filename, nil)
}

func download(repo, revision, filename string, progress Progress) (string, error) {
	if repo == "" || filename == "" {
		return "", fmt.Errorf("hub.Download: repo and filename are required")
	}
	if revision == "" {
		revision = "main"
	}
	// A local checkpoint directory (e.g. a tiny-random test model under a
	// testdata/ folder) is read in place. A missing file reports HTTP 404,
	// as the Hub would, so single-file → sharded fallbacks still work.
	if st, err := os.Stat(repo); err == nil && st.IsDir() {
		p := filepath.Join(repo, filepath.FromSlash(filename))
		if _, err := os.Stat(p); err != nil {
			return "", fmt.Errorf("hub.Download: %s not in local directory %s (HTTP 404)", filename, repo)
		}
		return p, nil
	}
	cache, err := CacheDir()
	if err != nil {
		return "", err
	}
	repoDir := filepath.Join(cache, "models--"+strings.ReplaceAll(repo, "/", "--"))

	// 1) Already in HF Python layout (preferred — what we want everything
	//    to converge on).
	if pyPath, ok := lookupHFPythonCache(repoDir, revision, filename); ok {
		return pyPath, nil
	}
	// 2) Found in our legacy `snapshots/<revision>/<file>` layout — try to
	//    migrate it into HF Python's `snapshots/<sha>/<file>` so other
	//    tools (Diffusers, Transformers, the `huggingface_hub` CLI) can
	//    pick it up. If the SHA can't be resolved (e.g. offline) we just
	//    keep using the legacy file — it's still readable by us.
	legacy := filepath.Join(repoDir, "snapshots", revision, filepath.FromSlash(filename))
	if st, err := os.Stat(legacy); err == nil && st.Size() > 0 {
		if migrated, ok := migrateLegacyFile(repoDir, repo, revision, filename, legacy); ok {
			return migrated, nil
		}
		return legacy, nil
	}
	// 3) Resolve the commit SHA so we can write the file at the HF Python
	//    location. If the API fails (offline, etc.) fall back to the legacy
	//    path so we still produce something usable.
	sha, err := resolveCommitSHA(repo, revision)
	if err != nil {
		// Couldn't reach the API — try fetching into the legacy layout so
		// we at least have the file. Subsequent runs (with API access) will
		// migrate to the HF Python layout on their own.
		if err := os.MkdirAll(filepath.Dir(legacy), 0o755); err != nil {
			return "", err
		}
		url := fmt.Sprintf("%s/%s/resolve/%s/%s", Endpoint(), repo, revision, filename)
		if dlErr := fetchWith(url, legacy, progress); dlErr != nil {
			return "", fmt.Errorf("resolve sha + fetch failed: sha-resolve: %v; fetch: %w", err, dlErr)
		}
		return legacy, nil
	}
	// 4) Persist refs/<revision> = sha so future runs (and other tools)
	//    can find the snapshot dir.
	refsDir := filepath.Join(repoDir, "refs")
	if err := os.MkdirAll(refsDir, 0o755); err != nil {
		return "", err
	}
	refsPath := filepath.Join(refsDir, revision)
	if existing, err := os.ReadFile(refsPath); err != nil || strings.TrimSpace(string(existing)) != sha {
		if err := os.WriteFile(refsPath, []byte(sha), 0o644); err != nil {
			return "", err
		}
	}
	// 5) Download to snapshots/<sha>/<file>.
	snapDir := filepath.Join(repoDir, "snapshots", sha)
	localPath := filepath.Join(snapDir, filepath.FromSlash(filename))
	if err := os.MkdirAll(filepath.Dir(localPath), 0o755); err != nil {
		return "", err
	}
	if st, err := os.Stat(localPath); err == nil && st.Size() > 0 {
		return localPath, nil
	}
	url := fmt.Sprintf("%s/%s/resolve/%s/%s", Endpoint(), repo, revision, filename)
	if err := fetchWith(url, localPath, progress); err != nil {
		return "", err
	}
	return localPath, nil
}

// migrateLegacyFile moves a file from our old `snapshots/<revision>/<file>`
// layout into HF Python's `snapshots/<sha>/<file>` (and writes the matching
// `refs/<revision>` SHA pointer). Returns the new path on success. Falls
// back silently on any error so the caller can keep using the legacy path.
//
// Uses os.Rename when source + destination are on the same filesystem (the
// common case under ~/.cache); the rename is atomic, so partial migrations
// can't corrupt the cache.
func migrateLegacyFile(repoDir, repo, revision, filename, legacy string) (string, bool) {
	sha, err := resolveCommitSHA(repo, revision)
	if err != nil {
		return "", false
	}
	refsDir := filepath.Join(repoDir, "refs")
	if err := os.MkdirAll(refsDir, 0o755); err != nil {
		return "", false
	}
	refsPath := filepath.Join(refsDir, revision)
	if existing, err := os.ReadFile(refsPath); err != nil || strings.TrimSpace(string(existing)) != sha {
		if err := os.WriteFile(refsPath, []byte(sha), 0o644); err != nil {
			return "", false
		}
	}
	dst := filepath.Join(repoDir, "snapshots", sha, filepath.FromSlash(filename))
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return "", false
	}
	if err := os.Rename(legacy, dst); err != nil {
		return "", false
	}
	return dst, true
}

// shaCache deduplicates the per-(repo, revision) SHA resolution within a
// single process. The HF API call is small (~JSON) but doing it once per
// download in a multi-shard repo would be wasteful.
var (
	shaCacheMu sync.Mutex
	shaCache   = make(map[string]string)
)

// resolveCommitSHA fetches the commit SHA for `repo`/`revision` from the
// HF model API. Cached per process. The returned hex SHA is what HF
// Python's `huggingface_hub` uses as the snapshot directory name.
func resolveCommitSHA(repo, revision string) (string, error) {
	key := repo + "@" + revision
	shaCacheMu.Lock()
	if sha, ok := shaCache[key]; ok {
		shaCacheMu.Unlock()
		return sha, nil
	}
	shaCacheMu.Unlock()
	url := fmt.Sprintf("%s/api/models/%s/revision/%s", Endpoint(), repo, revision)
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "light/0.1 (+https://github.com/sushvibez/light)")
	if tok := os.Getenv("HF_TOKEN"); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("hub.resolveCommitSHA: GET %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return "", fmt.Errorf("hub.resolveCommitSHA: GET %s: HTTP %d: %s", url, resp.StatusCode, string(body))
	}
	var info struct {
		SHA string `json:"sha"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return "", fmt.Errorf("hub.resolveCommitSHA: parse: %w", err)
	}
	if info.SHA == "" {
		return "", fmt.Errorf("hub.resolveCommitSHA: empty sha in response")
	}
	shaCacheMu.Lock()
	shaCache[key] = info.SHA
	shaCacheMu.Unlock()
	return info.SHA, nil
}

// fetchWith streams `url` to `dst` (with `.part` suffix during download),
// reporting progress when non-nil. Honours HuggingFace LFS redirects (the
// resolve endpoint 302s to a CDN URL). An existing .part is resumed with a
// Range request (multi-GB checkpoints over flaky links); a server answering
// 200 instead of 206 restarts it.
func fetchWith(url, dst string, progress Progress) error {
	tmp := dst + ".part"
	var have int64
	if st, err := os.Stat(tmp); err == nil {
		have = st.Size()
	}
	client := &http.Client{Timeout: 0} // model files can be 100s of MB
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	setHeaders(req)
	if have > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", have))
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("hub.Download: GET %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("hub.Download: GET %s: HTTP %d: %s", url, resp.StatusCode, string(body))
	}
	flags := os.O_WRONLY | os.O_CREATE
	if have > 0 && resp.StatusCode == http.StatusPartialContent {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	out, err := os.OpenFile(tmp, flags, 0o644)
	if err != nil {
		return err
	}
	var w io.Writer = out
	if progress != nil {
		start := int64(0)
		if flags&os.O_APPEND != 0 {
			start = have
		}
		total := int64(-1)
		if resp.ContentLength >= 0 {
			total = start + resp.ContentLength
		}
		w = &progressWriter{w: out, done: start, total: max(total, 0), report: progress}
		progress(start, max(total, 0))
	}
	if _, err := io.Copy(w, resp.Body); err != nil {
		_ = out.Close()
		return fmt.Errorf("hub.Download: copy: %w (partial file kept at %s for resume)", err, tmp)
	}
	if err := out.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, dst)
}

// lookupHFPythonCache returns the path to a file in HuggingFace Python's
// cache layout if it's present, so we don't re-download artifacts the user
// already pulled with `huggingface_hub`. Layout: `refs/<revision>` contains
// the commit SHA, `snapshots/<sha>/<file>` is a symlink into `blobs/`.
func lookupHFPythonCache(repoDir, revision, filename string) (string, bool) {
	shaBytes, err := os.ReadFile(filepath.Join(repoDir, "refs", revision))
	if err != nil {
		return "", false
	}
	sha := strings.TrimSpace(string(shaBytes))
	if sha == "" {
		return "", false
	}
	candidate := filepath.Join(repoDir, "snapshots", sha, filepath.FromSlash(filename))
	if st, err := os.Stat(candidate); err == nil && st.Size() > 0 {
		return candidate, true
	}
	return "", false
}

// Sha256 computes the SHA-256 of a local file. Useful for users wanting to
// pin a known-good revision.
func Sha256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
