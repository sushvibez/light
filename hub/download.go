package hub

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
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

// Download fetches `<endpoint>/<repo>/resolve/<sha>/<filename>` to the
// local cache and returns the path to the cached file, where <sha> is the
// commit `revision` resolves to.
//
// The cache mirrors the official `huggingface_hub` Python client's layout —
// `models--<org>--<name>/snapshots/<commit-sha>/<filename>` plus
// `refs/<revision>` containing the SHA — so files we download are
// transparently picked up by Diffusers / Transformers without any extra
// configuration. Bytes are always fetched from the resolved commit, never
// the mutable revision, so a snapshots/<sha> directory only ever holds
// that commit's files.
//
// filename (and revision) must be relative, slash-separated repository
// paths: "..", rooted or drive paths, and backslashes are rejected so a
// name from an untrusted checkpoint index cannot escape the snapshot.
//
// Files in our older `snapshots/<revision>/<file>` layout are still served
// as an unpinned fallback, and so is a fetch into that layout when the
// commit cannot be resolved (offline API, mirror without the API). Those
// files are not moved into a snapshots/<sha> directory: nothing proves
// which commit produced them.
func Download(repo, revision, filename string) (string, error) {
	return DownloadContext(context.Background(), repo, revision, filename)
}

// DownloadContext is Download with a context: canceling ctx aborts the
// revision lookup, a wait on another caller's download of the same file,
// and the transfer itself (a partial file is kept for a later resume).
func DownloadContext(ctx context.Context, repo, revision, filename string) (string, error) {
	p, _, err := downloadAt(ctx, repo, revision, filename, nil)
	return p, err
}

// downloadAt is download that also reports the commit the file is pinned
// to: "" for a local directory or the unpinned legacy/offline layout. The
// commit is reported even when the fetch itself fails (e.g. a 404 probe).
// Callers that fetch several files of one model (DownloadModelSafetensors)
// pass that SHA back as the revision so every file comes from one commit.
func downloadAt(ctx context.Context, repo, revision, filename string, progress Progress) (string, string, error) {
	if repo == "" || filename == "" {
		return "", "", fmt.Errorf("hub.Download: repo and filename are required")
	}
	if revision == "" {
		revision = "main"
	}
	rel, err := artifactPath(filename)
	if err != nil {
		return "", "", fmt.Errorf("hub.Download: %w", err)
	}
	// A local checkpoint directory (e.g. a tiny-random test model under a
	// testdata/ folder) is read in place. A missing file reports HTTP 404,
	// as the Hub would, so single-file → sharded fallbacks still work.
	if st, err := os.Stat(repo); err == nil && st.IsDir() {
		p := filepath.Join(repo, rel)
		if _, err := os.Stat(p); err != nil {
			return "", "", fmt.Errorf("hub.Download: %s not in local directory %s: %w", filename, repo,
				&HTTPError{Status: http.StatusNotFound, URL: p})
		}
		return p, "", nil
	}
	if err := checkRepoID(repo); err != nil {
		return "", "", fmt.Errorf("hub.Download: %w", err)
	}
	if err := checkRevision(revision); err != nil {
		return "", "", fmt.Errorf("hub.Download: %w", err)
	}
	cache, err := CacheDir()
	if err != nil {
		return "", "", err
	}
	repoDir := filepath.Join(cache, "models--"+strings.ReplaceAll(repo, "/", "--"))

	// 0) A full commit SHA is already immutable: snapshots/<sha> directly,
	//    no refs entry, no API call (as huggingface_hub does).
	if isCommitSHA(revision) {
		p, err := fetchSnapshot(ctx, repoDir, repo, revision, rel, filename, progress)
		return p, revision, err
	}
	// 1) Already in HF Python layout (refs/<revision> → snapshots/<sha>).
	if p, sha, ok := lookupHFPythonCache(repoDir, revision, rel); ok {
		return p, sha, nil
	}
	// 2) Our legacy `snapshots/<revision>/<file>` layout: served as is
	//    (unpinned; see Download).
	legacy := filepath.Join(repoDir, "snapshots", filepath.FromSlash(revision), rel)
	if st, err := os.Stat(legacy); err == nil && st.Size() > 0 {
		return legacy, "", nil
	}
	// 3) Resolve the commit SHA. If the API fails (offline, etc.) fetch the
	//    revision into the legacy layout so we still produce something
	//    usable — explicitly unpinned.
	sha, err := resolveCommitSHA(ctx, repo, revision)
	if err != nil {
		if cerr := ctx.Err(); cerr != nil {
			return "", "", fmt.Errorf("hub.Download: %w", cerr)
		}
		if err := os.MkdirAll(filepath.Dir(legacy), 0o755); err != nil {
			return "", "", err
		}
		if dlErr := fetchLocked(ctx, resolveURL(repo, revision, filename), legacy, false, progress); dlErr != nil {
			return "", "", fmt.Errorf("resolve sha + fetch failed: sha-resolve: %v; fetch: %w", err, dlErr)
		}
		return legacy, "", nil
	}
	// 4) Persist refs/<revision> = sha so future runs (and other tools)
	//    can find the snapshot dir.
	if err := writeRef(repoDir, revision, sha); err != nil {
		return "", "", err
	}
	// 5) Download snapshots/<sha>/<file> from that same commit.
	p, err := fetchSnapshot(ctx, repoDir, repo, sha, rel, filename, progress)
	return p, sha, err
}

// fetchSnapshot returns snapshots/<sha>/<rel>, fetching it from
// resolve/<sha>/<filename> when it is not cached.
func fetchSnapshot(ctx context.Context, repoDir, repo, sha, rel, filename string, progress Progress) (string, error) {
	localPath := filepath.Join(repoDir, "snapshots", sha, rel)
	if st, err := os.Stat(localPath); err == nil && st.Size() > 0 {
		return localPath, nil
	}
	if err := os.MkdirAll(filepath.Dir(localPath), 0o755); err != nil {
		return "", err
	}
	if err := fetchLocked(ctx, resolveURL(repo, sha, filename), localPath, true, progress); err != nil {
		return "", err
	}
	return localPath, nil
}

// writeRef records refs/<revision> = sha. Revisions may contain slashes
// ("refs/pr/1" → refs/refs/pr/1, as huggingface_hub lays it out), so the
// full parent is created; the ref is published by rename so a concurrent
// reader never sees a truncated SHA.
func writeRef(repoDir, revision, sha string) error {
	refsPath := filepath.Join(repoDir, "refs", filepath.FromSlash(revision))
	if existing, err := os.ReadFile(refsPath); err == nil && strings.TrimSpace(string(existing)) == sha {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(refsPath), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(refsPath), ".ref-*")
	if err != nil {
		return err
	}
	_, werr := tmp.WriteString(sha)
	cerr := tmp.Close()
	if werr == nil {
		werr = cerr
	}
	if werr == nil {
		werr = os.Rename(tmp.Name(), refsPath)
	}
	if werr != nil {
		_ = os.Remove(tmp.Name())
	}
	return werr
}

// shaCache deduplicates the per-(repo, revision) SHA resolution within a
// single process. The HF API call is small (~JSON) but doing it once per
// download in a multi-shard repo would be wasteful; it also keeps every
// file a process loads for one revision on the same commit.
var (
	shaCacheMu sync.Mutex
	shaCache   = make(map[string]string)
)

// resolveCommitSHA fetches the commit SHA for `repo`/`revision` from the
// HF model API. Cached per process. The returned hex SHA is what HF
// Python's `huggingface_hub` uses as the snapshot directory name; anything
// that is not a full hex commit id is rejected.
func resolveCommitSHA(ctx context.Context, repo, revision string) (string, error) {
	key := repo + "@" + revision
	shaCacheMu.Lock()
	if sha, ok := shaCache[key]; ok {
		shaCacheMu.Unlock()
		return sha, nil
	}
	shaCacheMu.Unlock()
	apiURL := fmt.Sprintf("%s/api/models/%s/revision/%s", Endpoint(), repo, url.PathEscape(revision))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, apiURL, nil)
	if err != nil {
		return "", err
	}
	setHeaders(req)
	resp, err := httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("hub.resolveCommitSHA: GET %s: %w", apiURL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return "", fmt.Errorf("hub.resolveCommitSHA: GET %w", newHTTPError(apiURL, resp))
	}
	var info struct {
		SHA string `json:"sha"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return "", fmt.Errorf("hub.resolveCommitSHA: parse: %w", err)
	}
	if !isCommitSHA(info.SHA) {
		return "", fmt.Errorf("hub.resolveCommitSHA: %q is not a commit sha", info.SHA)
	}
	shaCacheMu.Lock()
	shaCache[key] = info.SHA
	shaCacheMu.Unlock()
	return info.SHA, nil
}

// fetchLocked is fetchWith under exclusive ownership of dst (in-process
// and across processes, see lockDest). It rechecks dst after acquiring the
// lock, so a caller that waited on another's download returns its result
// instead of fetching again.
func fetchLocked(ctx context.Context, url, dst string, pinned bool, progress Progress) error {
	unlock, err := lockDest(ctx, dst)
	if err != nil {
		return fmt.Errorf("hub.Download: lock %s: %w", dst, err)
	}
	defer unlock()
	if st, err := os.Stat(dst); err == nil && st.Size() > 0 {
		return nil
	}
	return fetchWith(ctx, url, dst, pinned, progress)
}

// fetchWith streams `url` to `dst` (with `.part` suffix during download),
// reporting progress when non-nil. Honours HuggingFace LFS redirects (the
// resolve endpoint 302s to a CDN URL). An existing .part is resumed with a
// Range request (multi-GB checkpoints over flaky links); a server answering
// 200 instead of 206 restarts it. A 206 must continue exactly at the
// partial's length and run to the end of the object; anything else, or a
// .part whose identity is not established, is discarded rather than
// spliced. dst is only published (renamed) once it holds the full
// advertised length. Callers hold lockDest(dst).
//
// A partial is only resumed when it provably belongs to the object being
// fetched. The .part.url sidecar records the URL and the strong ETag of the
// response that started it. pinned means url names an immutable object
// (resolve/<commit sha>/...), so a matching URL suffices. An unpinned URL
// (a branch, fetched when the commit could not be resolved) can serve a
// different object at any time, so it is resumed only with the recorded
// ETag sent as If-Range: the server then answers 200 with the whole new
// object if it changed, which restarts the partial. A partial with no
// sidecar (an older light), a different URL, or (unpinned) no strong ETag
// restarts from zero.
func fetchWith(ctx context.Context, url, dst string, pinned bool, progress Progress) error {
	tmp, srcFile := dst+".part", dst+".part.url"
	var have int64
	if st, err := os.Stat(tmp); err == nil {
		have = st.Size()
	}
	var etag string
	if have > 0 {
		prevURL, prevETag, ok := readPartMeta(srcFile)
		switch {
		case !ok || prevURL != url:
			have = 0 // unknown or other object
		case !pinned && prevETag == "":
			have = 0 // mutable URL without a validator
		default:
			etag = prevETag
		}
		if have == 0 {
			_ = os.Remove(tmp)
			_ = os.Remove(srcFile)
		}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	setHeaders(req)
	if have > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", have))
		if !pinned {
			req.Header.Set("If-Range", etag)
		}
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("hub.Download: GET %s: %w", url, err)
	}
	defer resp.Body.Close()
	if have > 0 && !pinned && (resp.StatusCode == http.StatusPartialContent || resp.StatusCode == http.StatusRequestedRangeNotSatisfiable) {
		// A server ignoring If-Range must at least not hand us bytes of an
		// object with another validator.
		if got := resp.Header.Get("ETag"); got != "" && got != etag {
			_ = os.Remove(tmp)
			_ = os.Remove(srcFile)
			return fmt.Errorf("hub.Download: GET %s: resuming object %s but the server sent %s; discarded the partial file, a retry restarts", url, etag, got)
		}
	}
	if have > 0 && resp.StatusCode == http.StatusRequestedRangeNotSatisfiable {
		// Nothing past the partial: it is complete if the server says the
		// object is exactly that long.
		if cr, ok := parseContentRange(resp.Header.Get("Content-Range")); ok && cr.start < 0 && cr.total == have {
			return publish(tmp, srcFile, dst)
		}
		_ = os.Remove(tmp)
		return fmt.Errorf("hub.Download: GET %s: HTTP 416 resuming at byte %d; discarded the partial file, a retry restarts", url, have)
	}
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return fmt.Errorf("hub.Download: GET %w", newHTTPError(url, resp))
	}
	start, total := int64(0), resp.ContentLength // -1 when unknown
	flags := os.O_WRONLY | os.O_CREATE
	if resp.StatusCode == http.StatusPartialContent {
		cr, ok := parseContentRange(resp.Header.Get("Content-Range"))
		if !ok || cr.start != have || cr.total < 0 || cr.end != cr.total-1 ||
			(resp.ContentLength >= 0 && resp.ContentLength != cr.end-cr.start+1) {
			_ = os.Remove(tmp)
			return fmt.Errorf("hub.Download: GET %s: HTTP 206 with Content-Range %q (Content-Length %d) does not continue the %d-byte partial file; discarded it, a retry restarts",
				url, resp.Header.Get("Content-Range"), resp.ContentLength, have)
		}
		start, total = have, cr.total
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	if start == 0 {
		if err := writePartMeta(srcFile, url, strongETag(resp.Header.Get("ETag"))); err != nil {
			return err
		}
	}
	out, err := os.OpenFile(tmp, flags, 0o644)
	if err != nil {
		return err
	}
	var w io.Writer = out
	if progress != nil {
		w = &progressWriter{w: out, done: start, total: max(total, 0), report: progress}
		progress(start, max(total, 0))
	}
	n, err := io.Copy(w, resp.Body)
	if err != nil {
		_ = out.Close()
		if cerr := ctx.Err(); cerr != nil {
			err = cerr // report the cancellation, not the transport's view of it
		}
		return fmt.Errorf("hub.Download: copy: %w (partial file kept at %s for resume)", err, tmp)
	}
	if err := out.Close(); err != nil {
		return err
	}
	if size := start + n; total >= 0 && size != total {
		if size > total {
			_ = os.Remove(tmp) // overlong: not resumable
		}
		return fmt.Errorf("hub.Download: GET %s: have %d of %d bytes; not publishing", url, size, total)
	}
	return publish(tmp, srcFile, dst)
}

// The .part.url sidecar holds the URL a partial was started from and, on
// a second line, the strong ETag of that response ("" when the server sent
// none or a weak one). Older lights wrote just the URL.
func readPartMeta(path string) (url, etag string, ok bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", "", false
	}
	url, etag, _ = strings.Cut(string(b), "\n")
	return url, strings.TrimSpace(etag), url != ""
}

func writePartMeta(path, url, etag string) error {
	return os.WriteFile(path, []byte(url+"\n"+etag+"\n"), 0o644)
}

// strongETag returns etag if it is a strong validator, usable in If-Range
// (RFC 9110 §13.1.5); weak (W/"...") or empty validators yield "".
func strongETag(etag string) string {
	etag = strings.TrimSpace(etag)
	if len(etag) < 2 || etag[0] != '"' || etag[len(etag)-1] != '"' {
		return ""
	}
	return etag
}

// publish atomically renames a completed .part into place.
func publish(tmp, srcFile, dst string) error {
	if err := os.Rename(tmp, dst); err != nil {
		return err
	}
	_ = os.Remove(srcFile)
	return nil
}

// contentRange is a parsed `Content-Range: bytes start-end/total` header;
// start and end are -1 for `bytes */total`, total is -1 for `/*`.
type contentRange struct{ start, end, total int64 }

func parseContentRange(s string) (contentRange, bool) {
	cr := contentRange{-1, -1, -1}
	rest, ok := strings.CutPrefix(strings.TrimSpace(s), "bytes ")
	if !ok {
		return cr, false
	}
	rng, tot, ok := strings.Cut(rest, "/")
	if !ok {
		return cr, false
	}
	if tot != "*" {
		n, err := strconv.ParseInt(tot, 10, 64)
		if err != nil || n < 0 {
			return cr, false
		}
		cr.total = n
	}
	if rng == "*" {
		return cr, cr.total >= 0
	}
	a, b, ok := strings.Cut(rng, "-")
	if !ok {
		return cr, false
	}
	start, err1 := strconv.ParseInt(a, 10, 64)
	end, err2 := strconv.ParseInt(b, 10, 64)
	if err1 != nil || err2 != nil || start < 0 || end < start || (cr.total >= 0 && end >= cr.total) {
		return cr, false
	}
	cr.start, cr.end = start, end
	return cr, true
}

// lookupHFPythonCache returns the path to a file in HuggingFace Python's
// cache layout if it's present, so we don't re-download artifacts the user
// already pulled with `huggingface_hub`. Layout: `refs/<revision>` contains
// the commit SHA, `snapshots/<sha>/<file>` is a symlink into `blobs/`. A
// ref that is not a commit SHA is ignored. rel is a validated artifactPath.
func lookupHFPythonCache(repoDir, revision, rel string) (string, string, bool) {
	shaBytes, err := os.ReadFile(filepath.Join(repoDir, "refs", filepath.FromSlash(revision)))
	if err != nil {
		return "", "", false
	}
	sha := strings.TrimSpace(string(shaBytes))
	if !isCommitSHA(sha) {
		return "", "", false
	}
	candidate := filepath.Join(repoDir, "snapshots", sha, rel)
	if st, err := os.Stat(candidate); err == nil && st.Size() > 0 {
		return candidate, sha, true
	}
	return "", "", false
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
