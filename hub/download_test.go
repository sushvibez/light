package hub

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

const (
	shaA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	shaB = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
)

// fakeHub is an httptest Hub: the revision API answers with head (mutable)
// and resolve/<rev>/<file> serves files[file]. Every request path is logged.
type fakeHub struct {
	t     *testing.T
	mu    sync.Mutex
	head  string
	files map[string]string
	log   []string
	// onResolve, when set, runs before a resolve request is answered.
	onResolve func(file string)
}

func newFakeHub(t *testing.T) (*fakeHub, string) {
	t.Helper()
	h := &fakeHub{t: t, head: shaA, files: map[string]string{}}
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	home := t.TempDir()
	t.Setenv("HF_HOME", home)
	t.Setenv("HF_ENDPOINT", srv.URL)
	return h, home
}

func (h *fakeHub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	p := r.URL.EscapedPath()
	h.mu.Lock()
	h.log = append(h.log, p)
	head := h.head
	h.mu.Unlock()
	if strings.HasPrefix(p, "/api/models/") {
		_ = json.NewEncoder(w).Encode(map[string]string{"sha": head})
		return
	}
	_, after, ok := strings.Cut(p, "/resolve/")
	if !ok {
		http.NotFound(w, r)
		return
	}
	_, file, _ := strings.Cut(after, "/")
	if h.onResolve != nil {
		h.onResolve(file)
	}
	h.mu.Lock()
	body, ok := h.files[file]
	h.mu.Unlock()
	if !ok {
		http.NotFound(w, r)
		return
	}
	_, _ = w.Write([]byte(body))
}

func (h *fakeHub) requests() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]string(nil), h.log...)
}

func forgetSHA(repo, rev string) {
	shaCacheMu.Lock()
	delete(shaCache, repo+"@"+rev)
	shaCacheMu.Unlock()
}

// Review R18: a hostile file or shard name must not escape the cache or a
// local checkpoint directory, nor touch the network.
func TestDownloadRejectsPathTraversal(t *testing.T) {
	h, home := newFakeHub(t)
	sentinel := filepath.Join(home, "escaped.bin")
	if err := os.WriteFile(sentinel, []byte("sentinel"), 0o644); err != nil {
		t.Fatal(err)
	}
	h.files["../../../../escaped.bin"] = "PWNED"
	bad := []string{
		"../../../../escaped.bin", "a/../../escaped.bin", "..", "./x", "a//b",
		`..\..\..\..\escaped.bin`, `sub\..\..\escaped.bin`,
		"C:/escaped.bin", `C:\escaped.bin`, "C:escaped.bin", "/escaped.bin", `\\host\share\escaped.bin`,
		"x\x00y", "",
	}
	for _, name := range bad {
		if p, err := Download("org/traversal", "main", name); err == nil {
			t.Errorf("Download(%q) = %s, want error", name, p)
		}
	}
	if reqs := h.requests(); len(reqs) != 0 {
		t.Errorf("rejected names reached the network: %v", reqs)
	}
	for _, rev := range []string{"../x", `..\x`, "C:/x", "/x"} {
		if _, err := Download("org/traversal", rev, "config.json"); err == nil {
			t.Errorf("revision %q: want error", rev)
		}
	}
	if _, err := Download(`..\..\evil`, "main", "config.json"); err == nil {
		t.Error("repo id with backslashes: want error")
	}

	// Local checkpoint directory.
	dir := filepath.Join(home, "ckpt")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"../escaped.bin", `..\escaped.bin`} {
		if p, err := Download(dir, "", name); err == nil {
			t.Errorf("local Download(%q) = %s, want error", name, p)
		}
	}
	// Malicious shard index, from a local checkpoint and from the Hub.
	index := `{"weight_map": {"a": "model-00001.safetensors", "b": "../../../../escaped.bin"}}`
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(index), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := DownloadModelSafetensors(dir, ""); err == nil || !strings.Contains(err.Error(), "invalid repository path") {
		t.Errorf("local malicious index: err = %v", err)
	}
	h.files["model.safetensors.index.json"] = index
	h.files["model-00001.safetensors"] = "shard"
	if _, err := DownloadModelSafetensors("org/traversal", "main"); err == nil || !strings.Contains(err.Error(), "invalid repository path") {
		t.Errorf("hub malicious index: err = %v", err)
	}
	for _, r := range h.requests() {
		if strings.Contains(r, "escaped") || strings.Contains(r, "model-00001") {
			t.Errorf("index with a bad shard still fetched %s", r)
		}
	}
	if b, _ := os.ReadFile(sentinel); string(b) != "sentinel" {
		t.Errorf("outside sentinel changed: %q", b)
	}
	if _, err := os.Stat(filepath.Join(home, "hub", "escaped.bin")); err == nil {
		t.Error("escaped.bin written under the cache root")
	}
}

// A refs file that is not a commit SHA must not become a path component.
func TestHostileRefIgnored(t *testing.T) {
	h, home := newFakeHub(t)
	h.files["config.json"] = "{}"
	repoDir := filepath.Join(home, "hub", "models--org--badref")
	if err := os.MkdirAll(filepath.Join(repoDir, "refs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repoDir, "refs", "main"), []byte("../../.."), 0o644); err != nil {
		t.Fatal(err)
	}
	p, err := Download("org/badref", "main", "config.json")
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(repoDir, "snapshots", shaA, "config.json"); p != want {
		t.Errorf("path %s, want %s", p, want)
	}
}

// Review R19: every file of a load comes from the resolved commit, even
// when the branch moves after resolution and between shard requests.
func TestDownloadPinsResolvedCommit(t *testing.T) {
	h, home := newFakeHub(t)
	const repo = "org/pinned"
	h.files["model.safetensors.index.json"] = `{"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}}`
	h.files["s1.safetensors"] = "one"
	h.files["s2.safetensors"] = "two"
	next := shaB
	h.onResolve = func(string) {
		// The branch moves on every artifact request; forget the
		// in-process resolution so a re-resolve would see it.
		h.mu.Lock()
		h.head = next
		h.mu.Unlock()
		forgetSHA(repo, "main")
	}
	paths, err := DownloadModelSafetensors(repo, "main")
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) != 2 {
		t.Fatalf("paths %v", paths)
	}
	snap := filepath.Join(home, "hub", "models--org--pinned", "snapshots", shaA)
	for _, p := range paths {
		if !strings.HasPrefix(p, snap) {
			t.Errorf("%s not under snapshots/%s", p, shaA)
		}
	}
	for _, r := range h.requests() {
		if strings.Contains(r, "/resolve/") && !strings.Contains(r, "/resolve/"+shaA+"/") {
			t.Errorf("fetched from a mutable or moved revision: %s", r)
		}
	}
}

// Review R22: a revision with slashes resolves, writes refs/refs/pr/1 and
// is served from the cache on the next call.
func TestDownloadRevisionWithSlashes(t *testing.T) {
	h, home := newFakeHub(t)
	h.files["config.json"] = "{}"
	const repo, rev = "org/prrev", "refs/pr/1"
	p, err := Download(repo, rev, "config.json")
	if err != nil {
		t.Fatal(err)
	}
	repoDir := filepath.Join(home, "hub", "models--org--prrev")
	if want := filepath.Join(repoDir, "snapshots", shaA, "config.json"); p != want {
		t.Errorf("path %s, want %s", p, want)
	}
	if b, err := os.ReadFile(filepath.Join(repoDir, "refs", "refs", "pr", "1")); err != nil || string(b) != shaA {
		t.Errorf("ref: %q %v", b, err)
	}
	reqs := h.requests()
	if len(reqs) == 0 || !strings.HasSuffix(reqs[0], "/revision/refs%2Fpr%2F1") {
		t.Errorf("revision not escaped as one segment: %v", reqs)
	}
	forgetSHA(repo, rev)
	n := len(h.requests())
	p2, err := Download(repo, rev, "config.json")
	if err != nil || p2 != p {
		t.Errorf("cache hit: %s %v", p2, err)
	}
	if len(h.requests()) != n {
		t.Errorf("cache hit went to the network: %v", h.requests()[n:])
	}
}

// Review R21: a 206 must continue the partial exactly; anything else is
// rejected without publishing.
func TestResumeValidatesContentRange(t *testing.T) {
	type resp struct {
		status int
		hdr    map[string]string
		body   string
	}
	cases := []struct {
		name      string
		part      string
		r         resp
		wantErr   bool
		want      string // published content when !wantErr
		keepsPart bool   // .part survives a failure (resumable)
	}{
		{"wrong start", "ABC", resp{206, map[string]string{"Content-Range": "bytes 0-2/3"}, "XYZ"}, true, "", false},
		{"missing range", "ABC", resp{206, nil, "XYZ"}, true, "", false},
		{"short of total", "ABC", resp{206, map[string]string{"Content-Range": "bytes 3-5/9"}, "DEF"}, true, "", false},
		{"unknown total", "ABC", resp{206, map[string]string{"Content-Range": "bytes 3-5/*"}, "DEF"}, true, "", false},
		{"valid resume", "ABC", resp{206, map[string]string{"Content-Range": "bytes 3-5/6"}, "DEF"}, false, "ABCDEF", false},
		{"server restarts", "ABC", resp{200, nil, "ABCDEF"}, false, "ABCDEF", false},
		{"already complete", "ABC", resp{416, map[string]string{"Content-Range": "bytes */3"}, ""}, false, "ABC", false},
		{"416 mismatch", "ABC", resp{416, map[string]string{"Content-Range": "bytes */9"}, ""}, true, "", false},
		{"truncated body", "ABC", resp{206, map[string]string{"Content-Range": "bytes 3-5/6", "Content-Length": "3"}, "D"}, true, "", true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var gotRange string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotRange = r.Header.Get("Range")
				for k, v := range c.r.hdr {
					w.Header().Set(k, v)
				}
				w.WriteHeader(c.r.status)
				_, _ = w.Write([]byte(c.r.body))
			}))
			defer srv.Close()
			dst := filepath.Join(t.TempDir(), "f.bin")
			if err := os.WriteFile(dst+".part", []byte(c.part), 0o644); err != nil {
				t.Fatal(err)
			}
			err := fetchWith(context.Background(), srv.URL+"/f", dst, nil)
			if gotRange != fmt.Sprintf("bytes=%d-", len(c.part)) {
				t.Errorf("Range %q", gotRange)
			}
			if c.wantErr {
				if err == nil {
					t.Fatal("want error")
				}
				if _, err := os.Stat(dst); err == nil {
					t.Error("corrupt file published")
				}
				if _, err := os.Stat(dst + ".part"); (err == nil) != c.keepsPart {
					t.Errorf(".part kept = %v, want %v", err == nil, c.keepsPart)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if b, _ := os.ReadFile(dst); string(b) != c.want {
				t.Errorf("published %q, want %q", b, c.want)
			}
		})
	}
}

// A partial recorded from a different URL (another commit) is restarted,
// not resumed.
func TestResumeDiscardsPartialFromOtherObject(t *testing.T) {
	var gotRange string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotRange = r.Header.Get("Range")
		_, _ = w.Write([]byte("NEWFILE"))
	}))
	defer srv.Close()
	dst := filepath.Join(t.TempDir(), "f.bin")
	_ = os.WriteFile(dst+".part", []byte("OLD"), 0o644)
	_ = os.WriteFile(dst+".part.url", []byte(srv.URL+"/other"), 0o644)
	if err := fetchWith(context.Background(), srv.URL+"/f", dst, nil); err != nil {
		t.Fatal(err)
	}
	if gotRange != "" {
		t.Errorf("resumed a foreign partial with Range %q", gotRange)
	}
	if b, _ := os.ReadFile(dst); string(b) != "NEWFILE" {
		t.Errorf("got %q", b)
	}
	if _, err := os.Stat(dst + ".part.url"); err == nil {
		t.Error(".part.url left behind")
	}
}

// Review R20: overlapping callers for one file share a single download and
// all see the complete bytes.
func TestConcurrentDownloadsSerialize(t *testing.T) {
	var gets atomic.Int32
	body := strings.Repeat("0123456789", 10000)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gets.Add(1)
		w.Header().Set("Content-Length", fmt.Sprint(len(body)))
		_, _ = w.Write([]byte(body[:len(body)/2]))
		w.(http.Flusher).Flush()
		time.Sleep(50 * time.Millisecond)
		_, _ = w.Write([]byte(body[len(body)/2:]))
	}))
	defer srv.Close()
	t.Setenv("HF_HOME", t.TempDir())
	t.Setenv("HF_ENDPOINT", srv.URL)
	const n = 6
	paths, errs := make([]string, n), make([]error, n)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			paths[i], errs[i] = Download("org/concurrent", shaA, "model.safetensors")
		}()
	}
	close(start)
	wg.Wait()
	for i := range n {
		if errs[i] != nil {
			t.Fatalf("caller %d: %v", i, errs[i])
		}
		if paths[i] != paths[0] {
			t.Errorf("caller %d path %s != %s", i, paths[i], paths[0])
		}
	}
	if b, _ := os.ReadFile(paths[0]); string(b) != body {
		t.Errorf("published %d bytes, want %d", len(b), len(body))
	}
	if g := gets.Load(); g != 1 {
		t.Errorf("%d GETs for one file, want 1", g)
	}
}

func TestParseContentRange(t *testing.T) {
	good := map[string]contentRange{
		"bytes 3-5/6":  {3, 5, 6},
		"bytes 0-0/*":  {0, 0, -1},
		"bytes */10":   {-1, -1, 10},
		" bytes 1-1/2": {1, 1, 2},
	}
	for s, want := range good {
		if got, ok := parseContentRange(s); !ok || got != want {
			t.Errorf("%q: %v %v", s, got, ok)
		}
	}
	for _, s := range []string{"", "bytes", "bytes 5-3/9", "bytes 3-9/9", "bytes -1-2/3", "bytes */*", "items 0-1/2", "bytes a-b/c"} {
		if _, ok := parseContentRange(s); ok {
			t.Errorf("%q accepted", s)
		}
	}
}
