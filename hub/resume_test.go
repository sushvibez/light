package hub

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// branchServer is a Hub whose revision API is down (so fetches fall back
// to the mutable branch URL) and whose resolve endpoint serves one object
// that the test can swap, with an ETag and Range / If-Range support.
type branchServer struct {
	mu            sync.Mutex
	body, etag    string
	truncateNext  bool // send 3 bytes of the next response, then drop the connection
	ignoreIfRange bool // honour Range even when If-Range does not match
	reqs          []branchReq
}

type branchReq struct {
	rng, ifRange string
	status       int
}

func newBranchServer(t *testing.T, body, etag string) *branchServer {
	t.Helper()
	b := &branchServer{body: body, etag: etag}
	srv := httptest.NewServer(b)
	t.Cleanup(srv.Close)
	t.Setenv("HF_HOME", t.TempDir())
	t.Setenv("HF_ENDPOINT", srv.URL)
	t.Setenv("HF_HUB_OFFLINE", "") // these tests talk to a fake Hub
	return b
}

func (b *branchServer) set(body, etag string) {
	b.mu.Lock()
	b.body, b.etag = body, etag
	b.mu.Unlock()
}

func (b *branchServer) last() branchReq {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.reqs[len(b.reqs)-1]
}

func (b *branchServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if strings.HasPrefix(r.URL.Path, "/api/") {
		http.Error(w, "metadata unavailable", http.StatusInternalServerError)
		return
	}
	b.mu.Lock()
	body, etag, trunc, ignore := b.body, b.etag, b.truncateNext, b.ignoreIfRange
	b.truncateNext = false
	rec := branchReq{rng: r.Header.Get("Range"), ifRange: r.Header.Get("If-Range")}
	status := http.StatusOK
	from := 0
	if n, ok := strings.CutPrefix(rec.rng, "bytes="); ok && (rec.ifRange == "" || rec.ifRange == etag || ignore) {
		from, _ = strconv.Atoi(strings.TrimSuffix(n, "-"))
		status = http.StatusPartialContent
	}
	rec.status = status
	b.reqs = append(b.reqs, rec)
	b.mu.Unlock()

	w.Header().Set("ETag", etag)
	if status == http.StatusPartialContent {
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", from, len(body)-1, len(body)))
	}
	w.Header().Set("Content-Length", strconv.Itoa(len(body)-from))
	w.WriteHeader(status)
	if trunc {
		_, _ = w.Write([]byte(body[from : from+3]))
		w.(http.Flusher).Flush()
		panic(http.ErrAbortHandler)
	}
	_, _ = w.Write([]byte(body[from:]))
}

// fetchers are the two unpinned paths: Download's legacy-layout fallback
// and DownloadTo's revision fallback, both used when the commit cannot be
// resolved.
func unpinnedFetchers(t *testing.T) map[string]func() (string, error) {
	dst := filepath.Join(t.TempDir(), "out", "model.safetensors")
	return map[string]func() (string, error){
		"Download": func() (string, error) {
			_, err := Download("org/moving", "main", "model.safetensors")
			cache, _ := CacheDir() // the unpinned legacy layout
			return filepath.Join(cache, "models--org--moving", "snapshots", "main", "model.safetensors"), err
		},
		"DownloadTo": func() (string, error) {
			return dst, DownloadTo("org/moving", "main", "model.safetensors", dst, nil)
		},
	}
}

// Review R21: with metadata resolution down, a branch that moves between
// an interrupted fetch and its resume restarts with the new object instead
// of splicing its tail onto the old partial.
func TestUnpinnedResumeRestartsWhenBranchMoves(t *testing.T) {
	for _, ignore := range []bool{false, true} {
		for name, fetch := range unpinnedFetchers(t) {
			t.Run(fmt.Sprintf("%s/ignoreIfRange=%v", name, ignore), func(t *testing.T) {
				b := newBranchServer(t, "OLDOBJECT1", `"v1"`)
				b.truncateNext, b.ignoreIfRange = true, ignore
				if _, err := fetch(); err == nil {
					t.Fatal("truncated fetch succeeded")
				}
				b.set("NEWOBJECT2", `"v2"`) // the branch moves
				p, err := fetch()
				if ignore {
					// A server that ignores If-Range answers 206 with the new
					// ETag: refused, partial dropped, the retry restarts.
					if err == nil {
						t.Fatal("spliced a 206 of another object")
					}
					if _, serr := os.Stat(p + ".part"); serr == nil {
						t.Error("partial of the old object kept")
					}
					if p, err = fetch(); err != nil {
						t.Fatal(err)
					}
				} else if err != nil {
					t.Fatal(err)
				}
				if got := b.reqs[1]; got.rng != "bytes=3-" || got.ifRange != `"v1"` {
					t.Errorf("resume request Range %q If-Range %q", got.rng, got.ifRange)
				}
				if bs, _ := os.ReadFile(p); string(bs) != "NEWOBJECT2" {
					t.Errorf("published %q, want NEWOBJECT2", bs)
				}
			})
		}
	}
}

// An unchanged object resumes: the recorded ETag matches and only the
// rest is transferred.
func TestUnpinnedResumeWithMatchingETag(t *testing.T) {
	for name, fetch := range unpinnedFetchers(t) {
		t.Run(name, func(t *testing.T) {
			b := newBranchServer(t, "SAMEOBJECT", `"v1"`)
			b.truncateNext = true
			if _, err := fetch(); err == nil {
				t.Fatal("truncated fetch succeeded")
			}
			p, err := fetch()
			if err != nil {
				t.Fatal(err)
			}
			if got := b.last(); got.rng != "bytes=3-" || got.ifRange != `"v1"` || got.status != http.StatusPartialContent {
				t.Errorf("resume: %+v", got)
			}
			if bs, _ := os.ReadFile(p); string(bs) != "SAMEOBJECT" {
				t.Errorf("published %q", bs)
			}
		})
	}
}

// Without a strong ETag an unpinned partial has no identity: restart.
func TestUnpinnedResumeWithoutETagRestarts(t *testing.T) {
	b := newBranchServer(t, "OBJECTNOTAG", "")
	b.truncateNext = true
	if _, err := Download("org/moving", "main", "model.safetensors"); err == nil {
		t.Fatal("truncated fetch succeeded")
	}
	p, err := Download("org/moving", "main", "model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if got := b.last(); got.rng != "" {
		t.Errorf("resumed without a validator: %+v", got)
	}
	if bs, _ := os.ReadFile(p); string(bs) != "OBJECTNOTAG" {
		t.Errorf("published %q", bs)
	}
}

// Review R21: a partial with no identity metadata (an older light's
// .part with no .part.url) restarts, pinned or not.
func TestLegacyPartialWithoutMetadataRestarts(t *testing.T) {
	t.Run("unpinned", func(t *testing.T) {
		b := newBranchServer(t, "FRESHBYTES", `"v1"`)
		dst := filepath.Join(t.TempDir(), "model.safetensors")
		if err := os.WriteFile(dst+".part", []byte("OLD"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := DownloadTo("org/moving", "main", "model.safetensors", dst, nil); err != nil {
			t.Fatal(err)
		}
		if got := b.last(); got.rng != "" {
			t.Errorf("legacy partial resumed: %+v", got)
		}
		if bs, _ := os.ReadFile(dst); string(bs) != "FRESHBYTES" {
			t.Errorf("published %q", bs)
		}
	})
	t.Run("pinned", func(t *testing.T) {
		var gotRange string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotRange = r.Header.Get("Range")
			_, _ = w.Write([]byte("FRESHBYTES"))
		}))
		defer srv.Close()
		dst := filepath.Join(t.TempDir(), "f.bin")
		if err := os.WriteFile(dst+".part", []byte("OLD"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := fetchWith(context.Background(), srv.URL+"/resolve/"+shaA+"/f", dst, true, nil); err != nil {
			t.Fatal(err)
		}
		if gotRange != "" {
			t.Errorf("legacy partial resumed with Range %q", gotRange)
		}
		if bs, _ := os.ReadFile(dst); string(bs) != "FRESHBYTES" {
			t.Errorf("published %q", bs)
		}
	})
}

func TestStrongETag(t *testing.T) {
	for in, want := range map[string]string{`"abc"`: `"abc"`, ` "x" `: `"x"`, `W/"abc"`: "", "abc": "", "": "", `"`: ""} {
		if got := strongETag(in); got != want {
			t.Errorf("strongETag(%q) = %q, want %q", in, got, want)
		}
	}
}
