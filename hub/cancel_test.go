package hub

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Review Q8: canceling the context aborts a body that has stalled, and the
// bytes so far stay behind as a resumable partial.
func TestDownloadCancelDuringSlowBody(t *testing.T) {
	stop := make(chan struct{})
	defer close(stop)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1000")
		_, _ = w.Write([]byte(strings.Repeat("x", 10)))
		w.(http.Flusher).Flush()
		select { // stall: never send the rest
		case <-r.Context().Done():
		case <-stop:
		}
	}))
	defer srv.Close()
	t.Setenv("HF_HOME", t.TempDir())
	t.Setenv("HF_ENDPOINT", srv.URL)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	progress := func(done, total int64) {
		if done > 0 {
			cancel() // first bytes arrived; the rest never will
		}
	}
	errc := make(chan error, 1)
	go func() {
		_, err := DownloadWithProgressContext(ctx, "org/slow", shaA, "model.safetensors", progress)
		errc <- err
	}()
	select {
	case err := <-errc:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("download did not return after cancel")
	}
	cache, _ := CacheDir()
	dst := filepath.Join(cache, "models--org--slow", "snapshots", shaA, "model.safetensors")
	if _, err := os.Stat(dst); err == nil {
		t.Error("canceled download was published")
	}
	if st, err := os.Stat(dst + ".part"); err != nil || st.Size() != 10 {
		t.Errorf("partial after cancel: %v %v", st, err)
	}
}

// Review Q8: a caller waiting for another's download of the same file can
// give up.
func TestDownloadCancelWhileWaitingOnLock(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("waiting caller reached the network: %s", r.URL)
	}))
	defer srv.Close()
	t.Setenv("HF_HOME", t.TempDir())
	t.Setenv("HF_ENDPOINT", srv.URL)
	cache, _ := CacheDir()
	dst := filepath.Join(cache, "models--org--locked", "snapshots", shaA, "config.json")
	unlock, err := lockDest(context.Background(), dst)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err = DownloadContext(ctx, "org/locked", shaA, "config.json")
	unlock() // the holder can still release, and the lock is usable afterwards
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context.DeadlineExceeded", err)
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("returned after %v", d)
	}
	u2, err := lockDest(context.Background(), dst)
	if err != nil {
		t.Fatal(err)
	}
	u2()
}

// Review Q8: only a real 404 status selects the sharded fallback; a 503
// whose body says "HTTP 404" is an error to report.
func TestHTTPNotFoundIsTyped(t *testing.T) {
	h, _ := newFakeHub(t)
	h.files["model.safetensors.index.json"] = `{"weight_map": {"a": "s1.safetensors"}}`
	h.files["s1.safetensors"] = "one"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/org/flaky/") && strings.HasSuffix(r.URL.Path, "/model.safetensors") {
			http.Error(w, "upstream said HTTP 404", http.StatusServiceUnavailable)
			return
		}
		h.ServeHTTP(w, r)
	}))
	defer srv.Close()
	t.Setenv("HF_ENDPOINT", srv.URL)

	_, err := DownloadModelSafetensors("org/flaky", shaA)
	var he *HTTPError
	if !errors.As(err, &he) || he.Status != http.StatusServiceUnavailable {
		t.Fatalf("err = %v, want the 503", err)
	}
	if isHTTPNotFound(err) {
		t.Error("503 treated as not found")
	}
	for _, r := range h.requests() {
		if strings.Contains(r, "index.json") {
			t.Errorf("fell back to the sharded layout on a 503: %s", r)
		}
	}
	// A real 404 (no model.safetensors in the fake hub) still falls back.
	paths, err := DownloadModelSafetensors("org/sharded404", shaA)
	if err != nil || len(paths) != 1 {
		t.Fatalf("404 fallback: %v %v", paths, err)
	}
	if !isHTTPNotFound(fmt.Errorf("wrapped: %w", &HTTPError{Status: 404})) {
		t.Error("wrapped 404 not detected")
	}
}
