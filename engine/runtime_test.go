package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"light/audio"
)

// fakeRuntime is a Runtime with model "a" loaded, fake loads (no
// checkpoints) and a generation that blocks until its context is
// cancelled, reporting entry on started.
func fakeRuntime(started chan<- string) *Runtime {
	rt := NewRuntime()
	rt.model = &Model{Spec: Spec{ID: "a"}}
	rt.loadFn = func(s Spec) (*Model, error) { return &Model{Spec: s}, nil }
	rt.genFn = func(m *Model, ctx context.Context, _ []Message, _ GenOpts, _ func(string)) (string, Stats, error) {
		if started == nil {
			return "reply from " + m.Spec.ID, Stats{Stop: "eos"}, nil
		}
		started <- m.Spec.ID
		<-ctx.Done()
		return "", Stats{Stop: "cancel"}, ctx.Err()
	}
	return rt
}

// R5: a request rejected with ErrBusy must not touch the active request's
// cancellation handle; Stop still cancels the active one.
func TestRuntimeRejectedGenerateKeepsStop(t *testing.T) {
	started := make(chan string)
	rt := fakeRuntime(started)
	done := make(chan error)
	go func() {
		_, _, err := rt.Generate(context.Background(), nil, GenOpts{}, nil)
		done <- err
	}()
	if id := <-started; id != "a" { // A is now inside generation
		t.Fatalf("generating on %q", id)
	}
	for range 3 {
		if _, _, err := rt.Generate(context.Background(), nil, GenOpts{}, nil); !errors.Is(err, ErrBusy) {
			t.Fatalf("second request: %v, want ErrBusy", err)
		}
	}
	rt.Stop()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("A after Stop: %v, want context.Canceled", err)
	}
	// The slot is free again.
	rt.genFn = fakeRuntime(nil).genFn
	if text, _, err := rt.Generate(context.Background(), nil, GenOpts{}, nil); err != nil || text != "reply from a" {
		t.Fatalf("after A: %q %v", text, err)
	}
}

// R6: while a request holds the model, nothing can replace or unload it,
// and a lease cannot be taken while a load is in progress.
func TestRuntimeLeasePinsModel(t *testing.T) {
	rt := fakeRuntime(nil)
	l, err := rt.Acquire(func(m *Model) bool { return m.Spec.ID == "a" })
	if err != nil {
		t.Fatal(err)
	}
	if _, err := rt.Load(Spec{ID: "b"}); !errors.Is(err, ErrBusy) {
		t.Fatalf("Load during lease: %v", err)
	}
	if _, err := rt.LoadAndAcquire(Spec{ID: "b"}); !errors.Is(err, ErrBusy) {
		t.Fatalf("LoadAndAcquire during lease: %v", err)
	}
	if _, err := rt.LoadSpeech(Spec{ID: "moshi"}); !errors.Is(err, ErrBusy) {
		t.Fatalf("LoadSpeech during lease: %v", err)
	}
	if err := rt.Unload(); !errors.Is(err, ErrBusy) {
		t.Fatalf("Unload during lease: %v", err)
	}
	if _, err := rt.Acquire(nil); !errors.Is(err, ErrBusy) {
		t.Fatalf("second Acquire: %v", err)
	}
	if m := rt.Model(); m != l.Model() || m.Spec.ID != "a" {
		t.Fatalf("model replaced under the lease: %v", m.Spec.ID)
	}
	if text, _, err := l.Generate(context.Background(), nil, GenOpts{}, nil); err != nil || text != "reply from a" {
		t.Fatalf("lease generate: %q %v", text, err)
	}
	l.Release()
	l.Release() // idempotent
	if _, _, err := l.Generate(context.Background(), nil, GenOpts{}, nil); err == nil {
		t.Fatal("generate on a released lease succeeded")
	}
	if _, err := rt.Acquire(func(m *Model) bool { return m.Spec.ID == "b" }); !errors.Is(err, ErrNoModel) {
		t.Fatalf("Acquire of an unloaded model: %v", err)
	}

	// A load holds off every lease until the new model is in place, and
	// LoadAndAcquire hands its caller the new model before anyone else.
	loadStarted, finishLoad := make(chan struct{}), make(chan struct{})
	rt.loadFn = func(s Spec) (*Model, error) {
		close(loadStarted)
		<-finishLoad
		return &Model{Spec: s}, nil
	}
	got := make(chan *Lease)
	go func() {
		l, err := rt.LoadAndAcquire(Spec{ID: "b"})
		if err != nil {
			t.Error(err)
		}
		got <- l
	}()
	<-loadStarted
	if _, err := rt.Acquire(nil); !errors.Is(err, ErrBusy) {
		t.Fatalf("Acquire during load: %v", err)
	}
	if _, _, err := rt.Generate(context.Background(), nil, GenOpts{}, nil); !errors.Is(err, ErrBusy) {
		t.Fatalf("Generate during load: %v", err)
	}
	if _, err := rt.Load(Spec{ID: "c"}); !errors.Is(err, ErrBusy) {
		t.Fatalf("Load during load: %v", err)
	}
	close(finishLoad)
	lb := <-got
	if lb == nil || lb.Model().Spec.ID != "b" || rt.Model() != lb.Model() {
		t.Fatalf("LoadAndAcquire lease: %+v", lb)
	}
	if _, err := rt.Acquire(nil); !errors.Is(err, ErrBusy) {
		t.Fatalf("Acquire after LoadAndAcquire: %v", err)
	}
	lb.Release()
}

// The HTTP handler generates with the model it selected and reports that
// model, and a request arriving while another holds the model is refused.
func TestOpenAIChatUsesLeasedModel(t *testing.T) {
	started := make(chan string)
	rt := fakeRuntime(started)
	h := OpenAIHandler(rt)
	served := make(chan struct{}, 4) // one per finished request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.ServeHTTP(w, r)
		served <- struct{}{}
	}))
	defer srv.Close()
	post := func(ctx context.Context, body string) (*http.Response, error) {
		req, _ := http.NewRequestWithContext(ctx, "POST", srv.URL+"/v1/chat/completions", strings.NewReader(body))
		return http.DefaultClient.Do(req)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		if resp, err := post(ctx, `{"model":"a","messages":[{"role":"user","content":"hi"}]}`); err == nil {
			resp.Body.Close()
		}
	}()
	<-started // request A holds the model inside generation
	resp, err := post(context.Background(), `{"model":"a","messages":[{"role":"user","content":"hi"}]}`)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 503 {
		t.Fatalf("concurrent request: status %d, want 503", resp.StatusCode)
	}
	<-served // the refused request's handler
	cancel() // client goes away: A's generation is cancelled, its lease released
	<-done
	<-served // A's handler has returned, so its lease is released

	rt.genFn = fakeRuntime(nil).genFn
	resp, err = post(context.Background(), `{"messages":[{"role":"user","content":"hi"}]}`)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Model   string `json:"model"`
		Choices []struct {
			Message Message `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Model != "a" || len(out.Choices) != 1 || out.Choices[0].Message.Content != "reply from a" {
		t.Fatalf("response %+v", out)
	}
}

// R16: request bodies and their contents are bounded.
func TestOpenAIRequestLimits(t *testing.T) {
	rt := fakeRuntime(nil)
	lim := Limits{JSONBytes: 1 << 10, UploadBytes: 4 << 10, AudioSeconds: 1, ConverseSeconds: 1, EmbedInputs: 4}
	srv := httptest.NewServer(OpenAIHandlerLimits(rt, lim))
	defer srv.Close()
	status := func(path, ctype string, body []byte) int {
		t.Helper()
		resp, err := http.Post(srv.URL+path, ctype, bytes.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}
	big := strings.Repeat("x", 4<<10)
	chat := fmt.Sprintf(`{"messages":[{"role":"user","content":%q}]}`, big)
	if c := status("/v1/chat/completions", "application/json", []byte(chat)); c != 413 {
		t.Errorf("oversized chat JSON: %d, want 413", c)
	}
	if c := status("/v1/embeddings", "application/json", []byte(fmt.Sprintf(`{"input":%q}`, big))); c != 413 {
		t.Errorf("oversized embeddings JSON: %d, want 413", c)
	}
	if c := status("/v1/embeddings", "application/json", []byte(`{"input":["a","b","c","d","e"]}`)); c != 413 {
		t.Errorf("too many embedding inputs: %d, want 413", c)
	}
	for _, body := range []string{
		`{"messages":[{"role":"user","content":"hi"}],"max_tokens":-1}`,
		`{"messages":[{"role":"user","content":"hi"}],"max_tokens":1000000000}`,
		`{"messages":[{"role":"user","content":"hi"}],"top_p":1.5}`,
		`{"messages":[{"role":"user","content":"hi"}],"temperature":-1}`,
	} {
		if c := status("/v1/chat/completions", "application/json", []byte(body)); c != 400 {
			t.Errorf("%s: %d, want 400", body, c)
		}
	}

	upload := func(path string, wav []byte, fields map[string]string) int {
		var b bytes.Buffer
		mw := multipart.NewWriter(&b)
		fw, _ := mw.CreateFormFile("file", "in.wav")
		_, _ = fw.Write(wav)
		for k, v := range fields {
			_ = mw.WriteField(k, v)
		}
		_ = mw.Close()
		return status(path, mw.FormDataContentType(), b.Bytes())
	}
	wavOf := func(n, rate int) []byte {
		var b bytes.Buffer
		_ = audio.WriteWAV16(&b, make([]float32, n), rate)
		return b.Bytes()
	}
	for _, path := range []string{"/v1/audio/transcriptions", "/v1/audio/conversations"} {
		if c := upload(path, wavOf(8<<10, 16000), nil); c != 413 { // 16 KiB file > 4 KiB body
			t.Errorf("%s oversized upload: %d, want 413", path, c)
		}
		if c := upload(path, wavOf(1500, 1000), nil); c != 413 { // 1.5 s > 1 s
			t.Errorf("%s overlong audio: %d, want 413", path, c)
		}
		if c := upload(path, []byte("RIFF....WAVEjunk"), nil); c != 400 {
			t.Errorf("%s malformed WAV: %d, want 400", path, c)
		}
	}
	for _, v := range []string{"-1", "NaN", "1e9", "abc"} {
		if c := upload("/v1/audio/conversations", wavOf(100, 1000), map[string]string{"extra_seconds": v}); c != 400 {
			t.Errorf("extra_seconds=%s: %d, want 400", v, c)
		}
	}
}
