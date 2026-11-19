package engine

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"

	"light/internal/cudart"
	"light/storage"
)

// Runtime owns at most one loaded model at a time (a 16 GB card has no room
// for two) and serialises load/unload against generation.
//
// A request holds the chat model through a Lease, taken atomically with the
// model's selection (Acquire, LoadAndAcquire) and kept until the request
// ends: while a lease is out the model cannot be replaced or unloaded and
// no other request can generate, and while a load is in progress no lease
// can be taken.
type Runtime struct {
	mu      sync.Mutex
	model   *Model
	stt     *Transcriber
	emb     *Embedder
	speech  *Speech
	loading bool
	lease   *Lease // the request holding the chat model, if any

	// Test seams: nil means Load and (*Model).Generate.
	loadFn func(Spec) (*Model, error)
	genFn  func(m *Model, ctx context.Context, msgs []Message, opts GenOpts, onToken func(string)) (string, Stats, error)
}

// ErrNoModel is returned by Acquire when no chat model is loaded, or the
// loaded one is not the requested one.
var ErrNoModel = errors.New("engine: requested model is not loaded")

// Lease is one request's hold on the loaded chat model; see Runtime.
type Lease struct {
	rt     *Runtime
	m      *Model
	cancel context.CancelFunc // the running generation's, guarded by rt.mu
}

// Status is what the UI polls.
type Status struct {
	Loaded   bool    `json:"loaded"`
	Loading  bool    `json:"loading"`
	Busy     bool    `json:"busy"`
	Spec     *Spec   `json:"spec,omitempty"`
	LoadMS   float64 `json:"load_ms"`
	GPUUsed  uint64  `json:"gpu_used_mb"`
	GPUTotal uint64  `json:"gpu_total_mb"`
	HostMB   uint64  `json:"host_mb"`
	Params   float64 `json:"params_m"`
	Context  int     `json:"context"`
	Window   int     `json:"sliding_window"`
	STT      *Spec   `json:"stt,omitempty"`      // loaded speech-to-text model
	Embedder *Spec   `json:"embedder,omitempty"` // loaded text-embedding model
	Speech   *Spec   `json:"speech,omitempty"`   // loaded speech-to-speech model (Moshi)
}

func NewRuntime() *Runtime { return &Runtime{} }

// busyLocked reports whether the chat model is held: leased, or generating
// through a direct (*Model).Generate call. r.mu must be held.
func (r *Runtime) busyLocked() bool {
	return r.lease != nil || (r.model != nil && r.model.Busy())
}

// Load replaces the current model with spec. The previous model is released
// first so its device memory is free before the new weights land.
func (r *Runtime) Load(spec Spec) (*Model, error) {
	m, _, err := r.load(spec, false)
	return m, err
}

// LoadAndAcquire is Load that also leases the new model before any other
// request can see it. The caller must Release the lease.
func (r *Runtime) LoadAndAcquire(spec Spec) (*Lease, error) {
	_, l, err := r.load(spec, true)
	return l, err
}

func (r *Runtime) load(spec Spec, lease bool) (*Model, *Lease, error) {
	r.mu.Lock()
	if r.loading {
		r.mu.Unlock()
		return nil, nil, fmt.Errorf("%w: a load is already in progress", ErrBusy)
	}
	if r.busyLocked() {
		r.mu.Unlock()
		return nil, nil, ErrBusy
	}
	r.loading = true
	r.model = nil
	r.speech = nil // Moshi fills the card; the two never coexist
	r.mu.Unlock()
	runtime.GC()
	storage.DrainCUDAPool()

	load := r.loadFn
	if load == nil {
		load = Load
	}
	m, err := load(spec)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.loading = false
	if err != nil {
		return nil, nil, err
	}
	r.model = m
	var l *Lease
	if lease {
		l = &Lease{rt: r, m: m}
		r.lease = l
	}
	return m, l, nil
}

// Unload drops the model and frees its memory.
func (r *Runtime) Unload() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.busyLocked() {
		return ErrBusy
	}
	r.model = nil
	runtime.GC()
	storage.DrainCUDAPool()
	return nil
}

// Model returns the loaded model or nil.
func (r *Runtime) Model() *Model {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.model
}

// Acquire leases the loaded chat model if want accepts it (nil: any
// model). It fails with ErrBusy while a load is in progress or another
// request holds the model, and with ErrNoModel when no model is loaded or
// want rejects it. The caller must Release the lease.
func (r *Runtime) Acquire(want func(*Model) bool) (*Lease, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.loading {
		return nil, fmt.Errorf("%w: a load is in progress", ErrBusy)
	}
	if r.busyLocked() {
		return nil, ErrBusy
	}
	if r.model == nil || (want != nil && !want(r.model)) {
		return nil, ErrNoModel
	}
	l := &Lease{rt: r, m: r.model}
	r.lease = l
	return l, nil
}

// Model is the leased model.
func (l *Lease) Model() *Model { return l.m }

// Release ends the lease; later calls are no-ops.
func (l *Lease) Release() {
	r := l.rt
	r.mu.Lock()
	if r.lease == l {
		r.lease = nil
	}
	r.mu.Unlock()
}

// Generate runs a chat completion on the leased model; Runtime.Stop
// cancels it. One generation per lease at a time.
func (l *Lease) Generate(ctx context.Context, msgs []Message, opts GenOpts, onToken func(string)) (string, Stats, error) {
	r := l.rt
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	r.mu.Lock()
	if r.lease != l {
		r.mu.Unlock()
		return "", Stats{}, errors.New("engine: generate on a released lease")
	}
	if l.cancel != nil {
		r.mu.Unlock()
		return "", Stats{}, ErrBusy
	}
	l.cancel = cancel
	r.mu.Unlock()
	defer func() {
		r.mu.Lock()
		l.cancel = nil
		r.mu.Unlock()
	}()
	if r.genFn != nil {
		return r.genFn(l.m, ctx, msgs, opts, onToken)
	}
	return l.m.Generate(ctx, msgs, opts, onToken)
}

// Generate runs a chat completion on the loaded model, holding a lease for
// its duration. Stop() cancels it.
func (r *Runtime) Generate(ctx context.Context, msgs []Message, opts GenOpts, onToken func(string)) (string, Stats, error) {
	l, err := r.Acquire(nil)
	if errors.Is(err, ErrNoModel) {
		return "", Stats{}, errors.New("engine: no model loaded")
	}
	if err != nil {
		return "", Stats{}, err
	}
	defer l.Release()
	return l.Generate(ctx, msgs, opts, onToken)
}

// Stop cancels the running generation, if any. Only the lease holder's
// generation is ever cancelled; a rejected request never touches it.
func (r *Runtime) Stop() {
	r.mu.Lock()
	var c context.CancelFunc
	if r.lease != nil {
		c = r.lease.cancel
	}
	r.mu.Unlock()
	if c != nil {
		c()
	}
}

// LoadSTT loads (or replaces) the speech-to-text model. Whisper is small
// enough to sit beside the chat model.
func (r *Runtime) LoadSTT(spec Spec) (*Transcriber, error) {
	r.mu.Lock()
	r.stt = nil
	r.mu.Unlock()
	runtime.GC()
	t, err := LoadTranscriber(spec)
	if err != nil {
		return nil, err
	}
	r.mu.Lock()
	r.stt = t
	r.mu.Unlock()
	return t, nil
}

// STT returns the loaded transcriber or nil.
func (r *Runtime) STT() *Transcriber {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.stt
}

// UnloadSTT frees the transcriber.
func (r *Runtime) UnloadSTT() {
	r.mu.Lock()
	r.stt = nil
	r.mu.Unlock()
	runtime.GC()
}

// LoadEmbedder loads (or replaces) the text-embedding model, which is
// small enough (270M) to sit beside the chat model.
func (r *Runtime) LoadEmbedder(spec Spec) (*Embedder, error) {
	r.mu.Lock()
	r.emb = nil
	r.mu.Unlock()
	runtime.GC()
	e, err := LoadEmbedder(spec)
	if err != nil {
		return nil, err
	}
	r.mu.Lock()
	r.emb = e
	r.mu.Unlock()
	return e, nil
}

// Embedder returns the loaded text-embedding model or nil.
func (r *Runtime) Embedder() *Embedder {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.emb
}

// UnloadEmbedder frees the text-embedding model.
func (r *Runtime) UnloadEmbedder() {
	r.mu.Lock()
	r.emb = nil
	r.mu.Unlock()
	runtime.GC()
}

// LoadSpeech loads (or replaces) the Moshi speech-to-speech stack. Its 8.2
// GB of weights need the whole card, so the chat model is released first
// (and Load releases Moshi in turn).
func (r *Runtime) LoadSpeech(spec Spec) (*Speech, error) {
	r.mu.Lock()
	if r.loading {
		r.mu.Unlock()
		return nil, fmt.Errorf("%w: a load is already in progress", ErrBusy)
	}
	if r.busyLocked() {
		r.mu.Unlock()
		return nil, ErrBusy
	}
	r.loading = true
	r.model = nil
	r.speech = nil
	r.mu.Unlock()
	runtime.GC()
	storage.DrainCUDAPool()
	s, err := LoadSpeech(spec)
	r.mu.Lock()
	r.loading = false
	if err == nil {
		r.speech = s
	}
	r.mu.Unlock()
	return s, err
}

// Speech returns the loaded speech-to-speech stack or nil.
func (r *Runtime) Speech() *Speech {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.speech
}

// UnloadSpeech frees the speech-to-speech stack.
func (r *Runtime) UnloadSpeech() {
	r.mu.Lock()
	r.speech = nil
	r.mu.Unlock()
	runtime.GC()
	storage.DrainCUDAPool()
}

// Status snapshots load state and memory use.
func (r *Runtime) Status() Status {
	r.mu.Lock()
	m, loading, stt, emb, speech := r.model, r.loading, r.stt, r.emb, r.speech
	busy := r.busyLocked()
	r.mu.Unlock()
	var st Status
	st.Loading = loading
	if speech != nil {
		sp := speech.Spec
		st.Speech = &sp
	}
	if stt != nil {
		sp := stt.Spec
		st.STT = &sp
	}
	if emb != nil {
		sp := emb.Spec
		st.Embedder = &sp
	}
	if m != nil {
		st.Loaded = true
		st.Busy = busy
		sp := m.Spec
		st.Spec = &sp
		st.LoadMS = float64(m.LoadTime.Milliseconds())
		st.Params = float64(m.LM.ParamCount()) / 1e6
		st.Context = m.LM.MaxContext()
		st.Window = m.LM.SlidingWindow()
	}
	if free, total, err := cudart.MemInfo(); err == nil {
		st.GPUUsed = uint64(total-free) >> 20
		st.GPUTotal = uint64(total) >> 20
	}
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	st.HostMB = ms.HeapAlloc >> 20
	return st
}

// Describe is a one-line summary for logs.
func (r *Runtime) Describe() string {
	m := r.Model()
	if m == nil {
		return "no model"
	}
	return fmt.Sprintf("%s (%s, %s)", m.Spec.ID, m.Spec.Family, m.Dev)
}
