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
type Runtime struct {
	mu      sync.Mutex
	model   *Model
	stt     *Transcriber
	emb     *Embedder
	speech  *Speech
	loading bool
	cancel  context.CancelFunc
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

// Load replaces the current model with spec. The previous model is released
// first so its device memory is free before the new weights land.
func (r *Runtime) Load(spec Spec) (*Model, error) {
	r.mu.Lock()
	if r.loading {
		r.mu.Unlock()
		return nil, errors.New("engine: a load is already in progress")
	}
	if r.model != nil && r.model.Busy() {
		r.mu.Unlock()
		return nil, ErrBusy
	}
	r.loading = true
	r.model = nil
	r.speech = nil // Moshi fills the card; the two never coexist
	r.mu.Unlock()
	runtime.GC()
	storage.DrainCUDAPool()

	m, err := Load(spec)
	r.mu.Lock()
	r.loading = false
	if err == nil {
		r.model = m
	}
	r.mu.Unlock()
	if err != nil {
		return nil, err
	}
	return m, nil
}

// Unload drops the model and frees its memory.
func (r *Runtime) Unload() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.model != nil && r.model.Busy() {
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

// Generate runs a chat completion on the loaded model. Stop() cancels it.
func (r *Runtime) Generate(ctx context.Context, msgs []Message, opts GenOpts, onToken func(string)) (string, Stats, error) {
	m := r.Model()
	if m == nil {
		return "", Stats{}, errors.New("engine: no model loaded")
	}
	ctx, cancel := context.WithCancel(ctx)
	r.mu.Lock()
	r.cancel = cancel
	r.mu.Unlock()
	defer func() {
		cancel()
		r.mu.Lock()
		r.cancel = nil
		r.mu.Unlock()
	}()
	return m.Generate(ctx, msgs, opts, onToken)
}

// Stop cancels the running generation, if any.
func (r *Runtime) Stop() {
	r.mu.Lock()
	c := r.cancel
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
		return nil, errors.New("engine: a load is already in progress")
	}
	if r.model != nil && r.model.Busy() {
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
		st.Busy = m.Busy()
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
