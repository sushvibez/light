// Package engine turns the models/tokenizer packages into a chat runtime:
// load a checkpoint (HF safetensors or GGUF) once, then stream replies for
// OpenAI-style message lists. It is the backend of the Light Studio desktop
// app and of the OpenAI-compatible HTTP server.
package engine

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/models"
	"light/nn"
	"light/tensor"
	"light/tokenizer"
)

// Spec identifies a loadable chat model.
type Spec struct {
	ID     string `json:"id"`     // catalog id shown to users
	Repo   string `json:"repo"`   // HF repo: config + tokenizer (+ weights unless GGUF)
	GGUF   string `json:"gguf"`   // optional GGUF file providing the weights
	DType  string `json:"dtype"`  // "f32" or "bf16" for safetensors weights
	Device string `json:"device"` // "gpu" or "cpu"
	Family string `json:"family"` // chat template family (filled on load)
	// TokenizerPath overrides Repo's tokenizer.json (sidecar next to a GGUF).
	TokenizerPath string `json:"tokenizer_path,omitempty"`
}

// Model is a loaded checkpoint plus everything needed to chat with it.
type Model struct {
	Spec     Spec
	LM       models.CausalLM
	Tok      *tokenizer.BPE
	Template Template
	Dev      device.Device
	eos      map[int64]bool
	bos      int64
	hasBOS   bool
	mu       sync.Mutex // one generation at a time
	LoadTime time.Duration
	// Diffusion is set for block-diffusion LMs (model_type "sdar"): Generate
	// then decodes with models.Llama.GenerateDiffusion instead of the
	// autoregressive loop. SDAR holds its decoding settings.
	Diffusion bool
	SDAR      models.SDAROpts
}

// GenOpts controls sampling. Temperature 0 is greedy.
type GenOpts struct {
	MaxTokens   int     `json:"max_tokens"`
	Temperature float32 `json:"temperature"`
	TopP        float32 `json:"top_p"`
	TopK        int     `json:"top_k"`
	Seed        uint64  `json:"seed"`
}

// Stats describes one generation.
type Stats struct {
	PromptTokens int     `json:"prompt_tokens"`
	Tokens       int     `json:"tokens"`
	PrefillMS    float64 `json:"prefill_ms"`
	DecodeMS     float64 `json:"decode_ms"`
	TokPerSec    float64 `json:"tok_per_sec"`
	Stop         string  `json:"stop"` // "eos", "length" or "cancel"
	// Greedy requests on a model with an MTP head (MiMo) decode
	// speculatively: draft tokens proposed by the head, and how many the
	// main model kept (the output is identical to plain greedy).
	Drafted  int `json:"drafted,omitempty"`
	Accepted int `json:"accepted,omitempty"`
}

// ErrBusy is returned when a generation is already running on the model.
var ErrBusy = errors.New("engine: model is busy")

func pickDevice(name string) (device.Device, error) {
	switch strings.ToLower(name) {
	case "", "gpu", "cuda":
		if device.CUDAAvailable() {
			return device.CUDA(0)
		}
		if name == "" {
			return device.CPU(), nil
		}
		return nil, errors.New("engine: CUDA not available in this build")
	case "cpu":
		return device.CPU(), nil
	case "webgpu", "wgpu":
		return device.WebGPU(0) // any GPU through wgpu-native (Intel, AMD, Apple, NVIDIA); F32
	}
	return nil, fmt.Errorf("engine: unknown device %q (gpu, webgpu or cpu)", name)
}

// Load reads the tokenizer and weights for spec. GGUF weights take
// precedence over safetensors when both are given.
func Load(spec Spec) (*Model, error) {
	t0 := time.Now()
	dev, err := pickDevice(spec.Device)
	if err != nil {
		return nil, err
	}
	// "w16" (any device) keeps the matmul weights in BF16 storage under F32
	// compute (W16A32): exact for a BF16 checkpoint, in half the memory of
	// f32. WebGPU computes in F32 only, so "bf16" / "f16" there mean the same
	// (BF16 / F16 storage).
	var store16 dtype.DType
	w16 := false
	if spec.GGUF == "" {
		switch dt := strings.ToLower(spec.DType); {
		case dt == "w16":
			store16, w16 = dtype.BF16, true
		case device.IsWebGPU(dev) && dt == "bf16":
			store16, w16 = dtype.BF16, true
		case device.IsWebGPU(dev) && dt == "f16":
			store16, w16 = dtype.F16, true
		case device.IsWebGPU(dev) && dt != "" && dt != "f32":
			return nil, fmt.Errorf("engine: WebGPU loads dtype f32, bf16 / w16 or f16 (16-bit weight storage), got %q", spec.DType)
		}
	}
	var tok *tokenizer.BPE
	var ggufVocab *tokenizer.GGUFVocab
	tokPath := spec.TokenizerPath
	switch {
	case tokPath == "" && spec.Repo == "" && spec.GGUF != "":
		// No sidecar and no HF repo: the GGUF's own tokenizer.ggml.* metadata.
		tok, ggufVocab, err = tokenizer.LoadGGUF(spec.GGUF)
		if err != nil {
			return nil, fmt.Errorf("engine: GGUF tokenizer: %w", err)
		}
	case tokPath == "" && spec.Repo == "":
		return nil, errors.New("engine: spec needs Repo, TokenizerPath or GGUF for the tokenizer")
	default:
		if tokPath == "" {
			tokPath, err = hub.Download(spec.Repo, "main", "tokenizer.json")
			if err != nil {
				return nil, fmt.Errorf("engine: tokenizer: %w", err)
			}
		}
		f, err := os.Open(tokPath)
		if err != nil {
			return nil, err
		}
		tok, err = tokenizer.LoadBPE(f)
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("engine: tokenizer: %w", err)
		}
	}

	var lm models.CausalLM
	switch {
	case spec.GGUF != "":
		lm, err = models.LoadChatModelGGUF(spec.GGUF, dev)
	default:
		wdt := dtype.F32
		if strings.EqualFold(spec.DType, "bf16") && !w16 {
			wdt = dtype.BF16
		}
		if w16 {
			// The weights are allocated at 16 bits as the loader builds the
			// model, so no F32 copy of them exists; StoreWeights16 then
			// narrows any a loader built another way.
			err = nn.WithWeightStorage16(store16, func() error {
				var err error
				lm, err = models.LoadChatModelFromHF(spec.Repo, dev, dtype.F32)
				return err
			})
			if err == nil {
				_, err = models.StoreWeights16(lm, store16)
			}
		} else {
			lm, err = models.LoadChatModelFromHF(spec.Repo, dev, wdt)
		}
	}
	if err != nil {
		return nil, err
	}

	m := &Model{Spec: spec, LM: lm, Tok: tok, Dev: dev, eos: map[int64]bool{}}
	m.Template = TemplateFor(tok)
	if ct := loadChatTemplate(spec, tok); ct != nil {
		m.Template = withJinja(m.Template, ct)
	}
	m.Spec.Family = m.Template.Family
	if spec.Repo != "" {
		if ids, err := models.EOSTokenIDs(spec.Repo); err == nil {
			for _, id := range ids {
				m.eos[id] = true
			}
		}
		if bos, ok := models.BOSTokenID(spec.Repo); ok {
			m.bos, m.hasBOS = bos, true
		}
	}
	if ggufVocab != nil {
		if ggufVocab.EOS >= 0 {
			m.eos[int64(ggufVocab.EOS)] = true
		}
		if ggufVocab.EOT >= 0 { // end of turn (Muse Glimmer's <|eot|>)
			m.eos[int64(ggufVocab.EOT)] = true
		}
		if ggufVocab.AddBOS && ggufVocab.BOS >= 0 {
			m.bos, m.hasBOS = int64(ggufVocab.BOS), true
		}
	}
	stops := m.Template.Stop
	if m.Template.Jinja != nil {
		if eos := m.Template.Jinja.Special["eos_token"]; eos != "" {
			stops = append(stops[:len(stops):len(stops)], eos)
		}
	}
	for _, s := range stops {
		if id := tok.TokenToID(s); id >= 0 {
			m.eos[int64(id)] = true
		}
	}
	switch {
	case m.Template.Source == "jinja" && spec.GGUF == "":
		// transformers tokenizes the rendered chat with
		// add_special_tokens=False: BOS appears only if the template
		// writes bos_token itself.
		m.hasBOS = false
	case m.Template.Source == "jinja":
		// llama.cpp: add_bos still applies to GGUF models, but a BOS the
		// template already wrote is not doubled (Generate checks ids[0]).
	case m.Template.Family == "llama3" || m.Template.Family == "mistral" || m.Template.Family == "gemma":
		// Built-in templates that write the BOS token themselves must not
		// get a second.
		m.hasBOS = false
	}
	if spec.GGUF == "" && spec.Repo != "" {
		if mt, err := models.HFModelType(spec.Repo); err == nil && mt == "sdar" {
			m.Diffusion = true
			m.SDAR = models.DefaultSDAROpts()
			m.SDAR.EOS = m.SDAR.EOS[:0]
			for id := range m.eos {
				m.SDAR.EOS = append(m.SDAR.EOS, id)
			}
		}
	}
	m.LoadTime = time.Since(t0)
	return m, nil
}

// generateDiffusion is Generate for block-diffusion models: whole blocks
// arrive at once, so streaming happens per finished block.
func (m *Model) generateDiffusion(ctx context.Context, ids []int32, opts GenOpts, onToken func(string), st Stats) (string, Stats, error) {
	llm, ok := m.LM.(*models.Llama)
	if !ok {
		return "", st, errors.New("engine: diffusion decoding needs a models.Llama")
	}
	sd := m.SDAR
	sd.MaxTokens = opts.MaxTokens
	sd.Temperature, sd.TopK, sd.TopP, sd.Seed = opts.Temperature, opts.TopK, opts.TopP, opts.Seed
	if sd.Seed == 0 {
		sd.Seed = rand.Uint64()
	}
	var gen []int32
	var out strings.Builder
	emitted := 0
	flush := func(final bool) {
		text := m.Tok.Decode(gen)
		if !final && strings.HasSuffix(text, "�") {
			text = strings.TrimSuffix(text, "�")
		}
		if len(text) > emitted {
			delta := text[emitted:]
			emitted = len(text)
			out.WriteString(delta)
			if onToken != nil {
				onToken(delta)
			}
		}
	}
	t0 := time.Now()
	st.Stop = "length"
	toks, err := llm.GenerateDiffusion(toI64(ids), sd, func(block []int64) {
		for _, t := range block {
			gen = append(gen, int32(t))
		}
		st.Tokens = len(gen)
		flush(false)
	})
	if err != nil {
		return out.String(), st, err
	}
	if len(toks) < opts.MaxTokens {
		st.Stop = "eos"
	}
	if ctx.Err() != nil {
		st.Stop = "cancel"
	}
	flush(true)
	st.DecodeMS = float64(time.Since(t0).Microseconds()) / 1000
	if st.DecodeMS > 0 {
		st.TokPerSec = float64(st.Tokens) / (st.DecodeMS / 1000)
	}
	return out.String(), st, nil
}

// Busy reports whether a generation is in flight.
func (m *Model) Busy() bool {
	if m.mu.TryLock() {
		m.mu.Unlock()
		return false
	}
	return true
}

// Prompt renders msgs with the model's chat template.
func (m *Model) Prompt(msgs []Message) string { return m.Template.Render(msgs) }

// Generate streams a reply for msgs. onToken receives decoded text deltas
// (never partial UTF-8). It returns the full reply text. Each call uses a
// fresh KV cache, so requests are independent; cancel via ctx.
func (m *Model) Generate(ctx context.Context, msgs []Message, opts GenOpts, onToken func(string)) (string, Stats, error) {
	if !m.mu.TryLock() {
		return "", Stats{}, ErrBusy
	}
	defer m.mu.Unlock()

	if opts.MaxTokens <= 0 {
		opts.MaxTokens = 512
	}
	var rng *rand.Rand
	if opts.Seed != 0 {
		rng = rand.New(rand.NewPCG(opts.Seed, opts.Seed^0x9E3779B97F4A7C15))
	} else {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	sample := models.SampleOpts{Temperature: opts.Temperature, TopK: opts.TopK, TopP: opts.TopP, Rng: rng}

	ids := m.Tok.Encode(m.Prompt(msgs))
	if m.hasBOS && (len(ids) == 0 || int64(ids[0]) != m.bos) {
		ids = append([]int32{int32(m.bos)}, ids...)
	}
	if len(ids) == 0 {
		return "", Stats{}, errors.New("engine: empty prompt")
	}
	maxPos := m.LM.MaxContext()
	if len(ids)+opts.MaxTokens > maxPos {
		if len(ids) >= maxPos {
			return "", Stats{}, fmt.Errorf("engine: prompt has %d tokens, model max is %d", len(ids), maxPos)
		}
		opts.MaxTokens = maxPos - len(ids)
	}

	var st Stats
	st.PromptTokens = len(ids)
	if m.Diffusion {
		return m.generateDiffusion(ctx, ids, opts, onToken, st)
	}
	if l, ok := m.LM.(*models.Llama); ok && l.MTP != nil && opts.Temperature == 0 {
		return m.generateSpec(ctx, l, ids, opts, onToken, st)
	}
	cache := models.NewKVCache(m.LM.NumLayers())
	defer cache.Reset()

	t0 := time.Now()
	row, err := m.step(toI64(ids), cache)
	if err != nil {
		return "", st, fmt.Errorf("prefill: %w", err)
	}
	st.PrefillMS = float64(time.Since(t0).Microseconds()) / 1000

	var gen []int32
	var out strings.Builder
	flush := m.flusher(&gen, &out, onToken)

	t1 := time.Now()
	st.Stop = "length"
	for step := 0; step < opts.MaxTokens; step++ {
		if ctx.Err() != nil {
			st.Stop = "cancel"
			break
		}
		next := models.SampleLogits(row, sample)
		if m.eos[int64(next)] {
			st.Stop = "eos"
			break
		}
		gen = append(gen, int32(next))
		st.Tokens++
		flush(false)
		if step+1 == opts.MaxTokens {
			break
		}
		row, err = m.step([]int64{int64(next)}, cache)
		if err != nil {
			return out.String(), st, fmt.Errorf("decode step %d: %w", step, err)
		}
		if step%8 == 7 {
			runtime.GC() // release device buffers of finished steps
		}
	}
	flush(true)
	st.DecodeMS = float64(time.Since(t1).Microseconds()) / 1000
	if st.DecodeMS > 0 {
		st.TokPerSec = float64(st.Tokens) / (st.DecodeMS / 1000)
	}
	return out.String(), st, nil
}

// flusher returns a function that streams the text of *gen not yet sent
// (to out and onToken), holding back a trailing partial UTF-8 sequence
// unless final.
func (m *Model) flusher(gen *[]int32, out *strings.Builder, onToken func(string)) func(final bool) {
	emitted := 0
	return func(final bool) {
		text := m.Tok.Decode(*gen)
		// Hold back a trailing replacement char: the next token may complete
		// the multi-byte sequence.
		if !final && strings.HasSuffix(text, "�") {
			text = strings.TrimSuffix(text, "�")
		}
		if len(text) > emitted {
			delta := text[emitted:]
			emitted = len(text)
			out.WriteString(delta)
			if onToken != nil {
				onToken(delta)
			}
		}
	}
}

// generateSpec is greedy decoding through models.SpecGreedy (MTP draft
// head): the same tokens as the plain loop, fewer main-model forwards when
// drafts are accepted. Prefill time is measured to the first token.
func (m *Model) generateSpec(ctx context.Context, l *models.Llama, ids []int32, opts GenOpts, onToken func(string), st Stats) (string, Stats, error) {
	var gen []int32
	var out strings.Builder
	flush := m.flusher(&gen, &out, onToken)
	t0 := time.Now()
	var t1 time.Time
	st.Stop = "length"
	ss, err := models.SpecGreedy(l, toI64(ids), opts.MaxTokens, func(tok int64) bool {
		if t1.IsZero() {
			t1 = time.Now()
			st.PrefillMS = float64(t1.Sub(t0).Microseconds()) / 1000
		}
		if ctx.Err() != nil {
			st.Stop = "cancel"
			return false
		}
		if m.eos[tok] {
			st.Stop = "eos"
			return false
		}
		gen = append(gen, int32(tok))
		st.Tokens++
		flush(false)
		if st.Tokens%8 == 0 {
			runtime.GC() // release device buffers of finished steps
		}
		return true
	})
	st.Drafted, st.Accepted = ss.Drafted, ss.Accepted
	flush(true)
	if err != nil {
		return out.String(), st, fmt.Errorf("speculative decode: %w", err)
	}
	if !t1.IsZero() {
		st.DecodeMS = float64(time.Since(t1).Microseconds()) / 1000
	}
	if st.DecodeMS > 0 {
		st.TokPerSec = float64(st.Tokens) / (st.DecodeMS / 1000)
	}
	return out.String(), st, nil
}

// step runs tokens through the model and returns the last-position logits.
func (m *Model) step(tokens []int64, cache *models.KVCache) ([]float32, error) {
	x, err := tensor.New(tokens, []int{1, len(tokens)}, m.Dev)
	if err != nil {
		return nil, err
	}
	logits, err := m.LM.ForwardWithCache(x, cache)
	if err != nil {
		return nil, err
	}
	cpu, err := logits.To(device.CPU())
	if err != nil {
		return nil, err
	}
	flat := tensor.MustSlice[float32](cpu)
	V := m.LM.VocabSize()
	off := (len(tokens) - 1) * V
	return flat[off : off+V], nil
}

func toI64(ids []int32) []int64 {
	out := make([]int64, len(ids))
	for i, v := range ids {
		out[i] = int64(v)
	}
	return out
}

// ggufDir is where the catalog looks for GGUF files by default.
// GGUFDir is where the catalog looks for GGUF files: $LIGHT_GGUF_DIR, or
// ~/.cache/light/gguf.
func GGUFDir() string { return ggufDir() }

func ggufDir() string {
	if v := os.Getenv("LIGHT_GGUF_DIR"); v != "" {
		return v
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "light", "gguf")
}
