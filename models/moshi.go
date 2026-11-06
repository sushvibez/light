package models

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// Moshi (Kyutai, 2024) is the speech-text foundation model behind Moshiko /
// Moshika: a 7B "temporal" transformer that consumes, per 80 ms frame, one
// text token plus the Mimi codes of two audio streams (its own voice and the
// user's), and predicts the next text token; a small "depth" transformer
// (depformer) then predicts the 8 Mimi codebooks of its own next frame one
// after another, each slice conditioned on the previous code.
//
//	emb   = text_emb(text) + Σ_k audio_emb_k(code_k)           (16 codebooks)
//	ys    = out_norm(transformer(emb))                         (32 layers, RoPE,
//	                                                            sliding context)
//	text  ~ text_linear(ys)
//	code_s ~ linear_out_s(depformer_s(linear_in_s(ys) + emb_s(prev)))  s = 0..7
//
// Transformer blocks are pre-norm RMSNorm (eps 1e-8) with a fused QKV
// projection, interleaved-pair RoPE applied in F32, a SiLU-gated MLP whose
// projection is split in two halves (activated half × linear half), and no
// biases. The depformer slices have their own weights but share one KV
// cache across the 8 positions of a frame (no positional embedding).
//
// Reference: kyutai-labs/moshi rust/moshi-core/src/{lm,transformer}.rs and
// moshi/models/lm.py. Weights: kyutai/moshiko-candle-q8 model.q8.gguf
// (Q8_0 linears and embeddings, F32 norms) via LoadMoshiGGUF.
type Moshi struct {
	Cfg         MoshiConfig
	Transformer *moshiTransformer
	TextEmb     *nn.Embedding
	AudioEmb    []*nn.Embedding
	OutNorm     *nn.RMSNorm
	TextLinear  *nn.Linear
	Depformer   []*MoshiDepSlice
}

// MoshiDepSlice is one depformer step: its own 6-layer transformer, the
// embedding of the previous token (text for slice 0, the previous codebook
// otherwise), the projection from the main model width and the code head.
type MoshiDepSlice struct {
	Transformer *moshiTransformer
	Emb         *nn.Embedding
	LinearIn    *nn.Linear
	LinearOut   *nn.Linear
}

// MoshiTransformerConfig describes one of Moshi's transformers.
type MoshiTransformerConfig struct {
	Dim       int
	NumHeads  int
	NumLayers int
	Hidden    int     // gated-MLP hidden width (linear_in is Dim → 2·Hidden)
	Context   int     // keys visible to a query, including itself
	RoPE      bool    // interleaved RoPE (false: no positional embedding)
	MaxPeriod float64 // RoPE base
}

func (c MoshiTransformerConfig) headDim() int { return c.Dim / c.NumHeads }

// MoshiConfig is the full model: temporal transformer, depformer, vocabs.
type MoshiConfig struct {
	Transformer    MoshiTransformerConfig
	Depformer      MoshiTransformerConfig
	TextInVocab    int // 32001: text vocab + start token
	TextOutVocab   int // 32000
	AudioVocab     int // 2049: codebook entries + pad token
	AudioCodebooks int // input codebooks: own stream + user stream (8 + 8)
	DepQ           int // generated codebooks per frame (depformer slices)
}

// DefaultMoshiConfig is Moshi v0.1 (moshiko / moshika 7B).
func DefaultMoshiConfig() MoshiConfig {
	return MoshiConfig{
		Transformer: MoshiTransformerConfig{Dim: 4096, NumHeads: 32, NumLayers: 32, Hidden: 11264, Context: 3000, RoPE: true, MaxPeriod: 10000},
		Depformer:   MoshiTransformerConfig{Dim: 1024, NumHeads: 16, NumLayers: 6, Hidden: 2816, Context: 8, RoPE: false, MaxPeriod: 10000},
		TextInVocab: 32001, TextOutVocab: 32000, AudioVocab: 2049, AudioCodebooks: 16, DepQ: 8,
	}
}

// AudioPadToken is the code fed for not-yet-available audio positions.
func (c MoshiConfig) AudioPadToken() int64 { return int64(c.AudioVocab - 1) }

// TextStartToken is the text token fed before any text was generated.
func (c MoshiConfig) TextStartToken() int64 { return int64(c.TextInVocab - 1) }

type moshiLayer struct {
	Norm1, Norm2    *nn.RMSNorm
	InProj, OutProj *nn.Linear // fused (Dim → 3·Dim) and (Dim → Dim)
	GateIn, GateOut *nn.Linear // Dim → 2·Hidden, Hidden → Dim
}

type moshiTransformer struct {
	Cfg      MoshiTransformerConfig
	Layers   []*moshiLayer
	scale    *tensor.Tensor
	masks    map[[3]int]*tensor.Tensor
	masksDev device.Device
}

const moshiRMSEps = 1e-8

func newMoshiTransformer(cfg MoshiTransformerConfig, dev device.Device) (*moshiTransformer, error) {
	if cfg.Dim%cfg.NumHeads != 0 || cfg.headDim()%2 != 0 {
		return nil, fmt.Errorf("moshi: dim %d / heads %d must give an even head dim", cfg.Dim, cfg.NumHeads)
	}
	t := &moshiTransformer{Cfg: cfg, Layers: make([]*moshiLayer, cfg.NumLayers), masks: map[[3]int]*tensor.Tensor{}}
	for i := range t.Layers {
		l := &moshiLayer{}
		var err error
		if l.Norm1, err = nn.NewRMSNormWithDType(cfg.Dim, moshiRMSEps, dtype.F32, dev); err != nil {
			return nil, err
		}
		if l.Norm2, err = nn.NewRMSNormWithDType(cfg.Dim, moshiRMSEps, dtype.F32, dev); err != nil {
			return nil, err
		}
		if l.InProj, err = nn.NewLinearPlaceholder(cfg.Dim, 3*cfg.Dim, false, dev); err != nil {
			return nil, err
		}
		if l.OutProj, err = nn.NewLinearPlaceholder(cfg.Dim, cfg.Dim, false, dev); err != nil {
			return nil, err
		}
		if l.GateIn, err = nn.NewLinearPlaceholder(cfg.Dim, 2*cfg.Hidden, false, dev); err != nil {
			return nil, err
		}
		if l.GateOut, err = nn.NewLinearPlaceholder(cfg.Hidden, cfg.Dim, false, dev); err != nil {
			return nil, err
		}
		t.Layers[i] = l
	}
	var err error
	if t.scale, err = tensor.New([]float32{float32(1 / math.Sqrt(float64(cfg.headDim())))}, []int{1}, dev); err != nil {
		return nil, err
	}
	return t, nil
}

// NewMoshi allocates a model whose linear weights are placeholders to be
// installed by a loader (or filled by a test).
func NewMoshi(cfg MoshiConfig, dev device.Device) (*Moshi, error) {
	m := &Moshi{Cfg: cfg}
	var err error
	if m.Transformer, err = newMoshiTransformer(cfg.Transformer, dev); err != nil {
		return nil, err
	}
	D := cfg.Transformer.Dim
	if m.TextEmb, err = nn.NewEmbeddingUninit(cfg.TextInVocab, D, dtype.F32, dev); err != nil {
		return nil, err
	}
	m.AudioEmb = make([]*nn.Embedding, cfg.AudioCodebooks)
	for k := range m.AudioEmb {
		if m.AudioEmb[k], err = nn.NewEmbeddingUninit(cfg.AudioVocab, D, dtype.F32, dev); err != nil {
			return nil, err
		}
	}
	if m.OutNorm, err = nn.NewRMSNormWithDType(D, moshiRMSEps, dtype.F32, dev); err != nil {
		return nil, err
	}
	if m.TextLinear, err = nn.NewLinearPlaceholder(D, cfg.TextOutVocab, false, dev); err != nil {
		return nil, err
	}
	m.Depformer = make([]*MoshiDepSlice, cfg.DepQ)
	for s := range m.Depformer {
		sl := &MoshiDepSlice{}
		if sl.Transformer, err = newMoshiTransformer(cfg.Depformer, dev); err != nil {
			return nil, err
		}
		inVocab := cfg.AudioVocab
		if s == 0 {
			inVocab = cfg.TextInVocab
		}
		if sl.Emb, err = nn.NewEmbeddingUninit(inVocab, cfg.Depformer.Dim, dtype.F32, dev); err != nil {
			return nil, err
		}
		if sl.LinearIn, err = nn.NewLinearPlaceholder(D, cfg.Depformer.Dim, false, dev); err != nil {
			return nil, err
		}
		// The depformer never emits the audio pad token.
		if sl.LinearOut, err = nn.NewLinearPlaceholder(cfg.Depformer.Dim, cfg.AudioVocab-1, false, dev); err != nil {
			return nil, err
		}
		m.Depformer[s] = sl
	}
	return m, nil
}

func (t *moshiTransformer) linears() []*nn.Linear {
	var ls []*nn.Linear
	for _, l := range t.Layers {
		ls = append(ls, l.InProj, l.OutProj, l.GateIn, l.GateOut)
	}
	return ls
}

func (t *moshiTransformer) norms() []*nn.RMSNorm {
	var ns []*nn.RMSNorm
	for _, l := range t.Layers {
		ns = append(ns, l.Norm1, l.Norm2)
	}
	return ns
}

// Parameters lists every weight tensor (nil for placeholders not yet set).
func (m *Moshi) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	add := func(t *moshiTransformer) {
		for _, l := range t.linears() {
			ps = append(ps, l.W)
		}
		for _, n := range t.norms() {
			ps = append(ps, n.Weight)
		}
	}
	add(m.Transformer)
	ps = append(ps, m.TextEmb.Weight)
	for _, e := range m.AudioEmb {
		ps = append(ps, e.Weight)
	}
	ps = append(ps, m.OutNorm.Weight, m.TextLinear.W)
	for _, s := range m.Depformer {
		add(s.Transformer)
		ps = append(ps, s.Emb.Weight, s.LinearIn.W, s.LinearOut.W)
	}
	return ps
}

// ParamCount counts elements over the installed weights.
func (m *Moshi) ParamCount() int {
	n := 0
	for _, p := range m.Parameters() {
		if p != nil {
			n += p.NumElements()
		}
	}
	return n
}

// mask returns the (T, kvPrior+T) additive mask: key j is hidden from query
// i (absolute position kvPrior+i) when it lies in the future or more than
// window-1 steps back.
func (t *moshiTransformer) mask(T, kvPrior, window int, dev device.Device) (*tensor.Tensor, error) {
	if t.masksDev == nil || !t.masksDev.SameAs(dev) {
		t.masks = map[[3]int]*tensor.Tensor{}
		t.masksDev = dev
	}
	key := [3]int{T, kvPrior, window}
	if m, ok := t.masks[key]; ok {
		return m, nil
	}
	total := kvPrior + T
	negInf := float32(math.Inf(-1))
	data := make([]float32, T*total)
	for i := range T {
		pos := kvPrior + i
		for j := range total {
			if j > pos || j <= pos-window {
				data[i*total+j] = negInf
			}
		}
	}
	m, err := tensor.New(data, []int{T, total}, dev)
	if err != nil {
		return nil, err
	}
	t.masks[key] = m
	return m, nil
}

// forward runs the layer stack over x (B, T, Dim) F32, appending to cache
// (positions continue from cache.Len) and keeping the last Context keys.
func (t *moshiTransformer) forward(x *tensor.Tensor, cache *KVCache) (*tensor.Tensor, error) {
	cfg := t.Cfg
	B, T, D := x.Dim(0), x.Dim(1), x.Dim(2)
	NH, HD := cfg.NumHeads, cfg.headDim()
	dev := x.Device()
	prior := 0
	if cache != nil {
		prior = cache.Len
	}
	var pos *tensor.Tensor
	if cfg.RoPE {
		data := make([]int64, T)
		for i := range data {
			data[i] = int64(prior + i)
		}
		var err error
		if pos, err = tensor.New(data, []int{T}, dev); err != nil {
			return nil, err
		}
	}
	// part extracts slot i of a (B, T, n, NH, HD) projection as (B, NH, T, HD).
	part := func(p *tensor.Tensor, i int) (*tensor.Tensor, error) {
		s, err := p.Narrow(2, i, 1)
		if err != nil {
			return nil, err
		}
		if s, err = autograd.Contiguous(s); err != nil {
			return nil, err
		}
		if s, err = autograd.Reshape(s, B, T, NH, HD); err != nil {
			return nil, err
		}
		return headsForward(s, B, T, NH, HD)
	}
	for li, l := range t.Layers {
		h, err := l.Norm1.Forward(x)
		if err != nil {
			return nil, err
		}
		qkv, err := l.InProj.Forward(h)
		if err != nil {
			return nil, err
		}
		if qkv, err = autograd.Reshape(qkv, B, T, 3, NH, HD); err != nil {
			return nil, err
		}
		q, err := part(qkv, 0)
		if err != nil {
			return nil, err
		}
		k, err := part(qkv, 1)
		if err != nil {
			return nil, err
		}
		v, err := part(qkv, 2)
		if err != nil {
			return nil, err
		}
		if cfg.RoPE {
			if q, err = ops.RoPEInterleaved(q, pos, cfg.MaxPeriod); err != nil {
				return nil, err
			}
			if k, err = ops.RoPEInterleaved(k, pos, cfg.MaxPeriod); err != nil {
				return nil, err
			}
		}
		kvPrior := 0
		if cache != nil {
			if cache.K[li] != nil {
				kvPrior = cache.K[li].Dim(2)
				if k, err = ops.Concat([]*tensor.Tensor{cache.K[li], k}, 2); err != nil {
					return nil, err
				}
				if v, err = ops.Concat([]*tensor.Tensor{cache.V[li], v}, 2); err != nil {
					return nil, err
				}
			}
			cache.K[li], cache.V[li] = k, v
			if total := kvPrior + T; total > cfg.Context {
				if cache.K[li], err = trimKV(k, total-cfg.Context, cfg.Context); err != nil {
					return nil, err
				}
				if cache.V[li], err = trimKV(v, total-cfg.Context, cfg.Context); err != nil {
					return nil, err
				}
			}
		}
		total := kvPrior + T
		if q, err = autograd.Reshape(q, B*NH, T, HD); err != nil {
			return nil, err
		}
		if k, err = autograd.Reshape(k, B*NH, total, HD); err != nil {
			return nil, err
		}
		if v, err = autograd.Reshape(v, B*NH, total, HD); err != nil {
			return nil, err
		}
		kT, err := autograd.Transpose(k, 1, 2)
		if err != nil {
			return nil, err
		}
		if kT, err = autograd.Contiguous(kT); err != nil {
			return nil, err
		}
		scores, err := autograd.MatMul(q, kT)
		if err != nil {
			return nil, err
		}
		if scores, err = ops.Mul(scores, t.scale); err != nil {
			return nil, err
		}
		if T > 1 || total > cfg.Context {
			mask, err := t.mask(T, kvPrior, cfg.Context, dev)
			if err != nil {
				return nil, err
			}
			if scores, err = ops.Add(scores, mask); err != nil {
				return nil, err
			}
		}
		attn, err := ops.Softmax(scores, -1)
		if err != nil {
			return nil, err
		}
		ctx, err := autograd.MatMul(attn, v)
		if err != nil {
			return nil, err
		}
		if ctx, err = autograd.Reshape(ctx, B, NH, T, HD); err != nil {
			return nil, err
		}
		if ctx, err = autograd.Permute(ctx, 0, 2, 1, 3); err != nil {
			return nil, err
		}
		if ctx, err = autograd.Contiguous(ctx); err != nil {
			return nil, err
		}
		if ctx, err = autograd.Reshape(ctx, B, T, D); err != nil {
			return nil, err
		}
		o, err := l.OutProj.Forward(ctx)
		if err != nil {
			return nil, err
		}
		if x, err = ops.Add(x, o); err != nil {
			return nil, err
		}
		// Gated MLP: linear_in → (…, 2, Hidden); silu(first half) × second half.
		if h, err = l.Norm2.Forward(x); err != nil {
			return nil, err
		}
		g, err := l.GateIn.Forward(h)
		if err != nil {
			return nil, err
		}
		if g, err = autograd.Reshape(g, B, T, 2, cfg.Hidden); err != nil {
			return nil, err
		}
		a, err := g.Narrow(2, 0, 1)
		if err != nil {
			return nil, err
		}
		if a, err = autograd.Contiguous(a); err != nil {
			return nil, err
		}
		if a, err = autograd.Reshape(a, B, T, cfg.Hidden); err != nil {
			return nil, err
		}
		b, err := g.Narrow(2, 1, 1)
		if err != nil {
			return nil, err
		}
		if b, err = autograd.Contiguous(b); err != nil {
			return nil, err
		}
		if b, err = autograd.Reshape(b, B, T, cfg.Hidden); err != nil {
			return nil, err
		}
		if a, err = ops.SiLU(a); err != nil {
			return nil, err
		}
		if a, err = ops.Mul(a, b); err != nil {
			return nil, err
		}
		if a, err = l.GateOut.Forward(a); err != nil {
			return nil, err
		}
		if x, err = ops.Add(x, a); err != nil {
			return nil, fmt.Errorf("moshi layer %d: %w", li, err)
		}
	}
	if cache != nil {
		cache.Len += T
	}
	return x, nil
}

// Forward runs the temporal transformer over T frames. text is (B, T) I64;
// audio holds one (B, T) I64 tensor per input codebook (nil entries are
// skipped, as the reference does for absent streams). Returns the text
// logits (B, T, TextOutVocab) and the normalised hidden states (B, T, Dim)
// the depformer conditions on.
func (m *Moshi) Forward(text *tensor.Tensor, audio []*tensor.Tensor, cache *KVCache) (logits, hidden *tensor.Tensor, err error) {
	if len(audio) > m.Cfg.AudioCodebooks {
		return nil, nil, fmt.Errorf("moshi: %d audio codebooks given, model takes %d", len(audio), m.Cfg.AudioCodebooks)
	}
	emb, err := m.TextEmb.Forward(text)
	if err != nil {
		return nil, nil, err
	}
	for k, codes := range audio {
		if codes == nil {
			continue
		}
		e, err := m.AudioEmb[k].Forward(codes)
		if err != nil {
			return nil, nil, err
		}
		if emb, err = ops.Add(emb, e); err != nil {
			return nil, nil, err
		}
	}
	ys, err := m.Transformer.forward(emb, cache)
	if err != nil {
		return nil, nil, err
	}
	if ys, err = m.OutNorm.Forward(ys); err != nil {
		return nil, nil, err
	}
	logits, err = m.TextLinear.Forward(ys)
	if err != nil {
		return nil, nil, err
	}
	return logits, ys, nil
}

// NewDepformerCache allocates the per-frame depformer KV cache.
func (m *Moshi) NewDepformerCache() *KVCache { return NewKVCache(m.Cfg.Depformer.NumLayers) }

// DepformerStep generates the DepQ codes of one frame from hidden (1, 1,
// Dim): slice s is fed linear_in_s(hidden) plus the embedding of the token
// chosen at slice s-1 (the text token for s = 0), attends over the slices
// before it through the shared cache (reset here), and its logits go to
// sample. forced, when non-nil, overrides the token passed on to the next
// slice (forced[s] >= 0) without changing what is returned, as the
// reference does during the acoustic delay. The per-slice logit rows are
// returned alongside the codes.
func (m *Moshi) DepformerStep(hidden *tensor.Tensor, textToken int64, forced []int64, sample func([]float32) int, cache *KVCache) ([]int64, [][]float32, error) {
	if hidden.Dim(0) != 1 || hidden.Dim(1) != 1 {
		return nil, nil, fmt.Errorf("moshi: DepformerStep wants hidden (1, 1, D), got %v", hidden.Shape())
	}
	cache.Reset()
	dev := hidden.Device()
	prev := textToken
	codes := make([]int64, 0, m.Cfg.DepQ)
	rows := make([][]float32, 0, m.Cfg.DepQ)
	for s, sl := range m.Depformer {
		x, err := sl.LinearIn.Forward(hidden)
		if err != nil {
			return nil, nil, err
		}
		tok, err := tensor.New([]int64{prev}, []int{1, 1}, dev)
		if err != nil {
			return nil, nil, err
		}
		e, err := sl.Emb.Forward(tok)
		if err != nil {
			return nil, nil, err
		}
		if x, err = ops.Add(x, e); err != nil {
			return nil, nil, err
		}
		if x, err = sl.Transformer.forward(x, cache); err != nil {
			return nil, nil, err
		}
		logits, err := sl.LinearOut.Forward(x)
		if err != nil {
			return nil, nil, err
		}
		row, err := rowF32(logits)
		if err != nil {
			return nil, nil, err
		}
		code := int64(sample(row))
		codes = append(codes, code)
		rows = append(rows, row)
		prev = code
		if s < len(forced) && forced[s] >= 0 {
			prev = forced[s]
		}
	}
	return codes, rows, nil
}

// rowF32 copies a (1, 1, V) tensor to a host slice.
func rowF32(t *tensor.Tensor) ([]float32, error) {
	c, err := t.Contiguous()
	if err != nil {
		return nil, err
	}
	if c, err = c.To(device.CPU()); err != nil {
		return nil, err
	}
	return append([]float32(nil), tensor.MustSlice[float32](c)...), nil
}

// Argmax is the greedy sampler.
func Argmax(row []float32) int {
	best := 0
	for i, v := range row {
		if v > row[best] {
			best = i
		}
	}
	return best
}

// TopKSampler samples from the temperature-scaled softmax restricted to the
// k highest logits (Moshi's defaults: k 250 / temp 0.8 for audio, 25 / 0.7
// for text). k <= 0 means the full distribution; temp <= 0 means argmax.
func TopKSampler(k int, temp float64, rng *rand.Rand) func([]float32) int {
	return func(row []float32) int {
		if temp <= 0 {
			return Argmax(row)
		}
		idx := make([]int, len(row))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return row[idx[a]] > row[idx[b]] })
		if k > 0 && k < len(idx) {
			idx = idx[:k]
		}
		maxv := float64(row[idx[0]])
		probs := make([]float64, len(idx))
		sum := 0.0
		for i, id := range idx {
			probs[i] = math.Exp((float64(row[id]) - maxv) / temp)
			sum += probs[i]
		}
		r := rng.Float64() * sum
		for i, p := range probs {
			r -= p
			if r <= 0 {
				return idx[i]
			}
		}
		return idx[len(idx)-1]
	}
}

// MoshiGen runs Moshi's full-duplex streaming protocol one 80 ms frame at a
// time (kyutai's lm_generate_multistream): the user's 8 Mimi codes go in
// with a one-frame lag, the model's own semantic code goes back in
// immediately and its acoustic codes after AcousticDelay extra frames
// (during which the depformer is forced to pad tokens), and the text token
// produced at one step is fed back at the next.
type MoshiGen struct {
	Model        *Moshi
	Cache        *KVCache
	DepCache     *KVCache
	TextSampler  func([]float32) int
	AudioSampler func([]float32) int
	// AcousticDelay is the number of frames the acoustic codes lag the
	// semantic one (2 for Moshi v0.1).
	AcousticDelay int

	audio      [][]int64 // per step: AudioCodebooks codes (own stream first)
	text       []int64
	step       int
	TextLogits []float32   // text logits of the last Step
	LastCodes  []int64     // depformer codes sampled at the last Step
	LastRows   [][]float32 // depformer logit rows of the last Step, per slice
}

const moshiUngenerated = int64(-1)

// NewMoshiGen starts a conversation. Nil samplers mean argmax.
func NewMoshiGen(m *Moshi, textSampler, audioSampler func([]float32) int) *MoshiGen {
	if textSampler == nil {
		textSampler = Argmax
	}
	if audioSampler == nil {
		audioSampler = Argmax
	}
	return &MoshiGen{Model: m, Cache: NewKVCache(m.Cfg.Transformer.NumLayers), DepCache: m.NewDepformerCache(),
		TextSampler: textSampler, AudioSampler: audioSampler, AcousticDelay: 2}
}

// Step feeds one frame: textIn is the previous text token (TextStartToken
// first) and userCodes the user's Mimi codes for this frame (len
// AudioCodebooks-DepQ; nil for silence-as-pad). It returns the sampled
// text token; LastAudioTokens then gives the frame ready for Mimi.
func (g *MoshiGen) Step(textIn int64, userCodes []int64) (int64, error) {
	m, cfg := g.Model, g.Model.Cfg
	dev := m.OutNorm.Weight.Device()
	total, gen := cfg.AudioCodebooks, cfg.DepQ
	pad := cfg.AudioPadToken()
	for len(g.audio) <= g.step+g.AcousticDelay+1 {
		row := make([]int64, total)
		for i := range row {
			row[i] = moshiUngenerated
		}
		g.audio = append(g.audio, row)
	}
	if userCodes == nil {
		userCodes = make([]int64, total-gen)
		for i := range userCodes {
			userCodes[i] = pad
		}
	}
	if len(userCodes) != total-gen {
		return 0, fmt.Errorf("moshi: %d user codes, want %d", len(userCodes), total-gen)
	}
	copy(g.audio[g.step][gen:], userCodes)
	audio := make([]*tensor.Tensor, total)
	for cb := range total {
		var t int64
		switch {
		case cb == 0 || cb == gen: // semantic codebooks: previous frame
			if g.step == 0 {
				t = pad
			} else {
				t = g.audio[g.step-1][cb]
			}
		case g.step <= g.AcousticDelay:
			t = pad
		default:
			t = g.audio[g.step-g.AcousticDelay-1][cb]
		}
		if t == moshiUngenerated {
			return 0, fmt.Errorf("moshi: step %d codebook %d not generated yet", g.step, cb)
		}
		var err error
		if audio[cb], err = tensor.New([]int64{t}, []int{1, 1}, dev); err != nil {
			return 0, err
		}
	}
	text, err := tensor.New([]int64{textIn}, []int{1, 1}, dev)
	if err != nil {
		return 0, err
	}
	logits, ys, err := m.Forward(text, audio, g.Cache)
	if err != nil {
		return 0, err
	}
	if g.TextLogits, err = rowF32(logits); err != nil {
		return 0, err
	}
	textTok := int64(g.TextSampler(g.TextLogits))
	g.text = append(g.text, textTok)
	var forced []int64
	if g.step < g.AcousticDelay {
		forced = make([]int64, gen)
		for s := range forced {
			forced[s] = pad
		}
		forced[0] = -1
	}
	codes, rows, err := m.DepformerStep(ys, textTok, forced, g.AudioSampler, g.DepCache)
	if err != nil {
		return 0, err
	}
	g.LastCodes, g.LastRows = codes, rows
	for c := range gen {
		delay := g.AcousticDelay
		if c == 0 {
			delay = 0
		}
		g.audio[max(g.step-delay, 0)][c] = codes[c]
	}
	g.step++
	return textTok, nil
}

// Steps is the number of frames processed so far.
func (g *MoshiGen) Steps() int { return g.step }

// TextTokens returns every text token sampled so far.
func (g *MoshiGen) TextTokens() []int64 { return g.text }

// LastAudioTokens returns the model's own DepQ codes for the frame that
// became complete at the last Step, or nil while the acoustic delay has
// not elapsed or the frame still holds pad tokens.
func (g *MoshiGen) LastAudioTokens() []int64 {
	if g.step <= g.AcousticDelay {
		return nil
	}
	row := g.audio[g.step-g.AcousticDelay-1][:g.Model.Cfg.DepQ]
	for _, v := range row {
		if v < 0 || v >= g.Model.Cfg.AudioPadToken() {
			return nil
		}
	}
	return append([]int64(nil), row...)
}
