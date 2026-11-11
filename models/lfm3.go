package models

import (
	"fmt"
	"math"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// LFM3Config is `model_type: lfm3` (Liquid AI's LFM3): a hybrid stack of
// gated short-convolution layers and GQA attention layers, full or sliding.
type LFM3Config struct {
	VocabSize, HiddenSize, IntermediateSize, NumLayers int
	// LayerTypes per layer: "conv", "full_attention" or "sliding_attention".
	LayerTypes []string

	NumHeads, NumKVHeads, HeadDim int
	SlidingWindow                 int // keys a sliding layer's query sees, itself included
	// RoPE per attention type. Partial rotation uses LFM3's zero-tail layout:
	// the first PartialRotary·HeadDim/2 frequencies of the full-head-dim
	// progression, the rest zero (those dims pass through), with rotate_half
	// over the whole head.
	FullRopeBase, FullPartialRotary       float64
	SlidingRopeBase, SlidingPartialRotary float64

	ConvKernel   int
	SandwichNorm bool // RMSNorm on each mixer and MLP output before the residual add
	NormEps      float64
	MaxPosition  int
	TieEmbed     bool
	WeightDType  dtype.DType // F32 or BF16; activations run at the same dtype

	lazyWeights bool
}

func (c LFM3Config) activationDType() dtype.DType {
	if c.WeightDType == dtype.BF16 {
		return dtype.BF16
	}
	return dtype.F32
}

type lfm3Block struct {
	kind string

	OpNorm, FFNNorm, PostOpNorm, PostFFNNorm *nn.RMSNorm

	// Short conv.
	InProj, OutProj *nn.Linear
	ConvW           *tensor.Tensor // (D, K) F32

	// Attention.
	QProj, KProj, VProj, OProj, GateProj *nn.Linear
	QNorm, KNorm                         *nn.RMSNorm

	W1, W3, W2 *nn.Linear
}

// LFM3 is the LFM3 causal LM. Its KVCache holds K/V for attention layers and
// the last K−1 conv inputs for conv layers.
type LFM3 struct {
	Cfg       LFM3Config
	TokenEmb  *nn.Embedding
	Blocks    []*lfm3Block
	FinalNorm *nn.RMSNorm // embedding_norm
	LMHead    *nn.Linear

	attnScale *tensor.Tensor
	masks     map[[3]int]*tensor.Tensor
	masksDev  device.Device
}

// NewLFM3 allocates an uninitialised model; LoadLFM3FromHF fills it.
func NewLFM3(cfg LFM3Config, dev device.Device) (*LFM3, error) {
	if len(cfg.LayerTypes) != cfg.NumLayers {
		return nil, fmt.Errorf("lfm3: %d layer types for %d layers", len(cfg.LayerTypes), cfg.NumLayers)
	}
	if cfg.NumKVHeads <= 0 || cfg.NumHeads%cfg.NumKVHeads != 0 {
		return nil, fmt.Errorf("lfm3: NumHeads %d not divisible by NumKVHeads %d", cfg.NumHeads, cfg.NumKVHeads)
	}
	if cfg.WeightDType != dtype.BF16 {
		cfg.WeightDType = dtype.F32
	}
	act := cfg.activationDType()
	D, NH, KH, HD, IM := cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.IntermediateSize
	var emb *nn.Embedding
	var err error
	if !cfg.lazyWeights {
		if emb, err = nn.NewEmbeddingUninit(cfg.VocabSize, D, act, dev); err != nil {
			return nil, err
		}
	}
	norm := func(n int) (*nn.RMSNorm, error) { return nn.NewRMSNormWithDType(n, cfg.NormEps, act, dev) }
	lin := func(in, out int) (*nn.Linear, error) {
		if cfg.lazyWeights {
			return nn.NewLinearPlaceholder(in, out, false, dev)
		}
		return nn.NewLinearUninit(in, out, false, act, dev)
	}
	blocks := make([]*lfm3Block, cfg.NumLayers)
	for i := range blocks {
		b := &lfm3Block{kind: cfg.LayerTypes[i]}
		var errs []error
		push := func(err error) {
			if err != nil {
				errs = append(errs, err)
			}
		}
		b.OpNorm, err = norm(D)
		push(err)
		b.FFNNorm, err = norm(D)
		push(err)
		if cfg.SandwichNorm {
			b.PostOpNorm, err = norm(D)
			push(err)
			b.PostFFNNorm, err = norm(D)
			push(err)
		}
		switch b.kind {
		case "conv":
			b.InProj, err = lin(D, 3*D)
			push(err)
			b.OutProj, err = lin(D, D)
			push(err)
			b.ConvW, err = tensor.Zeros(dtype.F32, []int{D, cfg.ConvKernel}, dev)
			push(err)
		case "full_attention", "sliding_attention":
			b.QProj, err = lin(D, NH*HD)
			push(err)
			b.KProj, err = lin(D, KH*HD)
			push(err)
			b.VProj, err = lin(D, KH*HD)
			push(err)
			b.OProj, err = lin(NH*HD, D)
			push(err)
			b.GateProj, err = lin(D, NH)
			push(err)
			b.QNorm, err = norm(HD)
			push(err)
			b.KNorm, err = norm(HD)
			push(err)
		default:
			return nil, fmt.Errorf("lfm3 layer %d: unknown layer type %q", i, b.kind)
		}
		b.W1, err = lin(D, IM)
		push(err)
		b.W3, err = lin(D, IM)
		push(err)
		b.W2, err = lin(IM, D)
		push(err)
		if len(errs) > 0 {
			return nil, fmt.Errorf("lfm3 layer %d: %w", i, errs[0])
		}
		blocks[i] = b
	}
	fn, err := norm(D)
	if err != nil {
		return nil, err
	}
	var head *nn.Linear
	if cfg.TieEmbed && !cfg.lazyWeights {
		head, err = nn.NewLinearTied(emb.Weight, false, dev)
	} else {
		head, err = lin(D, cfg.VocabSize)
	}
	if err != nil {
		return nil, err
	}
	scale, err := tensor.New([]float32{float32(1 / math.Sqrt(float64(HD)))}, []int{1}, dev)
	if err != nil {
		return nil, err
	}
	if act != dtype.F32 {
		if scale, err = ops.Cast(scale, act); err != nil {
			return nil, err
		}
	}
	return &LFM3{Cfg: cfg, TokenEmb: emb, Blocks: blocks, FinalNorm: fn, LMHead: head, attnScale: scale}, nil
}

func (m *LFM3) NumLayers() int  { return m.Cfg.NumLayers }
func (m *LFM3) VocabSize() int  { return m.Cfg.VocabSize }
func (m *LFM3) MaxContext() int { return m.Cfg.MaxPosition }

// SlidingWindow is 0: the full-attention layers need the whole cache.
func (m *LFM3) SlidingWindow() int { return 0 }

// Parameters lists every weight tensor (the tied head counted once).
func (m *LFM3) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokenEmb.Weight, m.FinalNorm.Weight}
	if !m.Cfg.TieEmbed {
		ps = append(ps, m.LMHead.W)
	}
	for _, b := range m.Blocks {
		ps = append(ps, b.OpNorm.Weight, b.FFNNorm.Weight, b.W1.W, b.W3.W, b.W2.W)
		if b.PostOpNorm != nil {
			ps = append(ps, b.PostOpNorm.Weight, b.PostFFNNorm.Weight)
		}
		if b.kind == "conv" {
			ps = append(ps, b.InProj.W, b.OutProj.W, b.ConvW)
		} else {
			ps = append(ps, b.QProj.W, b.KProj.W, b.VProj.W, b.OProj.W, b.GateProj.W, b.QNorm.Weight, b.KNorm.Weight)
		}
	}
	return ps
}

func (m *LFM3) ParamCount() int {
	n := 0
	for _, p := range m.Parameters() {
		n += p.Shape().NumElements()
	}
	return n
}

// ForwardWithCache runs tokens (B, T) I64 and returns F32 logits (B, T, V).
func (m *LFM3) ForwardWithCache(tokens *tensor.Tensor, cache *KVCache) (*tensor.Tensor, error) {
	if tokens.Rank() != 2 || tokens.DType() != dtype.I64 {
		return nil, fmt.Errorf("LFM3.Forward: tokens must be (B, T) I64, got %v %v", tokens.Shape(), tokens.DType())
	}
	B, T := tokens.Dim(0), tokens.Dim(1)
	dev := tokens.Device()
	prior := 0
	if cache != nil {
		if len(cache.K) != m.Cfg.NumLayers || len(cache.Conv) != m.Cfg.NumLayers {
			return nil, fmt.Errorf("LFM3.ForwardWithCache: cache has %d layers, model %d", len(cache.K), m.Cfg.NumLayers)
		}
		prior = cache.Len
	}
	if m.Cfg.MaxPosition > 0 && prior+T > m.Cfg.MaxPosition {
		return nil, fmt.Errorf("LFM3.Forward: sequence length %d > MaxPosition %d", prior+T, m.Cfg.MaxPosition)
	}
	h, err := m.TokenEmb.Forward(tokens)
	if err != nil {
		return nil, err
	}
	ropes := map[string][2]*tensor.Tensor{}
	for _, kind := range []string{"full_attention", "sliding_attention"} {
		cos, sin, err := m.ropeTables(kind, prior, T, h.DType(), dev)
		if err != nil {
			return nil, err
		}
		ropes[kind] = [2]*tensor.Tensor{cos, sin}
	}
	for i, blk := range m.Blocks {
		if h, err = m.runBlock(blk, h, B, T, prior, ropes[blk.kind], dev, cache, i); err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
		prefillGC(dev, T, i)
	}
	if h, err = m.FinalNorm.Forward(h); err != nil {
		return nil, err
	}
	if cache != nil {
		cache.Len = prior + T
	}
	logits, err := m.LMHead.Forward(h)
	if err != nil {
		return nil, err
	}
	if logits.DType() != dtype.F32 {
		return ops.Cast(logits, dtype.F32)
	}
	return logits, nil
}

// runBlock: x + post(mixer(norm(x))), then x + post(MLP(norm(x))).
func (m *LFM3) runBlock(blk *lfm3Block, x *tensor.Tensor, B, T, prior int, rope [2]*tensor.Tensor, dev device.Device, cache *KVCache, layer int) (*tensor.Tensor, error) {
	h, err := blk.OpNorm.Forward(x)
	if err != nil {
		return nil, err
	}
	var o *tensor.Tensor
	if blk.kind == "conv" {
		o, err = m.shortConv(blk, h, B, T, cache, layer)
	} else {
		o, err = m.attention(blk, h, B, T, prior, rope, dev, cache, layer)
	}
	if err != nil {
		return nil, err
	}
	if blk.PostOpNorm != nil {
		if o, err = blk.PostOpNorm.Forward(o); err != nil {
			return nil, err
		}
	}
	if x, err = ops.Add(x, o); err != nil {
		return nil, err
	}
	if h, err = blk.FFNNorm.Forward(x); err != nil {
		return nil, err
	}
	gate, err := blk.W1.Forward(h)
	if err != nil {
		return nil, err
	}
	if gate, err = ops.SiLU(gate); err != nil {
		return nil, err
	}
	up, err := blk.W3.Forward(h)
	if err != nil {
		return nil, err
	}
	if gate, err = ops.Mul(gate, up); err != nil {
		return nil, err
	}
	f, err := blk.W2.Forward(gate)
	if err != nil {
		return nil, err
	}
	if blk.PostFFNNorm != nil {
		if f, err = blk.PostFFNNorm.Forward(f); err != nil {
			return nil, err
		}
	}
	return ops.Add(x, f)
}

// shortConv is LFM2/LFM3's gated short convolution: in_proj splits into
// (B, C, x); y = C · conv(B · x) with a causal depthwise conv over time;
// out_proj. The conv runs in F32 and carries its last K−1 inputs in the
// cache.
func (m *LFM3) shortConv(blk *lfm3Block, h *tensor.Tensor, B, T int, cache *KVCache, layer int) (*tensor.Tensor, error) {
	D := m.Cfg.HiddenSize
	bcx, err := blk.InProj.Forward(h)
	if err != nil {
		return nil, err
	}
	if bcx.DType() != dtype.F32 {
		if bcx, err = ops.Cast(bcx, dtype.F32); err != nil {
			return nil, err
		}
	}
	bg, err := narrowContig(bcx, 2, 0, D)
	if err != nil {
		return nil, err
	}
	cg, err := narrowContig(bcx, 2, D, D)
	if err != nil {
		return nil, err
	}
	xv, err := narrowContig(bcx, 2, 2*D, D)
	if err != nil {
		return nil, err
	}
	y, err := ops.Mul(bg, xv)
	if err != nil {
		return nil, err
	}
	if y, err = transposeContig(y, 1, 2); err != nil { // (B, D, T)
		return nil, err
	}
	var state *tensor.Tensor
	if cache != nil {
		state = cache.Conv[layer]
	}
	y, state, err = ops.CausalConv1dDepthwise(y, blk.ConvW, state, false)
	if err != nil {
		return nil, err
	}
	if cache != nil {
		cache.Conv[layer] = state
	}
	if y, err = transposeContig(y, 1, 2); err != nil {
		return nil, err
	}
	if y, err = ops.Mul(cg, y); err != nil {
		return nil, err
	}
	if act := m.Cfg.activationDType(); act != dtype.F32 {
		if y, err = ops.Cast(y, act); err != nil {
			return nil, err
		}
	}
	return blk.OutProj.Forward(y)
}

// attention: GQA with per-head QK RMSNorm, zero-tail RoPE, a sliding window
// on sliding layers, and a per-head sigmoid gate from gate_proj on the
// output before o_proj.
func (m *LFM3) attention(blk *lfm3Block, h *tensor.Tensor, B, T, prior int, rope [2]*tensor.Tensor, dev device.Device, cache *KVCache, layer int) (*tensor.Tensor, error) {
	cfg := m.Cfg
	NH, KH, HD := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	q, err := blk.QProj.Forward(h)
	if err != nil {
		return nil, err
	}
	k, err := blk.KProj.Forward(h)
	if err != nil {
		return nil, err
	}
	v, err := blk.VProj.Forward(h)
	if err != nil {
		return nil, err
	}
	if q, err = headsForward(q, B, T, NH, HD); err != nil {
		return nil, err
	}
	if k, err = headsForward(k, B, T, KH, HD); err != nil {
		return nil, err
	}
	if v, err = headsForward(v, B, T, KH, HD); err != nil {
		return nil, err
	}
	if q, err = blk.QNorm.Forward(q); err != nil {
		return nil, err
	}
	if k, err = blk.KNorm.Forward(k); err != nil {
		return nil, err
	}
	if q, err = rotateHalfLast(q, rope[0], rope[1]); err != nil {
		return nil, err
	}
	if k, err = rotateHalfLast(k, rope[0], rope[1]); err != nil {
		return nil, err
	}
	kvPrior, totalT := 0, T
	if cache != nil {
		if cache.K[layer] != nil {
			kvPrior = cache.K[layer].Dim(2)
			if k, err = ops.Concat([]*tensor.Tensor{cache.K[layer], k}, 2); err != nil {
				return nil, err
			}
			if v, err = ops.Concat([]*tensor.Tensor{cache.V[layer], v}, 2); err != nil {
				return nil, err
			}
		}
		totalT = kvPrior + T
		cache.K[layer], cache.V[layer] = k, v
	}
	g := NH / KH
	if k, err = repeatKV(k, g, B, KH, totalT, HD); err != nil {
		return nil, err
	}
	if v, err = repeatKV(v, g, B, KH, totalT, HD); err != nil {
		return nil, err
	}
	if q, err = autograd.Reshape(q, B*NH, T, HD); err != nil {
		return nil, err
	}
	if k, err = autograd.Reshape(k, B*NH, totalT, HD); err != nil {
		return nil, err
	}
	if v, err = autograd.Reshape(v, B*NH, totalT, HD); err != nil {
		return nil, err
	}
	kT, err := transposeContig(k, 1, 2)
	if err != nil {
		return nil, err
	}
	scores, err := autograd.MatMul(q, kT)
	if err != nil {
		return nil, err
	}
	if scores, err = ops.Mul(scores, m.attnScale); err != nil {
		return nil, err
	}
	window := 0
	if blk.kind == "sliding_attention" {
		window = cfg.SlidingWindow
	}
	if T > 1 || (window > 0 && kvPrior+1 > window) {
		mask, err := m.mask(T, kvPrior, window, scores.DType(), dev)
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
	if ctx, err = autograd.Permute(ctx, 0, 2, 1, 3); err != nil { // (B, T, NH, HD)
		return nil, err
	}
	if ctx, err = autograd.Contiguous(ctx); err != nil {
		return nil, err
	}
	gate, err := blk.GateProj.Forward(h)
	if err != nil {
		return nil, err
	}
	if gate, err = ops.Sigmoid(gate); err != nil {
		return nil, err
	}
	if gate, err = autograd.Reshape(gate, B, T, NH, 1); err != nil {
		return nil, err
	}
	if ctx, err = ops.Mul(ctx, gate); err != nil {
		return nil, err
	}
	if ctx, err = autograd.Reshape(ctx, B, T, NH*HD); err != nil {
		return nil, err
	}
	return blk.OProj.Forward(ctx)
}

// ropeTables returns cos / sin (T, HD) for positions prior … prior+T−1 of
// one attention type, in the zero-tail layout.
func (m *LFM3) ropeTables(kind string, prior, T int, dt dtype.DType, dev device.Device) (*tensor.Tensor, *tensor.Tensor, error) {
	HD := m.Cfg.HeadDim
	base, partial := m.Cfg.FullRopeBase, m.Cfg.FullPartialRotary
	if kind == "sliding_attention" {
		base, partial = m.Cfg.SlidingRopeBase, m.Cfg.SlidingPartialRotary
	}
	if partial <= 0 {
		partial = 1
	}
	n := int(partial*float64(HD)) / 2
	inv := make([]float64, HD/2)
	for i := range n {
		inv[i] = 1 / math.Pow(base, float64(2*i)/float64(HD))
	}
	cos := make([]float32, T*HD)
	sin := make([]float32, T*HD)
	for t := range T {
		p := float64(prior + t)
		for i, f := range inv {
			// torch computes the angle in F32 (position · inv_freq) before cos / sin.
			a := float64(float32(p) * float32(f))
			c, s := float32(math.Cos(a)), float32(math.Sin(a))
			cos[t*HD+i], cos[t*HD+i+HD/2] = c, c
			sin[t*HD+i], sin[t*HD+i+HD/2] = s, s
		}
	}
	ct, err := tensor.New(cos, []int{T, HD}, dev)
	if err != nil {
		return nil, nil, err
	}
	st, err := tensor.New(sin, []int{T, HD}, dev)
	if err != nil {
		return nil, nil, err
	}
	if dt != dtype.F32 {
		if ct, err = ops.Cast(ct, dt); err != nil {
			return nil, nil, err
		}
		if st, err = ops.Cast(st, dt); err != nil {
			return nil, nil, err
		}
	}
	return ct, st, nil
}

// mask returns the cached additive (T, prior+T) mask: causal, and with
// window > 0 key j is visible from the query at position p only when
// p − window < j.
func (m *LFM3) mask(T, prior, window int, dt dtype.DType, dev device.Device) (*tensor.Tensor, error) {
	if m.masks == nil || m.masksDev == nil || !m.masksDev.SameAs(dev) {
		m.masks = map[[3]int]*tensor.Tensor{}
		m.masksDev = dev
	}
	key := [3]int{T, prior, window}
	if t, ok := m.masks[key]; ok && t.DType() == dt {
		return t, nil
	}
	total := prior + T
	negInf := float32(math.Inf(-1))
	data := make([]float32, T*total)
	for i := range T {
		p := prior + i
		for j := range total {
			if j > p || (window > 0 && j <= p-window) {
				data[i*total+j] = negInf
			}
		}
	}
	t, err := tensor.New(data, []int{T, total}, dev)
	if err != nil {
		return nil, err
	}
	if dt != dtype.F32 {
		if t, err = ops.Cast(t, dt); err != nil {
			return nil, err
		}
	}
	if len(m.masks) > 64 {
		m.masks = map[[3]int]*tensor.Tensor{}
	}
	m.masks[key] = t
	return t, nil
}

var _ CausalLM = (*LFM3)(nil)
