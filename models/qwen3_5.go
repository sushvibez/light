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

// Qwen35Config is the text decoder of `model_type: qwen3_5` (Qwen3.5 0.8B …
// 27B, Qwen3.6, Qwen3.8): a hybrid stack in which most layers are Gated
// DeltaNet linear attention and every few is gated full attention.
// See docs/design/2026-11-06-gated-deltanet.md.
type Qwen35Config struct {
	VocabSize, HiddenSize, IntermediateSize, NumLayers int
	LinearLayers                                       []bool // per layer: true = Gated DeltaNet, false = full attention

	// Full attention: GQA with QK-norm, partial RoPE and (AttnOutputGate)
	// a per-head sigmoid output gate packed into q_proj.
	NumHeads, NumKVHeads, HeadDim int
	PartialRotary                 float64 // fraction of HeadDim RoPE rotates (0.25); 0 or 1 = all
	RopeBase                      float64
	AttnOutputGate                bool

	// Gated DeltaNet: key / value heads and dims, short-conv kernel.
	LinearKHeads, LinearVHeads, LinearKDim, LinearVDim, ConvKernel int

	RmsEps      float64
	MaxPosition int
	TieEmbed    bool
	WeightDType dtype.DType // F32 or BF16; activations run at the same dtype
}

func (c Qwen35Config) activationDType() dtype.DType {
	if c.WeightDType == dtype.BF16 {
		return dtype.BF16
	}
	return dtype.F32
}

func (c Qwen35Config) rotaryDim() int {
	if f := c.PartialRotary; f > 0 && f < 1 {
		return int(f*float64(c.HeadDim)) &^ 1
	}
	return c.HeadDim
}

// convChannels is the width of the fused q/k/v projection the short conv
// runs over.
func (c Qwen35Config) convChannels() int {
	return 2*c.LinearKHeads*c.LinearKDim + c.LinearVHeads*c.LinearVDim
}

type qwen35Block struct {
	linear           bool
	InNorm, PostNorm *nn.RMSNorm // zero-centred: (1 + w) stored pre-added

	// Full attention.
	QProj, KProj, VProj, OProj *nn.Linear
	QNorm, KNorm               *nn.RMSNorm

	// Gated DeltaNet.
	InQKV, InZ, InB, InA, OutProj *nn.Linear
	ConvW                         *tensor.Tensor // (C, K) F32
	NegA                          *tensor.Tensor // −exp(A_log), (Hv,) F32
	DtBias                        *tensor.Tensor // (Hv,) F32
	GNorm                         *nn.RMSNorm    // gated output norm over dv, plain weight, F32

	Gate, Up, Down *nn.Linear
}

// Qwen35 is the Qwen3.5 text model. It implements CausalLM; its KVCache
// holds K/V for full-attention layers and conv + delta-rule state for the
// linear ones.
type Qwen35 struct {
	Cfg       Qwen35Config
	TokenEmb  *nn.Embedding
	Blocks    []*qwen35Block
	FinalNorm *nn.RMSNorm
	LMHead    *nn.Linear

	attnScale *tensor.Tensor
	masks     map[[2]int]*tensor.Tensor
	masksDev  device.Device
}

// NewQwen35 allocates an uninitialised model; LoadQwen35FromHF fills it.
func NewQwen35(cfg Qwen35Config, dev device.Device) (*Qwen35, error) {
	if len(cfg.LinearLayers) != cfg.NumLayers {
		return nil, fmt.Errorf("qwen3.5: %d layer types for %d layers", len(cfg.LinearLayers), cfg.NumLayers)
	}
	if cfg.NumKVHeads <= 0 || cfg.NumHeads%cfg.NumKVHeads != 0 {
		return nil, fmt.Errorf("qwen3.5: NumHeads %d not divisible by NumKVHeads %d", cfg.NumHeads, cfg.NumKVHeads)
	}
	if cfg.LinearKHeads <= 0 || cfg.LinearVHeads%cfg.LinearKHeads != 0 {
		return nil, fmt.Errorf("qwen3.5: linear value heads %d not divisible by key heads %d", cfg.LinearVHeads, cfg.LinearKHeads)
	}
	if cfg.WeightDType != dtype.BF16 {
		cfg.WeightDType = dtype.F32
	}
	act := cfg.activationDType()
	D, NH, KH, HD, IM := cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.IntermediateSize
	emb, err := nn.NewEmbeddingUninit(cfg.VocabSize, D, act, dev)
	if err != nil {
		return nil, err
	}
	norm := func(n int, dt dtype.DType) (*nn.RMSNorm, error) {
		return nn.NewRMSNormWithDType(n, cfg.RmsEps, dt, dev)
	}
	lin := func(in, out int) (*nn.Linear, error) { return nn.NewLinearUninit(in, out, false, act, dev) }
	f32 := func(dims ...int) (*tensor.Tensor, error) { return tensor.Zeros(dtype.F32, dims, dev) }
	qOut := NH * HD
	if cfg.AttnOutputGate {
		qOut *= 2
	}
	Hv, dv := cfg.LinearVHeads, cfg.LinearVDim
	blocks := make([]*qwen35Block, cfg.NumLayers)
	for i := range blocks {
		b := &qwen35Block{linear: cfg.LinearLayers[i]}
		var errs []error
		push := func(err error) {
			if err != nil {
				errs = append(errs, err)
			}
		}
		b.InNorm, err = norm(D, act)
		push(err)
		b.PostNorm, err = norm(D, act)
		push(err)
		if b.linear {
			b.InQKV, err = lin(D, cfg.convChannels())
			push(err)
			b.InZ, err = lin(D, Hv*dv)
			push(err)
			b.InB, err = lin(D, Hv)
			push(err)
			b.InA, err = lin(D, Hv)
			push(err)
			b.OutProj, err = lin(Hv*dv, D)
			push(err)
			b.ConvW, err = f32(cfg.convChannels(), cfg.ConvKernel)
			push(err)
			b.NegA, err = f32(Hv)
			push(err)
			b.DtBias, err = f32(Hv)
			push(err)
			b.GNorm, err = norm(dv, dtype.F32)
			push(err)
		} else {
			b.QProj, err = lin(D, qOut)
			push(err)
			b.KProj, err = lin(D, KH*HD)
			push(err)
			b.VProj, err = lin(D, KH*HD)
			push(err)
			b.OProj, err = lin(NH*HD, D)
			push(err)
			b.QNorm, err = norm(HD, act)
			push(err)
			b.KNorm, err = norm(HD, act)
			push(err)
		}
		b.Gate, err = lin(D, IM)
		push(err)
		b.Up, err = lin(D, IM)
		push(err)
		b.Down, err = lin(IM, D)
		push(err)
		if len(errs) > 0 {
			return nil, fmt.Errorf("qwen3.5 layer %d: %w", i, errs[0])
		}
		blocks[i] = b
	}
	fn, err := norm(D, act)
	if err != nil {
		return nil, err
	}
	var head *nn.Linear
	if cfg.TieEmbed {
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
	return &Qwen35{Cfg: cfg, TokenEmb: emb, Blocks: blocks, FinalNorm: fn, LMHead: head, attnScale: scale}, nil
}

func (m *Qwen35) NumLayers() int     { return m.Cfg.NumLayers }
func (m *Qwen35) VocabSize() int     { return m.Cfg.VocabSize }
func (m *Qwen35) MaxContext() int    { return m.Cfg.MaxPosition }
func (m *Qwen35) SlidingWindow() int { return 0 }

// Parameters lists every weight tensor (the tied head counted once).
func (m *Qwen35) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokenEmb.Weight, m.FinalNorm.Weight}
	if !m.Cfg.TieEmbed {
		ps = append(ps, m.LMHead.W)
	}
	for _, b := range m.Blocks {
		ps = append(ps, b.InNorm.Weight, b.PostNorm.Weight, b.Gate.W, b.Up.W, b.Down.W)
		if b.linear {
			ps = append(ps, b.InQKV.W, b.InZ.W, b.InB.W, b.InA.W, b.OutProj.W, b.ConvW, b.NegA, b.DtBias, b.GNorm.Weight)
		} else {
			ps = append(ps, b.QProj.W, b.KProj.W, b.VProj.W, b.OProj.W, b.QNorm.Weight, b.KNorm.Weight)
		}
	}
	return ps
}

func (m *Qwen35) ParamCount() int {
	n := 0
	for _, p := range m.Parameters() {
		n += p.Shape().NumElements()
	}
	return n
}

// ForwardWithCache runs tokens (B, T) I64 and returns F32 logits (B, T, V).
func (m *Qwen35) ForwardWithCache(tokens *tensor.Tensor, cache *KVCache) (*tensor.Tensor, error) {
	if tokens.Rank() != 2 || tokens.DType() != dtype.I64 {
		return nil, fmt.Errorf("Qwen35.Forward: tokens must be (B, T) I64, got %v %v", tokens.Shape(), tokens.DType())
	}
	B, T := tokens.Dim(0), tokens.Dim(1)
	dev := tokens.Device()
	prior := 0
	if cache != nil {
		if len(cache.K) != m.Cfg.NumLayers || len(cache.Conv) != m.Cfg.NumLayers {
			return nil, fmt.Errorf("Qwen35.ForwardWithCache: cache has %d layers, model %d", len(cache.K), m.Cfg.NumLayers)
		}
		prior = cache.Len
	}
	if m.Cfg.MaxPosition > 0 && prior+T > m.Cfg.MaxPosition {
		return nil, fmt.Errorf("Qwen35.Forward: sequence length %d > MaxPosition %d", prior+T, m.Cfg.MaxPosition)
	}
	h, err := m.TokenEmb.Forward(tokens)
	if err != nil {
		return nil, err
	}
	posData := make([]int64, T)
	for i := range posData {
		posData[i] = int64(prior + i)
	}
	pos, err := tensor.New(posData, []int{T}, dev)
	if err != nil {
		return nil, err
	}
	for i, blk := range m.Blocks {
		if h, err = m.runBlock(blk, h, B, T, prior, pos, dev, cache, i); err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
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

// runBlock: x + mixer(norm(x)), then x + MLP(norm(x)).
func (m *Qwen35) runBlock(blk *qwen35Block, x *tensor.Tensor, B, T, prior int, pos *tensor.Tensor, dev device.Device, cache *KVCache, layer int) (*tensor.Tensor, error) {
	h, err := blk.InNorm.Forward(x)
	if err != nil {
		return nil, err
	}
	var o *tensor.Tensor
	if blk.linear {
		o, err = m.deltaNet(blk, h, B, T, cache, layer)
	} else {
		o, err = m.attention(blk, h, B, T, prior, pos, dev, cache, layer)
	}
	if err != nil {
		return nil, err
	}
	if x, err = ops.Add(x, o); err != nil {
		return nil, err
	}
	if h, err = blk.PostNorm.Forward(x); err != nil {
		return nil, err
	}
	gate, err := blk.Gate.Forward(h)
	if err != nil {
		return nil, err
	}
	if gate, err = ops.SiLU(gate); err != nil {
		return nil, err
	}
	up, err := blk.Up.Forward(h)
	if err != nil {
		return nil, err
	}
	if gate, err = ops.Mul(gate, up); err != nil {
		return nil, err
	}
	mlp, err := blk.Down.Forward(gate)
	if err != nil {
		return nil, err
	}
	return ops.Add(x, mlp)
}

// attention is gated full attention (transformers' Qwen3NextAttention):
// q_proj yields [query | gate] per head; QK-norm; partial RoPE; causal
// SDPA over the cached K/V; output × sigmoid(gate); o_proj.
func (m *Qwen35) attention(blk *qwen35Block, h *tensor.Tensor, B, T, prior int, pos *tensor.Tensor, dev device.Device, cache *KVCache, layer int) (*tensor.Tensor, error) {
	cfg := m.Cfg
	NH, KH, HD := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	qg, err := blk.QProj.Forward(h)
	if err != nil {
		return nil, err
	}
	q := qg
	var gate *tensor.Tensor
	if cfg.AttnOutputGate {
		if qg, err = autograd.Reshape(qg, B, T, NH, 2*HD); err != nil {
			return nil, err
		}
		if q, err = narrowContig(qg, 3, 0, HD); err != nil {
			return nil, err
		}
		if gate, err = narrowContig(qg, 3, HD, HD); err != nil {
			return nil, err
		}
		if q, err = autograd.Reshape(q, B, T, NH*HD); err != nil {
			return nil, err
		}
		if gate, err = autograd.Reshape(gate, B, T, NH*HD); err != nil {
			return nil, err
		}
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
	if q, err = m.rope(q, pos); err != nil {
		return nil, err
	}
	if k, err = m.rope(k, pos); err != nil {
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
	if scores, err = ops.Mul(scores, m.attnScale); err != nil {
		return nil, err
	}
	if T > 1 {
		mask, err := m.causalMask(T, kvPrior, scores.DType(), dev)
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
	if ctx, err = autograd.Reshape(ctx, B, T, NH*HD); err != nil {
		return nil, err
	}
	if gate != nil {
		sg, err := ops.Sigmoid(gate)
		if err != nil {
			return nil, err
		}
		if ctx, err = ops.Mul(ctx, sg); err != nil {
			return nil, err
		}
	}
	return blk.OProj.Forward(ctx)
}

// deltaNet is the Gated DeltaNet mixer (transformers' Qwen3_5GatedDeltaNet).
// The short conv, the delta rule and the gated norm run in F32, as the
// reference does; the projections run at the activation dtype.
func (m *Qwen35) deltaNet(blk *qwen35Block, h *tensor.Tensor, B, T int, cache *KVCache, layer int) (*tensor.Tensor, error) {
	cfg := m.Cfg
	Hk, Hv, dk, dv := cfg.LinearKHeads, cfg.LinearVHeads, cfg.LinearKDim, cfg.LinearVDim
	toF32 := func(x *tensor.Tensor) (*tensor.Tensor, error) {
		if x.DType() == dtype.F32 {
			return x, nil
		}
		return ops.Cast(x, dtype.F32)
	}
	qkv, err := blk.InQKV.Forward(h)
	if err != nil {
		return nil, err
	}
	if qkv, err = toF32(qkv); err != nil {
		return nil, err
	}
	// (B, T, C) → (B, C, T) for the conv, and back.
	if qkv, err = transposeContig(qkv, 1, 2); err != nil {
		return nil, err
	}
	var convState, recState *tensor.Tensor
	if cache != nil {
		convState, recState = cache.Conv[layer], cache.Rec[layer]
	}
	qkv, convState, err = ops.CausalConv1dDepthwise(qkv, blk.ConvW, convState, true)
	if err != nil {
		return nil, err
	}
	if qkv, err = transposeContig(qkv, 1, 2); err != nil {
		return nil, err
	}
	split := func(off, heads, dim int) (*tensor.Tensor, error) {
		x, err := narrowContig(qkv, 2, off, heads*dim)
		if err != nil {
			return nil, err
		}
		return autograd.Reshape(x, B, T, heads, dim)
	}
	q, err := split(0, Hk, dk)
	if err != nil {
		return nil, err
	}
	k, err := split(Hk*dk, Hk, dk)
	if err != nil {
		return nil, err
	}
	v, err := split(2*Hk*dk, Hv, dv)
	if err != nil {
		return nil, err
	}
	if r := Hv / Hk; r > 1 { // repeat_interleave over heads: value head j reads key head j / r
		if q, err = repeatHeadsInterleave(q, r); err != nil {
			return nil, err
		}
		if k, err = repeatHeadsInterleave(k, r); err != nil {
			return nil, err
		}
	}
	bProj, err := blk.InB.Forward(h)
	if err != nil {
		return nil, err
	}
	if bProj, err = toF32(bProj); err != nil {
		return nil, err
	}
	beta, err := ops.Sigmoid(bProj)
	if err != nil {
		return nil, err
	}
	a, err := blk.InA.Forward(h)
	if err != nil {
		return nil, err
	}
	if a, err = toF32(a); err != nil {
		return nil, err
	}
	if a, err = ops.Add(a, blk.DtBias); err != nil {
		return nil, err
	}
	if a, err = ops.Softplus(a); err != nil {
		return nil, err
	}
	g, err := ops.Mul(a, blk.NegA) // g = −exp(A_log) · softplus(a + dt_bias) ≤ 0
	if err != nil {
		return nil, err
	}
	out, recState, err := ops.GatedDeltaRule(q, k, v, g, beta, recState, ops.GatedDeltaOptions{QKL2Norm: true})
	if err != nil {
		return nil, err
	}
	if cache != nil {
		cache.Conv[layer], cache.Rec[layer] = convState, recState
	}
	// Gated RMSNorm over dv per head: norm(o) · w · silu(z).
	if out, err = blk.GNorm.Forward(out); err != nil {
		return nil, err
	}
	z, err := blk.InZ.Forward(h)
	if err != nil {
		return nil, err
	}
	if z, err = toF32(z); err != nil {
		return nil, err
	}
	if z, err = ops.SiLU(z); err != nil {
		return nil, err
	}
	if z, err = autograd.Reshape(z, B, T, Hv, dv); err != nil {
		return nil, err
	}
	if out, err = ops.Mul(out, z); err != nil {
		return nil, err
	}
	if out, err = autograd.Reshape(out, B, T, Hv*dv); err != nil {
		return nil, err
	}
	if act := cfg.activationDType(); act != dtype.F32 {
		if out, err = ops.Cast(out, act); err != nil {
			return nil, err
		}
	}
	return blk.OutProj.Forward(out)
}

// rope rotates the first rotaryDim dims of each head (partial RoPE); for
// text-only input MRoPE's three position streams are identical, so this is
// plain RoPE over those dims.
func (m *Qwen35) rope(x, pos *tensor.Tensor) (*tensor.Tensor, error) {
	rot := m.Cfg.rotaryDim()
	if rot >= m.Cfg.HeadDim {
		return ops.RoPE(x, pos, m.Cfg.RopeBase)
	}
	r, err := narrowContig(x, 3, 0, rot)
	if err != nil {
		return nil, err
	}
	if r, err = ops.RoPE(r, pos, m.Cfg.RopeBase); err != nil {
		return nil, err
	}
	pass, err := x.Narrow(3, rot, m.Cfg.HeadDim-rot)
	if err != nil {
		return nil, err
	}
	return ops.Concat([]*tensor.Tensor{r, pass}, 3)
}

// causalMask returns the cached (T, prior+T) additive causal mask.
func (m *Qwen35) causalMask(T, prior int, dt dtype.DType, dev device.Device) (*tensor.Tensor, error) {
	if m.masks == nil || m.masksDev == nil || !m.masksDev.SameAs(dev) {
		m.masks = map[[2]int]*tensor.Tensor{}
		m.masksDev = dev
	}
	key := [2]int{T, prior}
	if t, ok := m.masks[key]; ok && t.DType() == dt {
		return t, nil
	}
	total := prior + T
	negInf := float32(math.Inf(-1))
	data := make([]float32, T*total)
	for i := range T {
		for j := prior + i + 1; j < total; j++ {
			data[i*total+j] = negInf
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
		m.masks = map[[2]int]*tensor.Tensor{}
	}
	m.masks[key] = t
	return t, nil
}

// repeatHeadsInterleave maps (B, T, H, d) to (B, T, H·r, d) with each head
// repeated r times in place (torch repeat_interleave on dim 2).
func repeatHeadsInterleave(x *tensor.Tensor, r int) (*tensor.Tensor, error) {
	B, T, H, d := x.Dim(0), x.Dim(1), x.Dim(2), x.Dim(3)
	x5, err := autograd.Reshape(x, B, T, H, 1, d)
	if err != nil {
		return nil, err
	}
	parts := make([]*tensor.Tensor, r)
	for i := range parts {
		parts[i] = x5
	}
	cat, err := ops.Concat(parts, 3)
	if err != nil {
		return nil, err
	}
	return autograd.Reshape(cat, B, T, H*r, d)
}

func narrowContig(x *tensor.Tensor, dim, start, n int) (*tensor.Tensor, error) {
	v, err := x.Narrow(dim, start, n)
	if err != nil {
		return nil, err
	}
	return v.Contiguous()
}

func transposeContig(x *tensor.Tensor, i, j int) (*tensor.Tensor, error) {
	v, err := x.Transpose(i, j)
	if err != nil {
		return nil, err
	}
	return v.Contiguous()
}

var _ CausalLM = (*Qwen35)(nil)
