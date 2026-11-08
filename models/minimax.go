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

// MiniMaxConfig is `model_type: minimax` (MiniMax-Text-01), from
// transformers' modeling_minimax: lightning-attention layers (linear
// attention with a fixed per-head exponential decay) interleaved with
// softmax-attention layers, a top-k MoE in every layer, and DeepNorm-style
// residuals — each sub-block adds alpha · norm(x) + beta · f(norm(x)), so
// the residual stream carries the *normed* hidden state.
type MiniMaxConfig struct {
	VocabSize, HiddenSize, IntermediateSize, NumLayers int
	NumHeads, NumKVHeads, HeadDim                      int
	LinearLayers                                       []bool // per layer: true = lightning attention
	NumExperts, TopK                                   int
	// Residual factors: h = alpha · residual + beta · sublayer.
	LinearAlpha, LinearBeta, FullAlpha, FullBeta, MLPAlpha, MLPBeta float64
	RopeBase                                                        float64
	RmsEps                                                          float64
	MaxPosition                                                     int
	TieEmbed                                                        bool
	WeightDType                                                     dtype.DType
}

func (c MiniMaxConfig) activationDType() dtype.DType {
	if c.WeightDType == dtype.BF16 {
		return dtype.BF16
	}
	return dtype.F32
}

// slopes returns the per-head decay rates of lightning layer `layer`:
// base^(h+1) · (1 − layer / (L − 1 + 1e-5) + 1e-5), base = 2^(−8 / H)
// (transformers' get_slope_rate). The state decays by exp(−slope) a token.
func (c MiniMaxConfig) slopes(layer int) []float64 {
	base := 1 / math.Pow(2, 8/float64(c.NumHeads))
	factor := 1 - float64(layer)/(float64(c.NumLayers)-1+1e-5) + 1e-5
	s := make([]float64, c.NumHeads)
	for h := range s {
		s[h] = float64(float32(math.Pow(base, float64(h+1)))) * factor
	}
	return s
}

type minimaxBlock struct {
	linear           bool
	InNorm, PostNorm *nn.RMSNorm
	// Lightning attention.
	QKV, OutGate, Out *nn.Linear
	LNorm             *nn.RMSNorm // over NumHeads · HeadDim, eps 1e-6
	slopes            []float64
	// Softmax attention.
	QProj, KProj, VProj, OProj *nn.Linear
	MoE                        *nn.SparseMoE
}

// MiniMax is the MiniMax-Text-01 model; it implements CausalLM. Its KVCache
// keeps K / V for the softmax layers and the (B, H, d, d) F32 state in Rec
// for the lightning layers.
type MiniMax struct {
	Cfg       MiniMaxConfig
	TokenEmb  *nn.Embedding
	Blocks    []*minimaxBlock
	FinalNorm *nn.RMSNorm
	LMHead    *nn.Linear

	masks   map[[2]int]*tensor.Tensor
	maskDev device.Device
}

// NewMiniMax allocates an uninitialised model; LoadMiniMaxFromHF fills it.
func NewMiniMax(cfg MiniMaxConfig, dev device.Device) (*MiniMax, error) {
	if len(cfg.LinearLayers) != cfg.NumLayers {
		return nil, fmt.Errorf("minimax: %d layer types for %d layers", len(cfg.LinearLayers), cfg.NumLayers)
	}
	if cfg.NumKVHeads <= 0 || cfg.NumHeads%cfg.NumKVHeads != 0 {
		return nil, fmt.Errorf("minimax: NumHeads %d not divisible by NumKVHeads %d", cfg.NumHeads, cfg.NumKVHeads)
	}
	if cfg.WeightDType != dtype.BF16 {
		cfg.WeightDType = dtype.F32
	}
	act := cfg.activationDType()
	D, H, KH, HD := cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	emb, err := nn.NewEmbeddingUninit(cfg.VocabSize, D, act, dev)
	if err != nil {
		return nil, err
	}
	lin := func(in, out int) (*nn.Linear, error) { return nn.NewLinearUninit(in, out, false, act, dev) }
	blocks := make([]*minimaxBlock, cfg.NumLayers)
	for i := range blocks {
		b := &minimaxBlock{linear: cfg.LinearLayers[i]}
		var errs []error
		push := func(err error) {
			if err != nil {
				errs = append(errs, err)
			}
		}
		b.InNorm, err = nn.NewRMSNormWithDType(D, cfg.RmsEps, act, dev)
		push(err)
		b.PostNorm, err = nn.NewRMSNormWithDType(D, cfg.RmsEps, act, dev)
		push(err)
		if b.linear {
			b.QKV, err = lin(D, 3*H*HD)
			push(err)
			b.OutGate, err = lin(D, H*HD)
			push(err)
			b.Out, err = lin(H*HD, D)
			push(err)
			b.LNorm, err = nn.NewRMSNormWithDType(H*HD, 1e-6, act, dev) // MiniMaxRMSNorm default eps
			push(err)
			b.slopes = cfg.slopes(i)
		} else {
			b.QProj, err = lin(D, H*HD)
			push(err)
			b.KProj, err = lin(D, KH*HD)
			push(err)
			b.VProj, err = lin(D, KH*HD)
			push(err)
			b.OProj, err = lin(H*HD, D)
			push(err)
		}
		b.MoE, err = nn.NewSparseMoEUninit(D, cfg.IntermediateSize, cfg.NumExperts, cfg.TopK, true, act, dev)
		push(err)
		if len(errs) > 0 {
			return nil, fmt.Errorf("minimax layer %d: %w", i, errs[0])
		}
		blocks[i] = b
	}
	fn, err := nn.NewRMSNormWithDType(D, cfg.RmsEps, act, dev)
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
	return &MiniMax{Cfg: cfg, TokenEmb: emb, Blocks: blocks, FinalNorm: fn, LMHead: head}, nil
}

func (m *MiniMax) NumLayers() int     { return m.Cfg.NumLayers }
func (m *MiniMax) VocabSize() int     { return m.Cfg.VocabSize }
func (m *MiniMax) MaxContext() int    { return m.Cfg.MaxPosition }
func (m *MiniMax) SlidingWindow() int { return 0 }

// Parameters lists every weight once.
func (m *MiniMax) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokenEmb.Weight}
	for _, b := range m.Blocks {
		for _, n := range []*nn.RMSNorm{b.InNorm, b.PostNorm, b.LNorm} {
			if n != nil {
				ps = append(ps, n.Weight)
			}
		}
		for _, l := range []*nn.Linear{b.QKV, b.OutGate, b.Out, b.QProj, b.KProj, b.VProj, b.OProj} {
			if l != nil {
				ps = append(ps, l.W)
			}
		}
		ps = append(ps, b.MoE.Gate.W, b.MoE.GateUp, b.MoE.Down)
	}
	ps = append(ps, m.FinalNorm.Weight)
	if !m.Cfg.TieEmbed {
		ps = append(ps, m.LMHead.W)
	}
	return ps
}

func (m *MiniMax) ParamCount() int {
	n := 0
	for _, p := range m.Parameters() {
		n += p.NumElements()
	}
	return n
}

// ForwardWithCache runs tokens (B, T) I64 and returns F32 logits (B, T, V).
func (m *MiniMax) ForwardWithCache(tokens *tensor.Tensor, cache *KVCache) (*tensor.Tensor, error) {
	if tokens.Rank() != 2 || tokens.DType() != dtype.I64 {
		return nil, fmt.Errorf("MiniMax.Forward: tokens must be (B, T) I64, got %v %v", tokens.Shape(), tokens.DType())
	}
	B, T := tokens.Dim(0), tokens.Dim(1)
	dev := tokens.Device()
	prior := 0
	if cache != nil {
		if len(cache.K) != m.Cfg.NumLayers {
			return nil, fmt.Errorf("MiniMax: cache has %d layers, model %d", len(cache.K), m.Cfg.NumLayers)
		}
		if cache.Rec == nil {
			cache.Rec = make([]*tensor.Tensor, m.Cfg.NumLayers)
		}
		prior = cache.Len
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
		if h, err = m.runBlock(blk, h, B, T, prior, pos, cache, i); err != nil {
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

// residualMix returns alpha · r + beta · f, formed in F32 (the factors are
// Python floats in the reference) and rounded once to r's dtype.
func residualMix(r, f *tensor.Tensor, alpha, beta float64) (*tensor.Tensor, error) {
	if alpha == 1 && beta == 1 {
		return ops.Add(r, f)
	}
	dt := r.DType()
	toF32 := func(x *tensor.Tensor) (*tensor.Tensor, error) {
		if x.DType() == dtype.F32 {
			return x, nil
		}
		return ops.Cast(x, dtype.F32)
	}
	r32, err := toF32(r)
	if err != nil {
		return nil, err
	}
	f32, err := toF32(f)
	if err != nil {
		return nil, err
	}
	if r32, err = mulScalarF32(r32, alpha); err != nil {
		return nil, err
	}
	if f32, err = mulScalarF32(f32, beta); err != nil {
		return nil, err
	}
	out, err := ops.Add(r32, f32)
	if err != nil || dt == dtype.F32 {
		return out, err
	}
	return ops.Cast(out, dt)
}

func (m *MiniMax) runBlock(blk *minimaxBlock, x *tensor.Tensor, B, T, prior int, pos *tensor.Tensor, cache *KVCache, layer int) (*tensor.Tensor, error) {
	cfg := m.Cfg
	h, err := blk.InNorm.Forward(x)
	if err != nil {
		return nil, err
	}
	var a *tensor.Tensor
	alpha, beta := cfg.FullAlpha, cfg.FullBeta
	if blk.linear {
		a, err = m.lightning(blk, h, B, T, cache, layer)
		alpha, beta = cfg.LinearAlpha, cfg.LinearBeta
	} else {
		a, err = m.softmaxAttention(blk, h, B, T, prior, pos, cache, layer)
	}
	if err != nil {
		return nil, err
	}
	// DeepNorm-style: the residual is the normed input.
	if h, err = residualMix(h, a, alpha, beta); err != nil {
		return nil, err
	}
	if h, err = blk.PostNorm.Forward(h); err != nil {
		return nil, err
	}
	f, err := blk.MoE.Forward(h)
	if err != nil {
		return nil, err
	}
	return residualMix(h, f, cfg.MLPAlpha, cfg.MLPBeta)
}

// lightning is MiniMax's lightning attention: per head, a d × d state with a
// fixed decay a = exp(−slope),
//
//	S_t = a · S_{t−1} + k_tᵀ v_t,   o_t = q_t · S_t,
//
// with q, k, v = SiLU(qkv_proj(h)) split per head. A whole call is evaluated
// in closed form (transformers' block algorithm computes the same thing):
//
//	o = ((Q Kᵀ) ⊙ D) V + (Q ⊙ a^(i+1)) S₀,   D[i, j] = a^(i−j) for i ≥ j,
//	S_T = a^T S₀ + (K ⊙ a^(T−1−j))ᵀ V,
//
// in F32. The cost is O(T²) per call, fine for decode and moderate prompts.
// Then RMSNorm over all heads, a sigmoid output gate and out_proj.
func (m *MiniMax) lightning(blk *minimaxBlock, h *tensor.Tensor, B, T int, cache *KVCache, layer int) (*tensor.Tensor, error) {
	cfg := m.Cfg
	H, d := cfg.NumHeads, cfg.HeadDim
	dev := h.Device()
	qkv, err := blk.QKV.Forward(h)
	if err != nil {
		return nil, err
	}
	if qkv, err = ops.SiLU(qkv); err != nil {
		return nil, err
	}
	if qkv.DType() != dtype.F32 {
		if qkv, err = ops.Cast(qkv, dtype.F32); err != nil {
			return nil, err
		}
	}
	// (B, T, H·3d) → per head [q | k | v] → (B, H, T, 3d).
	if qkv, err = headsForward(qkv, B, T, H, 3*d); err != nil {
		return nil, err
	}
	q, err := narrowContig(qkv, 3, 0, d)
	if err != nil {
		return nil, err
	}
	k, err := narrowContig(qkv, 3, d, d)
	if err != nil {
		return nil, err
	}
	v, err := narrowContig(qkv, 3, 2*d, d)
	if err != nil {
		return nil, err
	}

	// Decay tables on the host: D (1, H, T, T), qdec (1, H, T, 1),
	// kdec (1, H, T, 1), aT (1, H, 1, 1).
	Dt := make([]float32, H*T*T)
	qd := make([]float32, H*T)
	kd := make([]float32, H*T)
	aT := make([]float32, H)
	for hh, s := range blk.slopes {
		for i := range T {
			for j := 0; j <= i; j++ {
				Dt[(hh*T+i)*T+j] = float32(math.Exp(-s * float64(i-j)))
			}
			qd[hh*T+i] = float32(math.Exp(-s * float64(i+1)))
			kd[hh*T+i] = float32(math.Exp(-s * float64(T-1-i)))
		}
		aT[hh] = float32(math.Exp(-s * float64(T)))
	}
	mk := func(data []float32, dims ...int) (*tensor.Tensor, error) { return tensor.New(data, dims, dev) }
	Dm, err := mk(Dt, 1, H, T, T)
	if err != nil {
		return nil, err
	}
	qdec, err := mk(qd, 1, H, T, 1)
	if err != nil {
		return nil, err
	}
	kdec, err := mk(kd, 1, H, T, 1)
	if err != nil {
		return nil, err
	}
	aTt, err := mk(aT, 1, H, 1, 1)
	if err != nil {
		return nil, err
	}

	bmm := func(a, b *tensor.Tensor, M, K, N int) (*tensor.Tensor, error) { // (B, H, M, K) · (B, H, K, N)
		a3, err := autograd.Reshape(a, B*H, M, K)
		if err != nil {
			return nil, err
		}
		b3, err := autograd.Reshape(b, B*H, K, N)
		if err != nil {
			return nil, err
		}
		o, err := autograd.MatMul(a3, b3)
		if err != nil {
			return nil, err
		}
		return autograd.Reshape(o, B, H, M, N)
	}
	tr := func(x *tensor.Tensor) (*tensor.Tensor, error) { return transposeContig(x, 2, 3) }

	kT, err := tr(k)
	if err != nil {
		return nil, err
	}
	scores, err := bmm(q, kT, T, d, T) // (B, H, T, T)
	if err != nil {
		return nil, err
	}
	if scores, err = ops.Mul(scores, Dm); err != nil {
		return nil, err
	}
	o, err := bmm(scores, v, T, T, d) // intra-call part
	if err != nil {
		return nil, err
	}
	var s0 *tensor.Tensor
	if cache != nil {
		s0 = cache.Rec[layer]
	}
	if s0 != nil {
		qa, err := ops.Mul(q, qdec)
		if err != nil {
			return nil, err
		}
		inter, err := bmm(qa, s0, T, d, d)
		if err != nil {
			return nil, err
		}
		if o, err = ops.Add(o, inter); err != nil {
			return nil, err
		}
	}
	if cache != nil {
		kw, err := ops.Mul(k, kdec)
		if err != nil {
			return nil, err
		}
		kwT, err := tr(kw)
		if err != nil {
			return nil, err
		}
		s, err := bmm(kwT, v, d, T, d)
		if err != nil {
			return nil, err
		}
		if s0 != nil {
			decayed, err := ops.Mul(s0, aTt)
			if err != nil {
				return nil, err
			}
			if s, err = ops.Add(s, decayed); err != nil {
				return nil, err
			}
		}
		cache.Rec[layer] = s
	}

	// (B, H, T, d) → (B, T, H·d), norm, gate, out.
	if o, err = autograd.Permute(o, 0, 2, 1, 3); err != nil {
		return nil, err
	}
	if o, err = autograd.Contiguous(o); err != nil {
		return nil, err
	}
	if o, err = autograd.Reshape(o, B, T, H*d); err != nil {
		return nil, err
	}
	if act := cfg.activationDType(); act != dtype.F32 {
		if o, err = ops.Cast(o, act); err != nil {
			return nil, err
		}
	}
	if o, err = blk.LNorm.Forward(o); err != nil {
		return nil, err
	}
	gate, err := blk.OutGate.Forward(h)
	if err != nil {
		return nil, err
	}
	if gate, err = ops.Sigmoid(gate); err != nil {
		return nil, err
	}
	if o, err = ops.Mul(o, gate); err != nil {
		return nil, err
	}
	return blk.Out.Forward(o)
}

// softmaxAttention is GQA with full-head rotate-half RoPE and a causal mask.
func (m *MiniMax) softmaxAttention(blk *minimaxBlock, h *tensor.Tensor, B, T, prior int, pos *tensor.Tensor, cache *KVCache, layer int) (*tensor.Tensor, error) {
	cfg := m.Cfg
	H, KH, HD := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
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
	if q, err = headsForward(q, B, T, H, HD); err != nil {
		return nil, err
	}
	if k, err = headsForward(k, B, T, KH, HD); err != nil {
		return nil, err
	}
	if v, err = headsForward(v, B, T, KH, HD); err != nil {
		return nil, err
	}
	if q, err = ops.RoPE(q, pos, cfg.RopeBase); err != nil {
		return nil, err
	}
	if k, err = ops.RoPE(k, pos, cfg.RopeBase); err != nil {
		return nil, err
	}
	if cache != nil {
		if cache.K[layer] != nil {
			if k, err = ops.Concat([]*tensor.Tensor{cache.K[layer], k}, 2); err != nil {
				return nil, err
			}
			if v, err = ops.Concat([]*tensor.Tensor{cache.V[layer], v}, 2); err != nil {
				return nil, err
			}
		}
		cache.K[layer], cache.V[layer] = k, v
	}
	Tt := k.Dim(2)
	if k, err = repeatKV(k, H/KH, B, KH, Tt, HD); err != nil {
		return nil, err
	}
	if v, err = repeatKV(v, H/KH, B, KH, Tt, HD); err != nil {
		return nil, err
	}
	if q, err = autograd.Reshape(q, B*H, T, HD); err != nil {
		return nil, err
	}
	if k, err = autograd.Reshape(k, B*H, Tt, HD); err != nil {
		return nil, err
	}
	if v, err = autograd.Reshape(v, B*H, Tt, HD); err != nil {
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
	if scores, err = mulScalarF32(scores, 1/math.Sqrt(float64(HD))); err != nil {
		return nil, err
	}
	if T > 1 {
		mask, err := m.causalMask(T, Tt-T, scores.DType(), scores.Device())
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
	if ctx, err = autograd.Reshape(ctx, B, H, T, HD); err != nil {
		return nil, err
	}
	if ctx, err = autograd.Permute(ctx, 0, 2, 1, 3); err != nil {
		return nil, err
	}
	if ctx, err = autograd.Contiguous(ctx); err != nil {
		return nil, err
	}
	if ctx, err = autograd.Reshape(ctx, B, T, H*HD); err != nil {
		return nil, err
	}
	return blk.OProj.Forward(ctx)
}

// causalMask returns the cached (T, prior+T) additive causal mask.
func (m *MiniMax) causalMask(T, prior int, dt dtype.DType, dev device.Device) (*tensor.Tensor, error) {
	if m.masks == nil || m.maskDev == nil || !m.maskDev.SameAs(dev) {
		m.masks, m.maskDev = map[[2]int]*tensor.Tensor{}, dev
	}
	key := [2]int{T, prior}
	if t, ok := m.masks[key]; ok && t.DType() == dt {
		return t, nil
	}
	total := prior + T
	data := make([]float32, T*total)
	negInf := float32(math.Inf(-1))
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

var _ CausalLM = (*MiniMax)(nil)
