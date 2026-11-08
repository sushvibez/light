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

// DeepseekV2Config is `model_type: deepseek_v2` (DeepSeek-V2, V2-Lite,
// V2.5): Multi-head Latent Attention + DeepSeekMoE. See transformers'
// modeling_deepseek_v2.
//
// MLA: keys and values come from a compressed latent. kv_a_proj_with_mqa
// maps the hidden state to KVLoraRank latent dims (RMS-normed, then expanded
// per head by kv_b_proj into a QKNopeDim key part and a VHeadDim value) plus
// one QKRopeDim key part shared by all heads that carries the rotary
// position. Queries come from q_proj, or from q_a_proj → RMSNorm → q_b_proj
// when QLoraRank > 0, split per head the same way. The KV cache holds only
// the normed latent and the rotated shared key.
type DeepseekV2Config struct {
	VocabSize, HiddenSize, IntermediateSize, NumLayers, NumHeads int
	QLoraRank, KVLoraRank, QKNopeDim, QKRopeDim, VHeadDim        int

	// DeepSeekMoE: every MoELayerFreq-th layer from FirstKDense on routes
	// each token to TopK of NumExperts (softmax, F32) plus NumShared
	// always-on experts; routed weights × RoutedScale, renormalised first
	// when NormTopK. The other layers are a dense SwiGLU MLP.
	MoEIntermediate, NumExperts, TopK, NumShared, FirstKDense, MoELayerFreq int
	NormTopK                                                                bool
	RoutedScale                                                             float64

	RopeBase    float64
	RopeScaling *RopeScaling // YaRN (V2: factor 40, mscale_all_dim 0.707); nil = plain RoPE
	RmsEps      float64
	MaxPosition int
	TieEmbed    bool
	WeightDType dtype.DType // F32 or BF16; activations run at the same dtype
}

func (c DeepseekV2Config) activationDType() dtype.DType {
	if c.WeightDType == dtype.BF16 {
		return dtype.BF16
	}
	return dtype.F32
}

func (c DeepseekV2Config) isMoE(layer int) bool {
	f := c.MoELayerFreq
	if f <= 0 {
		f = 1
	}
	return c.NumExperts > 0 && layer >= c.FirstKDense && layer%f == 0
}

func (c DeepseekV2Config) qkHeadDim() int { return c.QKNopeDim + c.QKRopeDim }

// softmaxScale is qk_head_dim^-½, times mscale(factor, mscale_all_dim)²
// under YaRN (transformers' yarn_apply_mscale).
func (c DeepseekV2Config) softmaxScale() float64 {
	s := 1 / math.Sqrt(float64(c.qkHeadDim()))
	if r := c.RopeScaling; r != nil && r.Type == "yarn" && r.MScaleAllDim != 0 {
		m := yarnMScale(r.Factor, r.MScaleAllDim)
		s *= m * m
	}
	return s
}

type deepseekBlock struct {
	InNorm, PostNorm  *nn.RMSNorm
	QProj             *nn.Linear // nil when QLoraRank > 0
	QA, QB            *nn.Linear
	QANorm            *nn.RMSNorm
	KVA, KVB, OProj   *nn.Linear
	KVANorm           *nn.RMSNorm
	Gate, Up, Down    *nn.Linear    // dense MLP; nil on MoE layers
	MoE               *nn.SparseMoE // routed experts; nil on dense layers
	SGate, SUp, SDown *nn.Linear    // shared experts (one MLP of NumShared · MoEIntermediate)
}

// DeepseekV2 is the DeepSeek-V2 text model; it implements CausalLM. Its
// KVCache keeps the normed KV latent in K and the rotated shared rope key in V.
type DeepseekV2 struct {
	Cfg       DeepseekV2Config
	TokenEmb  *nn.Embedding
	Blocks    []*deepseekBlock
	FinalNorm *nn.RMSNorm
	LMHead    *nn.Linear

	invFreq *tensor.Tensor // (QKRopeDim/2,) F32
	masks   map[[2]int]*tensor.Tensor
	maskDev device.Device
}

// NewDeepseekV2 allocates an uninitialised model; loaders install weights.
func NewDeepseekV2(cfg DeepseekV2Config, dev device.Device) (*DeepseekV2, error) {
	if cfg.QKRopeDim%2 != 0 {
		return nil, fmt.Errorf("deepseek_v2: qk_rope_head_dim %d must be even", cfg.QKRopeDim)
	}
	if cfg.WeightDType != dtype.BF16 {
		cfg.WeightDType = dtype.F32
	}
	act := cfg.activationDType()
	D, H := cfg.HiddenSize, cfg.NumHeads
	emb, err := nn.NewEmbeddingUninit(cfg.VocabSize, D, act, dev)
	if err != nil {
		return nil, err
	}
	lin := func(in, out int) (*nn.Linear, error) { return nn.NewLinearUninit(in, out, false, act, dev) }
	norm := func(n int) (*nn.RMSNorm, error) { return nn.NewRMSNormWithDType(n, cfg.RmsEps, act, dev) }
	blocks := make([]*deepseekBlock, cfg.NumLayers)
	for i := range blocks {
		b := &deepseekBlock{}
		var errs []error
		push := func(err error) {
			if err != nil {
				errs = append(errs, err)
			}
		}
		b.InNorm, err = norm(D)
		push(err)
		b.PostNorm, err = norm(D)
		push(err)
		if cfg.QLoraRank > 0 {
			b.QA, err = lin(D, cfg.QLoraRank)
			push(err)
			b.QANorm, err = norm(cfg.QLoraRank)
			push(err)
			b.QB, err = lin(cfg.QLoraRank, H*cfg.qkHeadDim())
			push(err)
		} else {
			b.QProj, err = lin(D, H*cfg.qkHeadDim())
			push(err)
		}
		b.KVA, err = lin(D, cfg.KVLoraRank+cfg.QKRopeDim)
		push(err)
		b.KVANorm, err = norm(cfg.KVLoraRank)
		push(err)
		b.KVB, err = lin(cfg.KVLoraRank, H*(cfg.QKNopeDim+cfg.VHeadDim))
		push(err)
		b.OProj, err = lin(H*cfg.VHeadDim, D)
		push(err)
		if cfg.isMoE(i) {
			// Experts and the (F32) router are installed by the loader.
			b.MoE, err = nn.NewSparseMoELazy(D, cfg.NumExperts, cfg.TopK, cfg.NormTopK, dev)
			push(err)
			if cfg.NumShared > 0 {
				si := cfg.NumShared * cfg.MoEIntermediate
				b.SGate, err = lin(D, si)
				push(err)
				b.SUp, err = lin(D, si)
				push(err)
				b.SDown, err = lin(si, D)
				push(err)
			}
		} else {
			b.Gate, err = lin(D, cfg.IntermediateSize)
			push(err)
			b.Up, err = lin(D, cfg.IntermediateSize)
			push(err)
			b.Down, err = lin(cfg.IntermediateSize, D)
			push(err)
		}
		if len(errs) > 0 {
			return nil, fmt.Errorf("deepseek_v2 layer %d: %w", i, errs[0])
		}
		blocks[i] = b
	}
	fn, err := norm(D)
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
	var invFreq []float32
	if cfg.RopeScaling != nil {
		if cfg.RopeScaling.Type != "yarn" {
			return nil, fmt.Errorf("deepseek_v2: rope scaling %q not supported (yarn)", cfg.RopeScaling.Type)
		}
		invFreq = YarnInvFreq(cfg.QKRopeDim, cfg.RopeBase, *cfg.RopeScaling)
	} else {
		invFreq = make([]float32, cfg.QKRopeDim/2)
		for i := range invFreq {
			invFreq[i] = float32(math.Pow(cfg.RopeBase, -2*float64(i)/float64(cfg.QKRopeDim)))
		}
	}
	inv, err := tensor.New(invFreq, []int{len(invFreq)}, dev)
	if err != nil {
		return nil, err
	}
	return &DeepseekV2{Cfg: cfg, TokenEmb: emb, Blocks: blocks, FinalNorm: fn, LMHead: head, invFreq: inv}, nil
}

func (m *DeepseekV2) NumLayers() int     { return m.Cfg.NumLayers }
func (m *DeepseekV2) VocabSize() int     { return m.Cfg.VocabSize }
func (m *DeepseekV2) MaxContext() int    { return m.Cfg.MaxPosition }
func (m *DeepseekV2) SlidingWindow() int { return 0 }

// Parameters lists every weight once.
func (m *DeepseekV2) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokenEmb.Weight}
	for _, b := range m.Blocks {
		for _, n := range []*nn.RMSNorm{b.InNorm, b.PostNorm, b.QANorm, b.KVANorm} {
			if n != nil {
				ps = append(ps, n.Weight)
			}
		}
		for _, l := range []*nn.Linear{b.QProj, b.QA, b.QB, b.KVA, b.KVB, b.OProj, b.Gate, b.Up, b.Down, b.SGate, b.SUp, b.SDown} {
			if l != nil {
				ps = append(ps, l.W)
			}
		}
		if b.MoE != nil {
			ps = append(ps, b.MoE.Gate.W, b.MoE.GateUp, b.MoE.Down)
		}
	}
	ps = append(ps, m.FinalNorm.Weight)
	if !m.Cfg.TieEmbed {
		ps = append(ps, m.LMHead.W)
	}
	return ps
}

func (m *DeepseekV2) ParamCount() int {
	n := 0
	for _, p := range m.Parameters() {
		n += p.NumElements()
	}
	return n
}

// ForwardWithCache runs tokens (B, T) I64 and returns F32 logits (B, T, V).
func (m *DeepseekV2) ForwardWithCache(tokens *tensor.Tensor, cache *KVCache) (*tensor.Tensor, error) {
	if tokens.Rank() != 2 || tokens.DType() != dtype.I64 {
		return nil, fmt.Errorf("DeepseekV2.Forward: tokens must be (B, T) I64, got %v %v", tokens.Shape(), tokens.DType())
	}
	B, T := tokens.Dim(0), tokens.Dim(1)
	dev := tokens.Device()
	prior := 0
	if cache != nil {
		if len(cache.K) != m.Cfg.NumLayers {
			return nil, fmt.Errorf("DeepseekV2: cache has %d layers, model %d", len(cache.K), m.Cfg.NumLayers)
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

func (m *DeepseekV2) runBlock(blk *deepseekBlock, x *tensor.Tensor, B, T, prior int, pos *tensor.Tensor, cache *KVCache, layer int) (*tensor.Tensor, error) {
	h, err := blk.InNorm.Forward(x)
	if err != nil {
		return nil, err
	}
	a, err := m.attention(blk, h, B, T, prior, pos, cache, layer)
	if err != nil {
		return nil, err
	}
	if x, err = ops.Add(x, a); err != nil {
		return nil, err
	}
	if h, err = blk.PostNorm.Forward(x); err != nil {
		return nil, err
	}
	var f *tensor.Tensor
	if blk.MoE != nil {
		if f, err = blk.MoE.Forward(h); err != nil {
			return nil, err
		}
		if s := m.Cfg.RoutedScale; s != 0 && s != 1 {
			if f, err = mulScalarF32(f, s); err != nil {
				return nil, err
			}
		}
		if blk.SGate != nil {
			sh, err := swiGLU(blk.SGate, blk.SUp, blk.SDown, h)
			if err != nil {
				return nil, err
			}
			if f, err = ops.Add(f, sh); err != nil {
				return nil, err
			}
		}
	} else if f, err = swiGLU(blk.Gate, blk.Up, blk.Down, h); err != nil {
		return nil, err
	}
	return ops.Add(x, f)
}

// swiGLU is down(silu(gate(h)) ⊙ up(h)).
func swiGLU(gate, up, down *nn.Linear, h *tensor.Tensor) (*tensor.Tensor, error) {
	g, err := gate.Forward(h)
	if err != nil {
		return nil, err
	}
	if g, err = ops.SiLU(g); err != nil {
		return nil, err
	}
	u, err := up.Forward(h)
	if err != nil {
		return nil, err
	}
	if g, err = ops.Mul(g, u); err != nil {
		return nil, err
	}
	return down.Forward(g)
}

// attention is MLA in its expanded form: the cached latent is projected to
// per-head keys and values at every step (the "absorbed" form, which folds
// kv_b_proj into the query and output projections, is a later optimisation).
func (m *DeepseekV2) attention(blk *deepseekBlock, h *tensor.Tensor, B, T, prior int, pos *tensor.Tensor, cache *KVCache, layer int) (*tensor.Tensor, error) {
	cfg := m.Cfg
	H, dn, dr, dv, R := cfg.NumHeads, cfg.QKNopeDim, cfg.QKRopeDim, cfg.VHeadDim, cfg.KVLoraRank
	dq := dn + dr

	// Queries: (B, T, H·dq) → (B, H, T, dq), split into no-RoPE / RoPE parts.
	var q *tensor.Tensor
	var err error
	if blk.QProj != nil {
		q, err = blk.QProj.Forward(h)
	} else {
		var qa *tensor.Tensor
		if qa, err = blk.QA.Forward(h); err == nil {
			if qa, err = blk.QANorm.Forward(qa); err == nil {
				q, err = blk.QB.Forward(qa)
			}
		}
	}
	if err != nil {
		return nil, err
	}
	if q, err = headsForward(q, B, T, H, dq); err != nil {
		return nil, err
	}
	qNope, err := narrowContig(q, 3, 0, dn)
	if err != nil {
		return nil, err
	}
	qPe, err := narrowContig(q, 3, dn, dr)
	if err != nil {
		return nil, err
	}

	// Latent + shared rope key.
	ckv, err := blk.KVA.Forward(h) // (B, T, R + dr)
	if err != nil {
		return nil, err
	}
	latent, err := narrowContig(ckv, 2, 0, R)
	if err != nil {
		return nil, err
	}
	if latent, err = blk.KVANorm.Forward(latent); err != nil {
		return nil, err
	}
	if latent, err = autograd.Reshape(latent, B, 1, T, R); err != nil {
		return nil, err
	}
	kPe, err := narrowContig(ckv, 2, R, dr)
	if err != nil {
		return nil, err
	}
	if kPe, err = autograd.Reshape(kPe, B, 1, T, dr); err != nil {
		return nil, err
	}
	// RoPE on the rope parts. The loader reorders the checkpoint's
	// interleaved (complex-pair) rope dims to rotate-half order in q and in
	// the shared key alike, so the ordinary kernel gives the same scores.
	if qPe, err = ops.RoPEWithInvFreq(qPe, pos, m.invFreq); err != nil {
		return nil, err
	}
	if kPe, err = ops.RoPEWithInvFreq(kPe, pos, m.invFreq); err != nil {
		return nil, err
	}
	if rs := cfg.RopeScaling; rs != nil {
		if af := rs.AttentionFactor(); af != 1 { // YaRN cos / sin multiplier, on both rope parts
			if qPe, err = mulScalarF32(qPe, af); err != nil {
				return nil, err
			}
			if kPe, err = mulScalarF32(kPe, af); err != nil {
				return nil, err
			}
		}
	}

	// Cache the compressed form, then expand every cached position.
	if cache != nil {
		if cache.K[layer] != nil {
			if latent, err = ops.Concat([]*tensor.Tensor{cache.K[layer], latent}, 2); err != nil {
				return nil, err
			}
			if kPe, err = ops.Concat([]*tensor.Tensor{cache.V[layer], kPe}, 2); err != nil {
				return nil, err
			}
		}
		cache.K[layer], cache.V[layer] = latent, kPe
	}
	Tt := latent.Dim(2)
	lat, err := autograd.Reshape(latent, B, Tt, R)
	if err != nil {
		return nil, err
	}
	kv, err := blk.KVB.Forward(lat) // (B, Tt, H·(dn+dv))
	if err != nil {
		return nil, err
	}
	if kv, err = headsForward(kv, B, Tt, H, dn+dv); err != nil {
		return nil, err
	}
	kNope, err := narrowContig(kv, 3, 0, dn)
	if err != nil {
		return nil, err
	}
	v, err := narrowContig(kv, 3, dn, dv)
	if err != nil {
		return nil, err
	}
	kPeH, err := repeatKV(kPe, H, B, 1, Tt, dr)
	if err != nil {
		return nil, err
	}
	k, err := ops.Concat([]*tensor.Tensor{kNope, kPeH}, 3)
	if err != nil {
		return nil, err
	}
	if q, err = ops.Concat([]*tensor.Tensor{qNope, qPe}, 3); err != nil {
		return nil, err
	}

	// Scores, mask, softmax, values.
	if q, err = autograd.Reshape(q, B*H, T, dq); err != nil {
		return nil, err
	}
	if k, err = autograd.Reshape(k, B*H, Tt, dq); err != nil {
		return nil, err
	}
	if v, err = autograd.Reshape(v, B*H, Tt, dv); err != nil {
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
	// The scale is ~0.11 for V2 (192^-½ · mscale²): multiplied in F32 as
	// transformers' python-float scaling does.
	if scores, err = mulScalarF32(scores, cfg.softmaxScale()); err != nil {
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
	ctx, err := autograd.MatMul(attn, v) // (B·H, T, dv)
	if err != nil {
		return nil, err
	}
	if ctx, err = autograd.Reshape(ctx, B, H, T, dv); err != nil {
		return nil, err
	}
	if ctx, err = autograd.Permute(ctx, 0, 2, 1, 3); err != nil {
		return nil, err
	}
	if ctx, err = autograd.Contiguous(ctx); err != nil {
		return nil, err
	}
	if ctx, err = autograd.Reshape(ctx, B, T, H*dv); err != nil {
		return nil, err
	}
	return blk.OProj.Forward(ctx)
}

// causalMask returns the cached (T, prior+T) additive causal mask.
func (m *DeepseekV2) causalMask(T, prior int, dt dtype.DType, dev device.Device) (*tensor.Tensor, error) {
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

var _ CausalLM = (*DeepseekV2)(nil)
