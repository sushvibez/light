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

// EmbeddingGemma2Config describes the EmbeddingGemma 2 text encoder
// (transformers modeling_embedding_gemma2.py): a Gemma 4 style stack run
// bidirectionally. Every layer has projection-only per-layer embeddings
// (no PLE token table), full-attention layers use GlobalHeadDim heads with a
// single KV head and full-width RoPE, sliding layers attend within a radius
// of SlidingWindow positions in both directions, and the final norm is
// followed by a linear projection to EmbeddingDim. Callers mean-pool and
// L2-normalise the token embeddings (sentence-transformers 1_Pooling +
// 2_Normalize).
type EmbeddingGemma2Config struct {
	VocabSize        int
	HiddenSize       int
	IntermediateSize int
	NumLayers        int
	NumHeads         int
	NumKVHeads       int // sliding layers
	NumGlobalKVHeads int // full-attention layers
	HeadDim          int // sliding layers
	GlobalHeadDim    int // full-attention layers
	MaxPosition      int
	RopeBase         float64 // full-attention layers
	RopeLocalBase    float64 // sliding layers
	RmsEps           float64
	SlidingWindow    int    // radius: a sliding layer sees |i-j| <= SlidingWindow
	SlidingLayers    []bool // per layer
	PLEDim           int    // hidden_size_per_layer_input
	EmbeddingDim     int    // output dimension (768)
	WeightDType      dtype.DType
}

func (cfg EmbeddingGemma2Config) activationDType() dtype.DType {
	if cfg.WeightDType == dtype.BF16 {
		return dtype.BF16
	}
	return dtype.F32
}

func (cfg EmbeddingGemma2Config) isSliding(layer int) bool {
	return layer < len(cfg.SlidingLayers) && cfg.SlidingLayers[layer]
}

// EmbeddingGemma2 is the EmbeddingGemma 2 text encoder. Weights are
// installed by LoadEmbeddingGemma2FromHF.
type EmbeddingGemma2 struct {
	Cfg       EmbeddingGemma2Config
	TokenEmb  *nn.Embedding
	PLEProj   *nn.Linear  // ple.per_layer_model_projection (D → L·P), scale D^-½ folded in
	PLENorm   *nn.RMSNorm // ple.per_layer_projection_norm over P
	Blocks    []*gemma4Block
	FinalNorm *nn.RMSNorm
	OutProj   *nn.Linear // embedding_projection (D → EmbeddingDim)

	embScale *tensor.Tensor // √D rounded to the activation dtype
	tabs     gemma4Tables
}

// NewEmbeddingGemma2 allocates an uninitialised model.
func NewEmbeddingGemma2(cfg EmbeddingGemma2Config, dev device.Device) (*EmbeddingGemma2, error) {
	if cfg.WeightDType != dtype.BF16 {
		cfg.WeightDType = dtype.F32
	}
	if cfg.GlobalHeadDim == 0 {
		cfg.GlobalHeadDim = cfg.HeadDim
	}
	if cfg.NumGlobalKVHeads == 0 {
		cfg.NumGlobalKVHeads = cfg.NumKVHeads
	}
	if len(cfg.SlidingLayers) != cfg.NumLayers {
		return nil, fmt.Errorf("embeddinggemma2: SlidingLayers has %d entries for %d layers", len(cfg.SlidingLayers), cfg.NumLayers)
	}
	if cfg.PLEDim <= 0 || cfg.EmbeddingDim <= 0 {
		return nil, fmt.Errorf("embeddinggemma2: PLEDim and EmbeddingDim must be set")
	}
	act := cfg.activationDType()
	D, NH, IM, P := cfg.HiddenSize, cfg.NumHeads, cfg.IntermediateSize, cfg.PLEDim
	emb, err := nn.NewEmbeddingUninit(cfg.VocabSize, D, act, dev)
	if err != nil {
		return nil, err
	}
	norm := func(n int) (*nn.RMSNorm, error) { return nn.NewRMSNormWithDType(n, cfg.RmsEps, act, dev) }
	lin := func(in, out int) (*nn.Linear, error) { return nn.NewLinearUninit(in, out, false, act, dev) }
	m := &EmbeddingGemma2{Cfg: cfg, TokenEmb: emb}
	if m.PLEProj, err = lin(D, cfg.NumLayers*P); err != nil {
		return nil, err
	}
	if m.PLENorm, err = norm(P); err != nil {
		return nil, err
	}
	m.Blocks = make([]*gemma4Block, cfg.NumLayers)
	for i := range m.Blocks {
		b := &gemma4Block{sliding: cfg.isSliding(i), headDim: cfg.HeadDim, kvHeads: cfg.NumKVHeads}
		if !b.sliding {
			b.headDim, b.kvHeads = cfg.GlobalHeadDim, cfg.NumGlobalKVHeads
		}
		if NH%b.kvHeads != 0 {
			return nil, fmt.Errorf("embeddinggemma2 layer %d: NumHeads %d not divisible by kv heads %d", i, NH, b.kvHeads)
		}
		var errs []error
		push := func(e error) {
			if e != nil {
				errs = append(errs, e)
			}
		}
		b.InNorm, err = norm(D)
		push(err)
		b.PostAttnNorm, err = norm(D)
		push(err)
		b.PreFFNorm, err = norm(D)
		push(err)
		b.PostFFNorm, err = norm(D)
		push(err)
		b.QNorm, err = norm(b.headDim)
		push(err)
		b.KNorm, err = norm(b.headDim)
		push(err)
		b.QProj, err = lin(D, NH*b.headDim)
		push(err)
		b.KProj, err = lin(D, b.kvHeads*b.headDim)
		push(err)
		b.VProj, err = lin(D, b.kvHeads*b.headDim)
		push(err)
		b.OProj, err = lin(NH*b.headDim, D)
		push(err)
		b.GateProj, err = lin(D, IM)
		push(err)
		b.UpProj, err = lin(D, IM)
		push(err)
		b.DownProj, err = lin(IM, D)
		push(err)
		b.PLEGate, err = lin(D, P)
		push(err)
		b.PLEProj, err = lin(P, D)
		push(err)
		b.PostPLENorm, err = norm(D)
		push(err)
		b.LayerScalar, err = tensor.Zeros(act, []int{1}, dev)
		push(err)
		if len(errs) > 0 {
			return nil, fmt.Errorf("embeddinggemma2 layer %d: %w", i, errs[0])
		}
		m.Blocks[i] = b
	}
	if m.FinalNorm, err = norm(D); err != nil {
		return nil, err
	}
	if m.OutProj, err = lin(D, cfg.EmbeddingDim); err != nil {
		return nil, err
	}
	s, err := tensor.New([]float32{float32(math.Sqrt(float64(D)))}, []int{1}, dev)
	if err != nil {
		return nil, err
	}
	if act != dtype.F32 {
		if s, err = ops.Cast(s, act); err != nil {
			return nil, err
		}
	}
	m.embScale = s
	return m, nil
}

func (m *EmbeddingGemma2) NumLayers() int    { return m.Cfg.NumLayers }
func (m *EmbeddingGemma2) EmbeddingDim() int { return m.Cfg.EmbeddingDim }
func (m *EmbeddingGemma2) MaxContext() int   { return m.Cfg.MaxPosition }

// Parameters lists every weight once.
func (m *EmbeddingGemma2) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokenEmb.Weight, m.PLEProj.W, m.PLENorm.Weight}
	for _, b := range m.Blocks {
		for _, n := range []*nn.RMSNorm{b.InNorm, b.PostAttnNorm, b.PreFFNorm, b.PostFFNorm, b.QNorm, b.KNorm, b.PostPLENorm} {
			ps = append(ps, n.Weight)
		}
		for _, l := range []*nn.Linear{b.QProj, b.KProj, b.VProj, b.OProj, b.GateProj, b.UpProj, b.DownProj, b.PLEGate, b.PLEProj} {
			ps = append(ps, l.W)
		}
		ps = append(ps, b.LayerScalar)
	}
	return append(ps, m.FinalNorm.Weight, m.OutProj.W)
}

func (m *EmbeddingGemma2) ParamCount() int {
	n := 0
	for _, p := range m.Parameters() {
		n += p.NumElements()
	}
	return n
}

// Forward runs tokens (B, T) I64 through the encoder and returns the
// projected token embeddings (B, T, EmbeddingDim) in the activation dtype.
// mask (B, T), nil for unpadded input, marks real tokens with 1 (I64 or
// F32); padded keys are excluded from attention.
func (m *EmbeddingGemma2) Forward(tokens, mask *tensor.Tensor) (*tensor.Tensor, error) {
	if tokens.Rank() != 2 || tokens.DType() != dtype.I64 {
		return nil, fmt.Errorf("EmbeddingGemma2.Forward: tokens must be (B, T) I64, got %v %v", tokens.Shape(), tokens.DType())
	}
	B, T := tokens.Dim(0), tokens.Dim(1)
	if T > m.Cfg.MaxPosition {
		return nil, fmt.Errorf("EmbeddingGemma2.Forward: sequence length %d > MaxPosition %d", T, m.Cfg.MaxPosition)
	}
	dev := tokens.Device()
	h, err := m.TokenEmb.Forward(tokens)
	if err != nil {
		return nil, err
	}
	if h, err = ops.Mul(h, m.embScale); err != nil {
		return nil, err
	}
	// Projection-only PLE: (L, B, T, P) = norm(proj(h)·D^-½).
	L, P := m.Cfg.NumLayers, m.Cfg.PLEDim
	ple, err := m.PLEProj.Forward(h)
	if err != nil {
		return nil, err
	}
	if ple, err = autograd.Reshape(ple, B, T, L, P); err != nil {
		return nil, err
	}
	if ple, err = m.PLENorm.Forward(ple); err != nil {
		return nil, err
	}
	if ple, err = autograd.Permute(ple, 2, 0, 1, 3); err != nil {
		return nil, err
	}
	if ple, err = autograd.Contiguous(ple); err != nil {
		return nil, err
	}
	posData := make([]int64, T)
	for i := range posData {
		posData[i] = int64(i)
	}
	pos, err := tensor.New(posData, []int{T}, dev)
	if err != nil {
		return nil, err
	}
	var padMask *tensor.Tensor // (B, 1, 1, T) additive
	if mask != nil {
		if padMask, err = m.padMask(mask, h.DType()); err != nil {
			return nil, err
		}
	}
	for i, blk := range m.Blocks {
		pleI, err := ple.Narrow(0, i, 1)
		if err != nil {
			return nil, err
		}
		if pleI, err = autograd.Reshape(pleI, B, T, P); err != nil {
			return nil, err
		}
		if h, err = m.runBlock(blk, h, pleI, padMask, B, T, pos, dev); err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
	}
	if h, err = m.FinalNorm.Forward(h); err != nil {
		return nil, err
	}
	return m.OutProj.Forward(h)
}

// Embed returns L2-normalised mean-pooled sentence embeddings (B,
// EmbeddingDim) as F32: the full sentence-transformers pipeline.
func (m *EmbeddingGemma2) Embed(tokens, mask *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := m.Forward(tokens, mask)
	if err != nil {
		return nil, err
	}
	if h.DType() != dtype.F32 {
		if h, err = ops.Cast(h, dtype.F32); err != nil {
			return nil, err
		}
	}
	pooled, err := MeanPooled(h, mask)
	if err != nil {
		return nil, err
	}
	return Normalize(pooled)
}

// padMask turns a (B, T) keep-mask into an additive (B, 1, 1, T) mask.
func (m *EmbeddingGemma2) padMask(mask *tensor.Tensor, dt dtype.DType) (*tensor.Tensor, error) {
	if mask.Rank() != 2 {
		return nil, fmt.Errorf("EmbeddingGemma2: mask must be (B, T), got %v", mask.Shape())
	}
	c, err := mask.To(device.CPU())
	if err != nil {
		return nil, err
	}
	if c.DType() != dtype.F32 {
		if c, err = ops.Cast(c, dtype.F32); err != nil {
			return nil, err
		}
	}
	if c, err = c.Contiguous(); err != nil {
		return nil, err
	}
	keep := tensor.MustSlice[float32](c)
	data := make([]float32, len(keep))
	for i, k := range keep {
		if k == 0 {
			data[i] = float32(math.Inf(-1))
		}
	}
	t, err := tensor.New(data, []int{mask.Dim(0), 1, 1, mask.Dim(1)}, mask.Device())
	if err != nil {
		return nil, err
	}
	if dt != dtype.F32 {
		return ops.Cast(t, dt)
	}
	return t, nil
}

func (m *EmbeddingGemma2) runBlock(blk *gemma4Block, x, pleI, padMask *tensor.Tensor, B, T int, pos *tensor.Tensor, dev device.Device) (*tensor.Tensor, error) {
	cfg := m.Cfg
	NH, KH, HD := cfg.NumHeads, blk.kvHeads, blk.headDim
	g := NH / KH
	base, window := cfg.RopeBase, 0
	if blk.sliding {
		base, window = cfg.RopeLocalBase, cfg.SlidingWindow
	}

	h, err := blk.InNorm.Forward(x)
	if err != nil {
		return nil, err
	}
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
	if v, err = m.tabs.vNorm(v, HD, cfg.RmsEps, dev); err != nil {
		return nil, err
	}
	if q, err = m.tabs.rope(q, pos, blk.sliding, HD, base, 1, dev); err != nil {
		return nil, err
	}
	if k, err = m.tabs.rope(k, pos, blk.sliding, HD, base, 1, dev); err != nil {
		return nil, err
	}
	if k, err = repeatKV(k, g, B, KH, T, HD); err != nil {
		return nil, err
	}
	if v, err = repeatKV(v, g, B, KH, T, HD); err != nil {
		return nil, err
	}
	if q, err = autograd.Reshape(q, B*NH, T, HD); err != nil {
		return nil, err
	}
	if k, err = autograd.Reshape(k, B*NH, T, HD); err != nil {
		return nil, err
	}
	if v, err = autograd.Reshape(v, B*NH, T, HD); err != nil {
		return nil, err
	}
	kT, err := autograd.Transpose(k, 1, 2)
	if err != nil {
		return nil, err
	}
	scores, err := autograd.MatMul(q, kT) // scaling = 1: q_norm carries the scale
	if err != nil {
		return nil, err
	}
	if window > 0 && T > window+1 {
		mask, err := m.tabs.mask(T, 0, window, true, scores.DType(), dev)
		if err != nil {
			return nil, err
		}
		if scores, err = ops.Add(scores, mask); err != nil {
			return nil, err
		}
	}
	if padMask != nil {
		if scores, err = autograd.Reshape(scores, B, NH, T, T); err != nil {
			return nil, err
		}
		if scores, err = ops.Add(scores, padMask); err != nil {
			return nil, err
		}
		if scores, err = autograd.Reshape(scores, B*NH, T, T); err != nil {
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
	o, err := blk.OProj.Forward(ctx)
	if err != nil {
		return nil, err
	}
	if o, err = blk.PostAttnNorm.Forward(o); err != nil {
		return nil, err
	}
	if x, err = ops.Add(x, o); err != nil {
		return nil, err
	}
	return blk.ffn(x, pleI)
}
