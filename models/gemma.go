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

// Gemma3Config describes a Gemma 3 text decoder (Gemma3ForCausalLM). It
// differs from Llama in: embeddings scaled by √D, RMSNorm with (1+w) gain,
// four norms per block (pre/post attention, pre/post MLP), QK-norm over the
// head dim, GELU-tanh gated MLP, per-layer local (sliding window, RoPE base
// 10k) / global (RoPE base 1M) attention in a 5:1 pattern, query scaling by
// query_pre_attn_scalar^-½ and optional logit soft-capping (Gemma 2).
type Gemma3Config struct {
	VocabSize          int
	HiddenSize         int
	IntermediateSize   int
	NumLayers          int
	NumHeads           int
	NumKVHeads         int
	HeadDim            int
	MaxPosition        int
	RopeBase           float64 // global layers (1e6)
	RopeLocalBase      float64 // sliding layers (1e4)
	RopeScale          float64 // linear rope_scaling factor on global layers (Gemma 3 4B+: 8); ≤1 = none
	RmsEps             float64
	QueryPreAttnScalar float64 // attention scale = QueryPreAttnScalar^-½ (HeadDim when 0)
	SlidingWindow      int
	SlidingLayers      []bool  // per layer; nil = all local when SlidingWindow > 0
	FinalLogitSoftcap  float64 // 0 = none (Gemma 3); 30 for Gemma 2
	AttnLogitSoftcap   float64 // 0 = none (Gemma 3); 50 for Gemma 2
	NoQKNorm           bool    // Gemma 2 has no q_norm / k_norm (Gemma 3 does)
	WeightDType        dtype.DType
}

type gemmaBlock struct {
	InNorm, PostAttnNorm, PreFFNorm, PostFFNorm *nn.RMSNorm // F32 weights holding (1+w)
	QNorm, KNorm                                *nn.RMSNorm // over HeadDim
	QProj, KProj, VProj, OProj                  *nn.Linear
	GateProj, UpProj, DownProj                  *nn.Linear
	sliding                                     bool
}

// Gemma3 is a Gemma 3 text model. Weights are installed by LoadGemma3FromHF.
type Gemma3 struct {
	Cfg       Gemma3Config
	TokenEmb  *nn.Embedding
	Blocks    []*gemmaBlock
	FinalNorm *nn.RMSNorm
	LMHead    *nn.Linear // tied to TokenEmb

	embScale  *tensor.Tensor // (1,) √D rounded to the activation dtype
	attnScale *tensor.Tensor // (1,) QueryPreAttnScalar^-½
	masks     map[[3]int]*tensor.Tensor
	masksDev  device.Device
	invFreq   *tensor.Tensor // (HeadDim/2,) global-layer table with RopeScale applied; nil when RopeScale ≤ 1
}

func (cfg Gemma3Config) activationDType() dtype.DType {
	if cfg.WeightDType == dtype.BF16 {
		return dtype.BF16
	}
	return dtype.F32
}

func (cfg Gemma3Config) isSliding(layer int) bool {
	if cfg.SlidingWindow <= 0 {
		return false
	}
	if cfg.SlidingLayers == nil {
		return true
	}
	return layer < len(cfg.SlidingLayers) && cfg.SlidingLayers[layer]
}

// NewGemma3 allocates an uninitialised model (loaders fill the weights).
func NewGemma3(cfg Gemma3Config, dev device.Device) (*Gemma3, error) {
	if cfg.HeadDim%2 != 0 {
		return nil, fmt.Errorf("gemma3: HeadDim must be even (got %d)", cfg.HeadDim)
	}
	if cfg.NumKVHeads <= 0 || cfg.NumHeads%cfg.NumKVHeads != 0 {
		return nil, fmt.Errorf("gemma3: NumHeads %d not divisible by NumKVHeads %d", cfg.NumHeads, cfg.NumKVHeads)
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
	norm := func(n int) (*nn.RMSNorm, error) { return nn.NewRMSNormWithDType(n, cfg.RmsEps, dtype.F32, dev) }
	lin := func(in, out int) (*nn.Linear, error) { return nn.NewLinearUninit(in, out, false, act, dev) }
	blocks := make([]*gemmaBlock, cfg.NumLayers)
	for i := range blocks {
		b := &gemmaBlock{sliding: cfg.isSliding(i)}
		var errs []error
		push := func(err error) {
			if err != nil {
				errs = append(errs, err)
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
		if !cfg.NoQKNorm {
			b.QNorm, err = norm(HD)
			push(err)
			b.KNorm, err = norm(HD)
			push(err)
		}
		b.QProj, err = lin(D, NH*HD)
		push(err)
		b.KProj, err = lin(D, KH*HD)
		push(err)
		b.VProj, err = lin(D, KH*HD)
		push(err)
		b.OProj, err = lin(NH*HD, D)
		push(err)
		b.GateProj, err = lin(D, IM)
		push(err)
		b.UpProj, err = lin(D, IM)
		push(err)
		b.DownProj, err = lin(IM, D)
		push(err)
		if len(errs) > 0 {
			return nil, fmt.Errorf("gemma3 layer %d: %w", i, errs[0])
		}
		blocks[i] = b
	}
	fn, err := norm(D)
	if err != nil {
		return nil, err
	}
	head, err := nn.NewLinearTied(emb.Weight, false, dev)
	if err != nil {
		return nil, err
	}
	scalar := func(v float64) (*tensor.Tensor, error) {
		t, err := tensor.New([]float32{float32(v)}, []int{1}, dev)
		if err != nil {
			return nil, err
		}
		if act != dtype.F32 {
			return ops.Cast(t, act) // rounds exactly as HF's `.to(weight.dtype)` does
		}
		return t, nil
	}
	embScale, err := scalar(math.Sqrt(float64(D)))
	if err != nil {
		return nil, err
	}
	qs := cfg.QueryPreAttnScalar
	if qs == 0 {
		qs = float64(HD)
	}
	attnScale, err := scalar(1 / math.Sqrt(qs))
	if err != nil {
		return nil, err
	}
	return &Gemma3{Cfg: cfg, TokenEmb: emb, Blocks: blocks, FinalNorm: fn, LMHead: head,
		embScale: embScale, attnScale: attnScale}, nil
}

func (m *Gemma3) NumLayers() int     { return m.Cfg.NumLayers }
func (m *Gemma3) VocabSize() int     { return m.Cfg.VocabSize }
func (m *Gemma3) MaxContext() int    { return m.Cfg.MaxPosition }
func (m *Gemma3) SlidingWindow() int { return m.Cfg.SlidingWindow }

// Parameters lists every weight once (the LM head is tied).
func (m *Gemma3) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokenEmb.Weight}
	for _, b := range m.Blocks {
		for _, n := range []*nn.RMSNorm{b.InNorm, b.PostAttnNorm, b.PreFFNorm, b.PostFFNorm, b.QNorm, b.KNorm} {
			if n != nil {
				ps = append(ps, n.Weight)
			}
		}
		for _, l := range []*nn.Linear{b.QProj, b.KProj, b.VProj, b.OProj, b.GateProj, b.UpProj, b.DownProj} {
			ps = append(ps, l.W)
		}
	}
	return append(ps, m.FinalNorm.Weight)
}

func (m *Gemma3) ParamCount() int {
	n := 0
	for _, p := range m.Parameters() {
		n += p.NumElements()
	}
	return n
}

// Forward is a stateless ForwardWithCache.
func (m *Gemma3) Forward(tokens *tensor.Tensor) (*tensor.Tensor, error) {
	return m.ForwardWithCache(tokens, nil)
}

// ForwardWithCache runs tokens (B, T) I64 and returns F32 logits (B, T, V).
func (m *Gemma3) ForwardWithCache(tokens *tensor.Tensor, cache *KVCache) (*tensor.Tensor, error) {
	if tokens.Rank() != 2 || tokens.DType() != dtype.I64 {
		return nil, fmt.Errorf("Gemma3.Forward: tokens must be (B, T) I64, got %v %v", tokens.Shape(), tokens.DType())
	}
	B, T := tokens.Dim(0), tokens.Dim(1)
	dev := tokens.Device()
	prior := 0
	if cache != nil {
		if len(cache.K) != m.Cfg.NumLayers {
			return nil, fmt.Errorf("Gemma3.ForwardWithCache: cache has %d layers, model %d", len(cache.K), m.Cfg.NumLayers)
		}
		prior = cache.Len
	}
	if prior+T > m.Cfg.MaxPosition {
		return nil, fmt.Errorf("Gemma3.Forward: sequence length %d > MaxPosition %d", prior+T, m.Cfg.MaxPosition)
	}
	h, err := m.TokenEmb.Forward(tokens)
	if err != nil {
		return nil, err
	}
	if h, err = ops.Mul(h, m.embScale); err != nil {
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
		if h, err = m.runBlock(blk, h, B, T, pos, dev, cache, i); err != nil {
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
		if logits, err = ops.Cast(logits, dtype.F32); err != nil {
			return nil, err
		}
	}
	if c := m.Cfg.FinalLogitSoftcap; c > 0 {
		if logits, err = softcap(logits, c); err != nil {
			return nil, err
		}
	}
	return logits, nil
}

// softcap returns c·tanh(x/c).
func softcap(x *tensor.Tensor, c float64) (*tensor.Tensor, error) {
	inv, err := tensor.New([]float32{float32(1 / c)}, []int{1}, x.Device())
	if err != nil {
		return nil, err
	}
	cap_, err := tensor.New([]float32{float32(c)}, []int{1}, x.Device())
	if err != nil {
		return nil, err
	}
	if x.DType() != dtype.F32 {
		if inv, err = ops.Cast(inv, x.DType()); err != nil {
			return nil, err
		}
		if cap_, err = ops.Cast(cap_, x.DType()); err != nil {
			return nil, err
		}
	}
	y, err := ops.Mul(x, inv)
	if err != nil {
		return nil, err
	}
	if y, err = ops.Tanh(y); err != nil {
		return nil, err
	}
	return ops.Mul(y, cap_)
}

func (m *Gemma3) runBlock(blk *gemmaBlock, x *tensor.Tensor, B, T int, pos *tensor.Tensor, dev device.Device, cache *KVCache, layer int) (*tensor.Tensor, error) {
	cfg := m.Cfg
	NH, KH, HD := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	g := NH / KH

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
	// QK-norm over the head dim, then RoPE with the layer's base.
	if blk.QNorm != nil {
		if q, err = blk.QNorm.Forward(q); err != nil {
			return nil, err
		}
		if k, err = blk.KNorm.Forward(k); err != nil {
			return nil, err
		}
	}
	base := cfg.RopeBase
	window := 0
	if blk.sliding {
		base = cfg.RopeLocalBase
		window = cfg.SlidingWindow
	}
	if q, err = m.rope(q, pos, base, blk.sliding); err != nil {
		return nil, err
	}
	if k, err = m.rope(k, pos, base, blk.sliding); err != nil {
		return nil, err
	}

	kvPrior := 0
	totalT := T
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
		if window > 0 && totalT > window {
			if cache.K[layer], err = trimKV(k, totalT-window, window); err != nil {
				return nil, err
			}
			if cache.V[layer], err = trimKV(v, totalT-window, window); err != nil {
				return nil, err
			}
		}
	}
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
	if c := cfg.AttnLogitSoftcap; c > 0 {
		if scores, err = softcap(scores, c); err != nil {
			return nil, err
		}
	}
	if T > 1 || (window > 0 && totalT > window) {
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

	// MLP: down(gelu_tanh(gate(h)) ⊙ up(h)), post-normed before the residual.
	if h, err = blk.PreFFNorm.Forward(x); err != nil {
		return nil, err
	}
	gate, err := blk.GateProj.Forward(h)
	if err != nil {
		return nil, err
	}
	if gate, err = ops.Gelu(gate); err != nil {
		return nil, err
	}
	up, err := blk.UpProj.Forward(h)
	if err != nil {
		return nil, err
	}
	if gate, err = ops.Mul(gate, up); err != nil {
		return nil, err
	}
	mlp, err := blk.DownProj.Forward(gate)
	if err != nil {
		return nil, err
	}
	if mlp, err = blk.PostFFNorm.Forward(mlp); err != nil {
		return nil, err
	}
	return ops.Add(x, mlp)
}

// mask returns the cached (T, prior+T) additive causal / sliding-window mask.
func (m *Gemma3) mask(T, prior, window int, dt dtype.DType, dev device.Device) (*tensor.Tensor, error) {
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
		for j := p + 1; j < total; j++ {
			data[i*total+j] = negInf
		}
		if window > 0 {
			for j := 0; j <= p-window; j++ {
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

var _ CausalLM = (*Gemma3)(nil)

// rope rotates x by position. Global layers of the 4B+ checkpoints use
// linear scaling (inv_freq / RopeScale, i.e. positions / 8); local layers
// never scale (transformers' rotary_emb_local drops rope_scaling).
func (m *Gemma3) rope(x, pos *tensor.Tensor, base float64, sliding bool) (*tensor.Tensor, error) {
	if sliding || m.Cfg.RopeScale <= 1 {
		return ops.RoPE(x, pos, base)
	}
	if m.invFreq == nil || !m.invFreq.Device().SameAs(x.Device()) {
		half := m.Cfg.HeadDim / 2
		table := make([]float32, half)
		for i := range table {
			table[i] = float32(math.Pow(base, -2*float64(i)/float64(m.Cfg.HeadDim)) / m.Cfg.RopeScale)
		}
		t, err := tensor.New(table, []int{half}, x.Device())
		if err != nil {
			return nil, err
		}
		m.invFreq = t
	}
	return ops.RoPEWithInvFreq(x, pos, m.invFreq)
}
