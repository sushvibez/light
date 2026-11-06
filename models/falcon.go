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

// FalconConfig covers every Falcon decoder variant transformers'
// modeling_falcon.py runs (Candle's falcon.rs only does the 7B shape):
//
//   - falcon-rw-1b: ALiBi, multi-head, sequential attention → MLP, biases;
//   - falcon-7b: rotary, multi-query (one K/V head), parallel attention
//     and MLP sharing one LayerNorm, no biases;
//   - falcon-40b/180b "new decoder architecture": rotary, grouped K/V
//     heads, parallel blocks with two LayerNorms (ln_attn / ln_mlp).
type FalconConfig struct {
	VocabSize      int
	HiddenSize     int
	FFNHiddenSize  int // dense_h_to_4h width (default 4·HiddenSize)
	NumLayers      int
	NumHeads       int
	NumKVHeads     int // 1 with MultiQuery, NumHeads otherwise (or num_kv_heads with NewDecoderArch)
	MaxPosition    int
	LayerNormEps   float64
	RopeBase       float64 // rotary layers (ALiBi off)
	Alibi          bool
	MultiQuery     bool
	NewDecoderArch bool
	ParallelAttn   bool
	Bias           bool
	NumLNParallel  int // 1 or 2 LayerNorms in a parallel block (2 for the new architecture)
	TieEmbed       bool
	WeightDType    dtype.DType
}

func (cfg FalconConfig) headDim() int { return cfg.HiddenSize / cfg.NumHeads }

func (cfg FalconConfig) activationDType() dtype.DType {
	if cfg.WeightDType == dtype.BF16 {
		return dtype.BF16
	}
	return dtype.F32
}

// kvHeads is the number of K/V heads the fused projection carries.
func (cfg FalconConfig) kvHeads() int {
	switch {
	case cfg.NewDecoderArch:
		return cfg.NumKVHeads
	case cfg.MultiQuery:
		return 1
	}
	return cfg.NumHeads
}

func (cfg FalconConfig) twoLN() bool { return cfg.ParallelAttn && cfg.NumLNParallel == 2 }

type falconBlock struct {
	LNAttn, LNMLP *nn.LayerNorm // input_layernorm (+ post_attention_layernorm, or ln_attn + ln_mlp)
	QKV, Dense    *nn.Linear
	Up, Down      *nn.Linear
}

// Falcon is a FalconForCausalLM. Weights are installed by LoadFalconFromHF.
type Falcon struct {
	Cfg      FalconConfig
	TokenEmb *nn.Embedding
	Blocks   []*falconBlock
	LNF      *nn.LayerNorm
	LMHead   *nn.Linear

	scale  *tensor.Tensor // 1/√headDim in the activation dtype
	masks  map[[2]int]*tensor.Tensor
	masksD device.Device
}

// NewFalcon allocates an uninitialised model.
func NewFalcon(cfg FalconConfig, dev device.Device) (*Falcon, error) {
	if cfg.WeightDType != dtype.BF16 {
		cfg.WeightDType = dtype.F32
	}
	if cfg.FFNHiddenSize == 0 {
		cfg.FFNHiddenSize = 4 * cfg.HiddenSize
	}
	if cfg.NumLNParallel == 0 {
		cfg.NumLNParallel = 1
		if cfg.NewDecoderArch {
			cfg.NumLNParallel = 2
		}
	}
	if cfg.NumKVHeads == 0 {
		cfg.NumKVHeads = cfg.NumHeads
	}
	if cfg.RopeBase == 0 {
		cfg.RopeBase = 1e4
	}
	if cfg.HiddenSize%cfg.NumHeads != 0 || cfg.NumHeads%cfg.kvHeads() != 0 {
		return nil, fmt.Errorf("falcon: hidden %d, heads %d, kv heads %d", cfg.HiddenSize, cfg.NumHeads, cfg.kvHeads())
	}
	act := cfg.activationDType()
	D, HD, KH := cfg.HiddenSize, cfg.headDim(), cfg.kvHeads()
	emb, err := nn.NewEmbeddingUninit(cfg.VocabSize, D, act, dev)
	if err != nil {
		return nil, err
	}
	ln := func() (*nn.LayerNorm, error) { return nn.NewLayerNormWithDType(D, cfg.LayerNormEps, act, dev) }
	lin := func(in, out int) (*nn.Linear, error) { return nn.NewLinearUninit(in, out, cfg.Bias, act, dev) }
	m := &Falcon{Cfg: cfg, TokenEmb: emb}
	for i := 0; i < cfg.NumLayers; i++ {
		b := &falconBlock{}
		var errs []error
		push := func(e error) {
			if e != nil {
				errs = append(errs, e)
			}
		}
		b.LNAttn, err = ln()
		push(err)
		if !cfg.ParallelAttn || cfg.twoLN() {
			b.LNMLP, err = ln()
			push(err)
		}
		b.QKV, err = lin(D, (cfg.NumHeads+2*KH)*HD)
		push(err)
		b.Dense, err = lin(D, D)
		push(err)
		b.Up, err = lin(D, cfg.FFNHiddenSize)
		push(err)
		b.Down, err = lin(cfg.FFNHiddenSize, D)
		push(err)
		if len(errs) > 0 {
			return nil, fmt.Errorf("falcon layer %d: %w", i, errs[0])
		}
		m.Blocks = append(m.Blocks, b)
	}
	if m.LNF, err = ln(); err != nil {
		return nil, err
	}
	if cfg.TieEmbed {
		m.LMHead, err = nn.NewLinearTied(emb.Weight, false, dev)
	} else {
		m.LMHead, err = nn.NewLinearUninit(D, cfg.VocabSize, false, act, dev)
	}
	if err != nil {
		return nil, err
	}
	s, err := tensor.New([]float32{float32(1 / math.Sqrt(float64(HD)))}, []int{1}, dev)
	if err != nil {
		return nil, err
	}
	if act != dtype.F32 {
		if s, err = ops.Cast(s, act); err != nil {
			return nil, err
		}
	}
	m.scale = s
	return m, nil
}

func (m *Falcon) NumLayers() int     { return m.Cfg.NumLayers }
func (m *Falcon) VocabSize() int     { return m.Cfg.VocabSize }
func (m *Falcon) MaxContext() int    { return m.Cfg.MaxPosition }
func (m *Falcon) SlidingWindow() int { return 0 }

// Parameters lists every weight once.
func (m *Falcon) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokenEmb.Weight}
	add := func(l *nn.Linear) {
		ps = append(ps, l.W)
		if l.B != nil {
			ps = append(ps, l.B)
		}
	}
	for _, b := range m.Blocks {
		ps = append(ps, b.LNAttn.Gamma, b.LNAttn.Beta)
		if b.LNMLP != nil {
			ps = append(ps, b.LNMLP.Gamma, b.LNMLP.Beta)
		}
		add(b.QKV)
		add(b.Dense)
		add(b.Up)
		add(b.Down)
	}
	ps = append(ps, m.LNF.Gamma, m.LNF.Beta)
	if !m.Cfg.TieEmbed {
		ps = append(ps, m.LMHead.W)
	}
	return ps
}

func (m *Falcon) ParamCount() int {
	n := 0
	for _, p := range m.Parameters() {
		n += p.NumElements()
	}
	return n
}

// Forward is a stateless ForwardWithCache.
func (m *Falcon) Forward(tokens *tensor.Tensor) (*tensor.Tensor, error) {
	return m.ForwardWithCache(tokens, nil)
}

// ForwardWithCache runs tokens (B, T) I64 and returns F32 logits (B, T, V).
func (m *Falcon) ForwardWithCache(tokens *tensor.Tensor, cache *KVCache) (*tensor.Tensor, error) {
	if tokens.Rank() != 2 || tokens.DType() != dtype.I64 {
		return nil, fmt.Errorf("Falcon.Forward: tokens must be (B, T) I64, got %v %v", tokens.Shape(), tokens.DType())
	}
	B, T := tokens.Dim(0), tokens.Dim(1)
	dev := tokens.Device()
	prior := 0
	if cache != nil {
		if len(cache.K) != m.Cfg.NumLayers {
			return nil, fmt.Errorf("Falcon.ForwardWithCache: cache has %d layers, model %d", len(cache.K), m.Cfg.NumLayers)
		}
		prior = cache.Len
	}
	if prior+T > m.Cfg.MaxPosition {
		return nil, fmt.Errorf("Falcon.Forward: sequence length %d > MaxPosition %d", prior+T, m.Cfg.MaxPosition)
	}
	h, err := m.TokenEmb.Forward(tokens)
	if err != nil {
		return nil, err
	}
	var pos *tensor.Tensor
	if !m.Cfg.Alibi {
		posData := make([]int64, T)
		for i := range posData {
			posData[i] = int64(prior + i)
		}
		if pos, err = tensor.New(posData, []int{T}, dev); err != nil {
			return nil, err
		}
	}
	mask, err := m.mask(T, prior, h.DType(), dev)
	if err != nil {
		return nil, err
	}
	for i, blk := range m.Blocks {
		if h, err = m.runBlock(blk, h, B, T, prior, pos, mask, cache, i); err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
	}
	if cache != nil {
		cache.Len = prior + T
	}
	if h, err = m.LNF.Forward(h); err != nil {
		return nil, err
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

func (m *Falcon) runBlock(blk *falconBlock, x *tensor.Tensor, B, T, prior int, pos, mask *tensor.Tensor, cache *KVCache, layer int) (*tensor.Tensor, error) {
	cfg := m.Cfg
	lnAttn, err := blk.LNAttn.Forward(x)
	if err != nil {
		return nil, err
	}
	attn, err := m.attention(blk, lnAttn, B, T, prior, pos, mask, cache, layer)
	if err != nil {
		return nil, err
	}
	var lnMLP, residual *tensor.Tensor
	switch {
	case !cfg.ParallelAttn:
		// Sequential: x += attn; x += mlp(post_attention_layernorm(x)).
		if residual, err = ops.Add(x, attn); err != nil {
			return nil, err
		}
		if lnMLP, err = blk.LNMLP.Forward(residual); err != nil {
			return nil, err
		}
	case cfg.twoLN():
		residual = x
		if lnMLP, err = blk.LNMLP.Forward(x); err != nil {
			return nil, err
		}
	default:
		residual, lnMLP = x, lnAttn
	}
	up, err := blk.Up.Forward(lnMLP)
	if err != nil {
		return nil, err
	}
	if up, err = ops.GeluErf(up); err != nil {
		return nil, err
	}
	mlp, err := blk.Down.Forward(up)
	if err != nil {
		return nil, err
	}
	if cfg.ParallelAttn {
		if mlp, err = ops.Add(mlp, attn); err != nil {
			return nil, err
		}
	}
	return ops.Add(residual, mlp)
}

// attention splits the fused projection per the variant's layout, applies
// RoPE (rotary models), grows the cache and runs scaled dot-product
// attention with the causal (+ALiBi) mask.
func (m *Falcon) attention(blk *falconBlock, h *tensor.Tensor, B, T, prior int, pos, mask *tensor.Tensor, cache *KVCache, layer int) (*tensor.Tensor, error) {
	cfg := m.Cfg
	NH, KH, HD := cfg.NumHeads, cfg.kvHeads(), cfg.headDim()
	g := NH / KH
	fused, err := blk.QKV.Forward(h) // (B, T, (NH+2KH)·HD)
	if err != nil {
		return nil, err
	}
	var q, k, v *tensor.Tensor
	switch {
	case cfg.NewDecoderArch:
		// (B, T, KH, g+2, HD): g query heads, then K, then V per KV group.
		if fused, err = autograd.Reshape(fused, B, T, KH, g+2, HD); err != nil {
			return nil, err
		}
		if q, err = fused.Narrow(3, 0, g); err != nil {
			return nil, err
		}
		if q, err = autograd.Contiguous(q); err != nil {
			return nil, err
		}
		if q, err = autograd.Reshape(q, B, T, NH, HD); err != nil {
			return nil, err
		}
		if k, err = fused.Narrow(3, g, 1); err != nil {
			return nil, err
		}
		if v, err = fused.Narrow(3, g+1, 1); err != nil {
			return nil, err
		}
	case cfg.MultiQuery:
		// (B, T, NH+2, HD): NH query heads, one K, one V.
		if fused, err = autograd.Reshape(fused, B, T, NH+2, HD); err != nil {
			return nil, err
		}
		if q, err = fused.Narrow(2, 0, NH); err != nil {
			return nil, err
		}
		if k, err = fused.Narrow(2, NH, 1); err != nil {
			return nil, err
		}
		if v, err = fused.Narrow(2, NH+1, 1); err != nil {
			return nil, err
		}
	default:
		// (B, T, NH, 3, HD): q, k, v interleaved per head.
		if fused, err = autograd.Reshape(fused, B, T, NH, 3, HD); err != nil {
			return nil, err
		}
		if q, err = fused.Narrow(3, 0, 1); err != nil {
			return nil, err
		}
		if k, err = fused.Narrow(3, 1, 1); err != nil {
			return nil, err
		}
		if v, err = fused.Narrow(3, 2, 1); err != nil {
			return nil, err
		}
	}
	// To (B, H, T, HD).
	toHeads := func(x *tensor.Tensor, H int) (*tensor.Tensor, error) {
		x, err := autograd.Contiguous(x)
		if err != nil {
			return nil, err
		}
		if x, err = autograd.Reshape(x, B, T, H, HD); err != nil {
			return nil, err
		}
		if x, err = autograd.Permute(x, 0, 2, 1, 3); err != nil {
			return nil, err
		}
		return autograd.Contiguous(x)
	}
	if q, err = toHeads(q, NH); err != nil {
		return nil, err
	}
	if k, err = toHeads(k, KH); err != nil {
		return nil, err
	}
	if v, err = toHeads(v, KH); err != nil {
		return nil, err
	}
	if pos != nil {
		if q, err = ops.RoPE(q, pos, cfg.RopeBase); err != nil {
			return nil, err
		}
		if k, err = ops.RoPE(k, pos, cfg.RopeBase); err != nil {
			return nil, err
		}
	}
	total := T
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
		total = prior + T
	}
	if k, err = repeatKV(k, g, B, KH, total, HD); err != nil {
		return nil, err
	}
	if v, err = repeatKV(v, g, B, KH, total, HD); err != nil {
		return nil, err
	}
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
	scores, err := autograd.MatMul(q, kT)
	if err != nil {
		return nil, err
	}
	if scores, err = ops.Mul(scores, m.scale); err != nil {
		return nil, err
	}
	if mask != nil {
		if scores, err = autograd.Reshape(scores, B, NH, T, total); err != nil {
			return nil, err
		}
		if scores, err = ops.Add(scores, mask); err != nil { // (NH or 1, T, total)
			return nil, err
		}
		if scores, err = autograd.Reshape(scores, B*NH, T, total); err != nil {
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
	return blk.Dense.Forward(ctx)
}

// AlibiSlopes returns transformers' build_alibi_tensor slopes for n heads:
// base^(1..p) for p = the largest power of two ≤ n, plus odd powers of a
// second base for the remaining heads.
func AlibiSlopes(n int) []float64 {
	p := 1 << int(math.Floor(math.Log2(float64(n))))
	base := math.Pow(2, -math.Pow(2, -(math.Log2(float64(p))-3)))
	out := make([]float64, 0, n)
	for i := 1; i <= p; i++ {
		out = append(out, math.Pow(base, float64(i)))
	}
	if p != n {
		extra := math.Pow(2, -math.Pow(2, -(math.Log2(float64(2*p))-3)))
		rem := min(p, n-p)
		for i := 0; i < rem; i++ {
			out = append(out, math.Pow(extra, float64(1+2*i)))
		}
	}
	return out
}

// mask returns the additive attention mask for T queries after `prior`
// cached keys: causal −∞ above the diagonal, shaped (1, T, total); with
// ALiBi it is (NH, T, total) and also carries slope·key_position/√HD
// (transformers computes slope·position in bfloat16, reproduced here).
// nil when nothing needs masking (single-token step without ALiBi).
func (m *Falcon) mask(T, prior int, dt dtype.DType, dev device.Device) (*tensor.Tensor, error) {
	if T == 1 && !m.Cfg.Alibi {
		return nil, nil
	}
	if m.masks == nil || m.masksD == nil || !m.masksD.SameAs(dev) {
		m.masks = map[[2]int]*tensor.Tensor{}
		m.masksD = dev
	}
	key := [2]int{T, prior}
	if t, ok := m.masks[key]; ok && t.DType() == dt {
		return t, nil
	}
	total := prior + T
	heads := 1
	if m.Cfg.Alibi {
		heads = m.Cfg.NumHeads
	}
	negInf := float32(math.Inf(-1))
	data := make([]float32, heads*T*total)
	var slopes []float64
	if m.Cfg.Alibi {
		slopes = AlibiSlopes(m.Cfg.NumHeads)
	}
	inv := 1 / math.Sqrt(float64(m.Cfg.headDim()))
	for hd := range heads {
		for i := range T {
			row := data[(hd*T+i)*total : (hd*T+i+1)*total]
			for j := range total {
				if j > prior+i {
					row[j] = negInf
					continue
				}
				if slopes != nil {
					s := dtype.BF16FromF32(float32(slopes[hd])).ToF32()
					row[j] = dtype.BF16FromF32(s*float32(j)).ToF32() * float32(inv)
				}
			}
		}
	}
	t, err := tensor.New(data, []int{heads, T, total}, dev)
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

var _ CausalLM = (*Falcon)(nil)
