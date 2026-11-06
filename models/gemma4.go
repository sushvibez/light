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

// Gemma4Config describes the Gemma 4 text decoder (transformers
// modeling_gemma4.py). Beyond Gemma 3 it adds per-layer embeddings (PLE),
// key/value sharing across the last NumKVSharedLayers layers, a double-wide
// MLP on those layers, a wider global-attention head with proportional
// (partial) RoPE, an unscaled attention logit (q_norm takes the role of the
// 1/√d), an unweighted RMS norm on V, a per-layer output scalar and direct
// (not 1+w) RMSNorm weights.
type Gemma4Config struct {
	VocabSize         int
	HiddenSize        int
	IntermediateSize  int
	NumLayers         int
	NumHeads          int
	NumKVHeads        int // sliding layers
	NumGlobalKVHeads  int // full-attention layers (defaults to NumKVHeads)
	HeadDim           int // sliding layers
	GlobalHeadDim     int // full-attention layers (512)
	MaxPosition       int
	RopeBase          float64 // full-attention layers (1e6)
	RopeLocalBase     float64 // sliding layers (1e4)
	PartialRotary     float64 // fraction of GlobalHeadDim that is rotated (0.25)
	RmsEps            float64
	SlidingWindow     int
	SlidingLayers     []bool // per layer
	FinalLogitSoftcap float64
	PLEDim            int // hidden_size_per_layer_input (0 = no PLE)
	PLEVocab          int // vocab_size_per_layer_input
	NumKVSharedLayers int
	DoubleWideMLP     bool // on KV-shared layers
	AttentionKEqV     bool // value = key projection
	WeightDType       dtype.DType
}

func (cfg Gemma4Config) activationDType() dtype.DType {
	if cfg.WeightDType == dtype.BF16 {
		return dtype.BF16
	}
	return dtype.F32
}

func (cfg Gemma4Config) isSliding(layer int) bool {
	return layer < len(cfg.SlidingLayers) && cfg.SlidingLayers[layer]
}

func (cfg Gemma4Config) firstSharedLayer() int {
	if cfg.NumKVSharedLayers <= 0 {
		return cfg.NumLayers
	}
	return cfg.NumLayers - cfg.NumKVSharedLayers
}

func (cfg Gemma4Config) isKVShared(layer int) bool {
	return cfg.NumKVSharedLayers > 0 && layer >= cfg.firstSharedLayer()
}

// headDim / kvHeads per layer type.
func (cfg Gemma4Config) layerHeadDim(layer int) int {
	if cfg.isSliding(layer) {
		return cfg.HeadDim
	}
	return cfg.GlobalHeadDim
}

func (cfg Gemma4Config) layerKVHeads(layer int) int {
	if cfg.isSliding(layer) || cfg.NumGlobalKVHeads == 0 {
		return cfg.NumKVHeads
	}
	return cfg.NumGlobalKVHeads
}

// kvSourceLayer returns, for a KV-shared layer, the last non-shared layer of
// the same attention type whose K/V it reuses.
func (cfg Gemma4Config) kvSourceLayer(layer int) int {
	first := cfg.firstSharedLayer()
	for i := first - 1; i >= 0; i-- {
		if cfg.isSliding(i) == cfg.isSliding(layer) {
			return i
		}
	}
	return -1
}

type gemma4Block struct {
	InNorm, PostAttnNorm, PreFFNorm, PostFFNorm *nn.RMSNorm
	QNorm, KNorm                                *nn.RMSNorm // over the layer's head dim; KNorm nil on shared layers
	QProj, KProj, VProj, OProj                  *nn.Linear  // KProj/VProj nil on shared layers (VProj also nil with AttentionKEqV)
	GateProj, UpProj, DownProj                  *nn.Linear
	PLEGate, PLEProj                            *nn.Linear  // per_layer_input_gate (D→P), per_layer_projection (P→D)
	PostPLENorm                                 *nn.RMSNorm // post_per_layer_input_norm
	LayerScalar                                 *tensor.Tensor
	sliding, shared                             bool
	headDim, kvHeads                            int
}

// Gemma4 is a Gemma 4 text model (E2B/E4B and the dense sizes; MoE blocks
// are not supported). Weights are installed by LoadGemma4FromHF.
type Gemma4 struct {
	Cfg       Gemma4Config
	TokenEmb  *nn.Embedding
	PLEEmb    *nn.Embedding // (PLEVocab, NumLayers·PLEDim), nil without PLE
	PLEProj   *nn.Linear    // per_layer_model_projection (D → NumLayers·PLEDim), scale folded in
	PLENorm   *nn.RMSNorm   // per_layer_projection_norm over PLEDim
	Blocks    []*gemma4Block
	FinalNorm *nn.RMSNorm
	LMHead    *nn.Linear // tied to TokenEmb

	embScale    *tensor.Tensor // √D rounded to the activation dtype
	pleEmbScale *tensor.Tensor // √PLEDim
	pleMix      *tensor.Tensor // 2^-½
	tabs        gemma4Tables
}

// gemma4Tables caches the device-resident constants a Gemma 4 style model
// needs: all-ones RMSNorm scales for V, rotary inverse-frequency tables per
// layer type and additive attention masks.
type gemma4Tables struct {
	vNormOnes map[int]*tensor.Tensor
	invFreq   map[bool]*tensor.Tensor // sliding → table
	tablesDev device.Device
	masks     map[[3]int]*tensor.Tensor
	masksDev  device.Device
}

// NewGemma4 allocates an uninitialised model.
func NewGemma4(cfg Gemma4Config, dev device.Device) (*Gemma4, error) {
	if cfg.WeightDType != dtype.BF16 {
		cfg.WeightDType = dtype.F32
	}
	if cfg.GlobalHeadDim == 0 {
		cfg.GlobalHeadDim = cfg.HeadDim
	}
	if cfg.PartialRotary == 0 {
		cfg.PartialRotary = 1
	}
	if len(cfg.SlidingLayers) != cfg.NumLayers {
		return nil, fmt.Errorf("gemma4: SlidingLayers has %d entries for %d layers", len(cfg.SlidingLayers), cfg.NumLayers)
	}
	act := cfg.activationDType()
	D, NH, IM := cfg.HiddenSize, cfg.NumHeads, cfg.IntermediateSize
	emb, err := nn.NewEmbeddingUninit(cfg.VocabSize, D, act, dev)
	if err != nil {
		return nil, err
	}
	norm := func(n int) (*nn.RMSNorm, error) { return nn.NewRMSNormWithDType(n, cfg.RmsEps, act, dev) }
	lin := func(in, out int) (*nn.Linear, error) { return nn.NewLinearUninit(in, out, false, act, dev) }
	m := &Gemma4{Cfg: cfg, TokenEmb: emb}
	if cfg.PLEDim > 0 {
		if m.PLEEmb, err = nn.NewEmbeddingUninit(cfg.PLEVocab, cfg.NumLayers*cfg.PLEDim, act, dev); err != nil {
			return nil, err
		}
		if m.PLEProj, err = lin(D, cfg.NumLayers*cfg.PLEDim); err != nil {
			return nil, err
		}
		if m.PLENorm, err = norm(cfg.PLEDim); err != nil {
			return nil, err
		}
	}
	m.Blocks = make([]*gemma4Block, cfg.NumLayers)
	for i := range m.Blocks {
		b := &gemma4Block{sliding: cfg.isSliding(i), shared: cfg.isKVShared(i), headDim: cfg.layerHeadDim(i), kvHeads: cfg.layerKVHeads(i)}
		if NH%b.kvHeads != 0 {
			return nil, fmt.Errorf("gemma4 layer %d: NumHeads %d not divisible by kv heads %d", i, NH, b.kvHeads)
		}
		im := IM
		if cfg.DoubleWideMLP && b.shared {
			im *= 2
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
		b.QProj, err = lin(D, NH*b.headDim)
		push(err)
		b.OProj, err = lin(NH*b.headDim, D)
		push(err)
		if !b.shared {
			b.KNorm, err = norm(b.headDim)
			push(err)
			b.KProj, err = lin(D, b.kvHeads*b.headDim)
			push(err)
			if !cfg.AttentionKEqV || b.sliding {
				b.VProj, err = lin(D, b.kvHeads*b.headDim)
				push(err)
			}
		}
		b.GateProj, err = lin(D, im)
		push(err)
		b.UpProj, err = lin(D, im)
		push(err)
		b.DownProj, err = lin(im, D)
		push(err)
		if cfg.PLEDim > 0 {
			b.PLEGate, err = lin(D, cfg.PLEDim)
			push(err)
			b.PLEProj, err = lin(cfg.PLEDim, D)
			push(err)
			b.PostPLENorm, err = norm(D)
			push(err)
		}
		b.LayerScalar, err = tensor.Zeros(act, []int{1}, dev)
		push(err)
		if len(errs) > 0 {
			return nil, fmt.Errorf("gemma4 layer %d: %w", i, errs[0])
		}
		m.Blocks[i] = b
	}
	if m.FinalNorm, err = norm(D); err != nil {
		return nil, err
	}
	if m.LMHead, err = nn.NewLinearTied(emb.Weight, false, dev); err != nil {
		return nil, err
	}
	scalar := func(v float64) (*tensor.Tensor, error) {
		t, err := tensor.New([]float32{float32(v)}, []int{1}, dev)
		if err != nil {
			return nil, err
		}
		if act != dtype.F32 {
			return ops.Cast(t, act)
		}
		return t, nil
	}
	if m.embScale, err = scalar(math.Sqrt(float64(D))); err != nil {
		return nil, err
	}
	if cfg.PLEDim > 0 {
		if m.pleEmbScale, err = scalar(math.Sqrt(float64(cfg.PLEDim))); err != nil {
			return nil, err
		}
		if m.pleMix, err = scalar(1 / math.Sqrt2); err != nil {
			return nil, err
		}
	}
	return m, nil
}

func (m *Gemma4) NumLayers() int     { return m.Cfg.NumLayers }
func (m *Gemma4) VocabSize() int     { return m.Cfg.VocabSize }
func (m *Gemma4) MaxContext() int    { return m.Cfg.MaxPosition }
func (m *Gemma4) SlidingWindow() int { return m.Cfg.SlidingWindow }

// Parameters lists every weight once (the LM head is tied).
func (m *Gemma4) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokenEmb.Weight}
	if m.PLEEmb != nil {
		ps = append(ps, m.PLEEmb.Weight, m.PLEProj.W, m.PLENorm.Weight)
	}
	for _, b := range m.Blocks {
		for _, n := range []*nn.RMSNorm{b.InNorm, b.PostAttnNorm, b.PreFFNorm, b.PostFFNorm, b.QNorm, b.KNorm, b.PostPLENorm} {
			if n != nil {
				ps = append(ps, n.Weight)
			}
		}
		for _, l := range []*nn.Linear{b.QProj, b.KProj, b.VProj, b.OProj, b.GateProj, b.UpProj, b.DownProj, b.PLEGate, b.PLEProj} {
			if l != nil {
				ps = append(ps, l.W)
			}
		}
		ps = append(ps, b.LayerScalar)
	}
	return append(ps, m.FinalNorm.Weight)
}

func (m *Gemma4) ParamCount() int {
	n := 0
	for _, p := range m.Parameters() {
		n += p.NumElements()
	}
	return n
}

// Forward is a stateless ForwardWithCache.
func (m *Gemma4) Forward(tokens *tensor.Tensor) (*tensor.Tensor, error) {
	return m.ForwardWithCache(tokens, nil)
}

// ForwardWithCache runs tokens (B, T) I64 and returns F32 logits (B, T, V).
func (m *Gemma4) ForwardWithCache(tokens *tensor.Tensor, cache *KVCache) (*tensor.Tensor, error) {
	if tokens.Rank() != 2 || tokens.DType() != dtype.I64 {
		return nil, fmt.Errorf("Gemma4.Forward: tokens must be (B, T) I64, got %v %v", tokens.Shape(), tokens.DType())
	}
	B, T := tokens.Dim(0), tokens.Dim(1)
	dev := tokens.Device()
	prior := 0
	if cache != nil {
		if len(cache.K) != m.Cfg.NumLayers {
			return nil, fmt.Errorf("Gemma4.ForwardWithCache: cache has %d layers, model %d", len(cache.K), m.Cfg.NumLayers)
		}
		prior = cache.Len
	}
	if prior+T > m.Cfg.MaxPosition {
		return nil, fmt.Errorf("Gemma4.Forward: sequence length %d > MaxPosition %d", prior+T, m.Cfg.MaxPosition)
	}
	h, err := m.TokenEmb.Forward(tokens)
	if err != nil {
		return nil, err
	}
	if h, err = ops.Mul(h, m.embScale); err != nil {
		return nil, err
	}
	// Per-layer inputs: (B, T, L, P) = (norm(proj(h)/√D) + ple_embed(ids)·√P) / √2.
	var ple *tensor.Tensor
	if m.PLEEmb != nil {
		L, P := m.Cfg.NumLayers, m.Cfg.PLEDim
		ids, err := m.PLEEmb.Forward(tokens) // (B, T, L·P)
		if err != nil {
			return nil, err
		}
		if ids, err = ops.Mul(ids, m.pleEmbScale); err != nil {
			return nil, err
		}
		proj, err := m.PLEProj.Forward(h) // scale 1/√D folded into the weight
		if err != nil {
			return nil, err
		}
		if proj, err = autograd.Reshape(proj, B, T, L, P); err != nil {
			return nil, err
		}
		if proj, err = m.PLENorm.Forward(proj); err != nil {
			return nil, err
		}
		if ids, err = autograd.Reshape(ids, B, T, L, P); err != nil {
			return nil, err
		}
		if ple, err = ops.Add(proj, ids); err != nil {
			return nil, err
		}
		if ple, err = ops.Mul(ple, m.pleMix); err != nil {
			return nil, err
		}
		if ple, err = autograd.Permute(ple, 2, 0, 1, 3); err != nil { // (L, B, T, P)
			return nil, err
		}
		if ple, err = autograd.Contiguous(ple); err != nil {
			return nil, err
		}
	}
	posData := make([]int64, T)
	for i := range posData {
		posData[i] = int64(prior + i)
	}
	pos, err := tensor.New(posData, []int{T}, dev)
	if err != nil {
		return nil, err
	}
	shared := map[int]*kvState{}
	for i, blk := range m.Blocks {
		var pleI *tensor.Tensor
		if ple != nil {
			if pleI, err = ple.Narrow(0, i, 1); err != nil {
				return nil, err
			}
			if pleI, err = autograd.Reshape(pleI, B, T, m.Cfg.PLEDim); err != nil {
				return nil, err
			}
		}
		if h, err = m.runBlock(blk, h, pleI, B, T, pos, dev, cache, i, shared); err != nil {
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

// kvState is what a KV-shared layer borrows from its source layer: the
// post-cache keys/values (B, KH, totalT, HD) and the cache geometry.
type kvState struct {
	k, v            *tensor.Tensor
	kvPrior, totalT int
}

func (m *Gemma4) runBlock(blk *gemma4Block, x, pleI *tensor.Tensor, B, T int, pos *tensor.Tensor, dev device.Device, cache *KVCache, layer int, shared map[int]*kvState) (*tensor.Tensor, error) {
	cfg := m.Cfg
	NH, KH, HD := cfg.NumHeads, blk.kvHeads, blk.headDim
	g := NH / KH

	h, err := blk.InNorm.Forward(x)
	if err != nil {
		return nil, err
	}
	q, err := blk.QProj.Forward(h)
	if err != nil {
		return nil, err
	}
	if q, err = headsForward(q, B, T, NH, HD); err != nil {
		return nil, err
	}
	if q, err = blk.QNorm.Forward(q); err != nil {
		return nil, err
	}
	if q, err = m.rope(q, pos, blk.sliding, dev); err != nil {
		return nil, err
	}
	window := 0
	if blk.sliding {
		window = cfg.SlidingWindow
	}

	var k, v *tensor.Tensor
	var kvPrior, totalT int
	if blk.shared {
		src := cfg.kvSourceLayer(layer)
		st, ok := shared[src]
		if !ok {
			return nil, fmt.Errorf("gemma4: KV-shared layer %d has no source state (layer %d)", layer, src)
		}
		k, v, kvPrior, totalT = st.k, st.v, st.kvPrior, st.totalT
	} else {
		if k, err = blk.KProj.Forward(h); err != nil {
			return nil, err
		}
		if blk.VProj != nil {
			if v, err = blk.VProj.Forward(h); err != nil {
				return nil, err
			}
		} else {
			v = k // attention_k_eq_v: value is the raw key projection
		}
		if k, err = headsForward(k, B, T, KH, HD); err != nil {
			return nil, err
		}
		if v, err = headsForward(v, B, T, KH, HD); err != nil {
			return nil, err
		}
		if k, err = blk.KNorm.Forward(k); err != nil {
			return nil, err
		}
		if k, err = m.rope(k, pos, blk.sliding, dev); err != nil {
			return nil, err
		}
		if v, err = m.tabs.vNorm(v, HD, cfg.RmsEps, dev); err != nil {
			return nil, err
		}
		totalT = T
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
		shared[layer] = &kvState{k: k, v: v, kvPrior: kvPrior, totalT: totalT}
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
	scores, err := autograd.MatMul(q, kT) // scaling = 1: q_norm carries the scale
	if err != nil {
		return nil, err
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

	return blk.ffn(x, pleI)
}

// ffn is the second half of a block: the gated-GELU MLP, the per-layer
// input mix and the layer scalar.
func (blk *gemma4Block) ffn(x, pleI *tensor.Tensor) (*tensor.Tensor, error) {
	var err error
	// MLP: down(gelu_tanh(gate(h)) ⊙ up(h)), post-normed before the residual.
	h, err := blk.PreFFNorm.Forward(x)
	if err != nil {
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
	if x, err = ops.Add(x, mlp); err != nil {
		return nil, err
	}

	// Per-layer input: x += norm(proj(gelu(gate(x)) ⊙ ple_i)).
	if pleI != nil {
		gp, err := blk.PLEGate.Forward(x)
		if err != nil {
			return nil, err
		}
		if gp, err = ops.Gelu(gp); err != nil {
			return nil, err
		}
		if gp, err = ops.Mul(gp, pleI); err != nil {
			return nil, err
		}
		if gp, err = blk.PLEProj.Forward(gp); err != nil {
			return nil, err
		}
		if gp, err = blk.PostPLENorm.Forward(gp); err != nil {
			return nil, err
		}
		if x, err = ops.Add(x, gp); err != nil {
			return nil, err
		}
	}
	return ops.Mul(x, blk.LayerScalar)
}

// vNorm is RMS normalisation without a learned scale (Gemma4RMSNorm with
// with_scale=False), applied to V over the head dim.
func (tb *gemma4Tables) vNorm(v *tensor.Tensor, hd int, eps float64, dev device.Device) (*tensor.Tensor, error) {
	if tb.vNormOnes == nil {
		tb.vNormOnes = map[int]*tensor.Tensor{}
	}
	ones, ok := tb.vNormOnes[hd]
	if !ok || !ones.Device().SameAs(dev) {
		data := make([]float32, hd)
		for i := range data {
			data[i] = 1
		}
		t, err := tensor.New(data, []int{hd}, dev)
		if err != nil {
			return nil, err
		}
		if v.DType() != dtype.F32 {
			if t, err = ops.Cast(t, v.DType()); err != nil {
				return nil, err
			}
		}
		ones = t
		tb.vNormOnes[hd] = t
	}
	return ops.RMSNorm(v, ones, eps)
}

// rope applies the layer type's rotary table: sliding layers rotate the full
// head dim at RopeLocalBase; full-attention layers rotate only the first
// PartialRotary·GlobalHeadDim dims at RopeBase (proportional RoPE: the
// inverse frequencies are base^(-2i/GlobalHeadDim) for the rotated pairs
// and 0 for the rest, which leaves those dims untouched).
func (m *Gemma4) rope(x, pos *tensor.Tensor, sliding bool, dev device.Device) (*tensor.Tensor, error) {
	hd, base, frac := m.Cfg.GlobalHeadDim, m.Cfg.RopeBase, m.Cfg.PartialRotary
	if sliding {
		hd, base, frac = m.Cfg.HeadDim, m.Cfg.RopeLocalBase, 1
	}
	return m.tabs.rope(x, pos, sliding, hd, base, frac, dev)
}

// rope applies a cached rotary table (keyed by sliding) that rotates the
// first frac·hd dims of x at base.
func (tb *gemma4Tables) rope(x, pos *tensor.Tensor, sliding bool, hd int, base, frac float64, dev device.Device) (*tensor.Tensor, error) {
	if tb.tablesDev == nil || !tb.tablesDev.SameAs(dev) {
		tb.invFreq = map[bool]*tensor.Tensor{}
		tb.tablesDev = dev
	}
	table, ok := tb.invFreq[sliding]
	if !ok {
		half := hd / 2
		angles := int(frac * float64(hd) / 2)
		data := make([]float32, half)
		for i := 0; i < angles && i < half; i++ {
			data[i] = float32(math.Pow(base, -2*float64(i)/float64(hd)))
		}
		t, err := tensor.New(data, []int{half}, dev)
		if err != nil {
			return nil, err
		}
		table = t
		tb.invFreq[sliding] = t
	}
	return ops.RoPEWithInvFreq(x, pos, table)
}

func (m *Gemma4) mask(T, prior, window int, dt dtype.DType, dev device.Device) (*tensor.Tensor, error) {
	return m.tabs.mask(T, prior, window, false, dt, dev)
}

// mask returns the additive (T, prior+T) attention mask: causal with an
// optional sliding window of `window` past keys, or, with bidirectional
// set, every key within |i-j| <= window (prior must be 0).
func (tb *gemma4Tables) mask(T, prior, window int, bidirectional bool, dt dtype.DType, dev device.Device) (*tensor.Tensor, error) {
	if tb.masks == nil || tb.masksDev == nil || !tb.masksDev.SameAs(dev) {
		tb.masks = map[[3]int]*tensor.Tensor{}
		tb.masksDev = dev
	}
	key := [3]int{T, prior, window}
	if bidirectional {
		key[1] = -1
	}
	if t, ok := tb.masks[key]; ok && t.DType() == dt {
		return t, nil
	}
	total := prior + T
	negInf := float32(math.Inf(-1))
	data := make([]float32, T*total)
	for i := range T {
		p := prior + i
		if bidirectional {
			for j := range total {
				if d := j - p; window > 0 && (d > window || d < -window) {
					data[i*total+j] = negInf
				}
			}
			continue
		}
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
	if len(tb.masks) > 64 {
		tb.masks = map[[3]int]*tensor.Tensor{}
	}
	tb.masks[key] = t
	return t, nil
}

var _ CausalLM = (*Gemma4)(nil)
