package models

import (
	"fmt"
	"math"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/quant"
	"light/tensor"
)

// LlamaConfig describes a LLaMA / Qwen2 / Mistral-style decoder. The shape
// covers the union of the published variants: optional bias on Q/K/V (Qwen2
// has them, plain LLaMA does not), grouped-query attention via
// `NumKVHeads <= NumHeads`, and tied LM head for the smallest models.
type LlamaConfig struct {
	VocabSize        int
	HiddenSize       int          // model width D
	IntermediateSize int          // FFN inner width
	NumLayers        int          // L
	NumHeads         int          // NH (query heads)
	NumKVHeads       int          // KH; NumHeads must be a multiple of NumKVHeads
	HeadDim          int          // typically HiddenSize / NumHeads
	MaxPosition      int          // max sequence length the RoPE table supports
	RopeBase         float64      // RoPE θ-base (10000 for LLaMA-1, 1e6 for Qwen2/LLaMA-3)
	RmsEps           float64      // RMSNorm epsilon
	TieEmbed         bool         // share LM-head weight with the token embedding
	QKVBias          bool         // Q/K/V projections include a bias term
	WeightDType      dtype.DType  // BF16 for mixed-precision inference; defaults to F32
	SkipInit         bool         // skip Kaiming/uniform random init (loader will fill); turns model construction from minutes into seconds for 3B+ models
	RopeScaling      *RopeScaling // optional frequency scaling (Llama 3.1/3.2 "llama3"); nil = plain RoPE on RopeBase
	// Per-layer-type RoPE (Mellum 2.1: YaRN on full layers, plain RoPE on
	// sliding ones). When SlidingRopeBase > 0, layers that use SlidingWindow
	// rotate with SlidingRopeBase / SlidingRopeScaling, and RopeBase /
	// RopeScaling apply to the full-attention layers only.
	SlidingRopeBase    float64
	SlidingRopeScaling *RopeScaling
	PartialRotary      float64 // fraction of HeadDim that RoPE rotates (Phi-4-mini 0.75); 0 or 1 = all of it
	LazyLinear         bool    // allocate no Linear weights; the loader installs them (GGUF quantized models). Implies SkipInit.
	SlidingWindow      int     // Mistral/Gemma-style sliding-window attention: each query sees at most this many keys back. 0 = full causal.
	SlidingLayers      []bool  // optional per-layer switch (len NumLayers); nil = every layer uses SlidingWindow. Gemma alternates local/global layers.
	QKNorm             bool    // Qwen3: RMSNorm over the head dim of Q and K (per layer weights) before RoPE
	PostNorms          bool    // GLM-4-0414: RMSNorm on the attention and MLP outputs before each residual add
	// Granite scaling (0 = off): token embeddings × EmbedMultiplier, the
	// attention softmax scale AttnScale instead of 1/√HeadDim, residual
	// branches × ResidualMultiplier before each add, logits ÷ LogitsScaling.
	EmbedMultiplier    float64
	AttnScale          float64
	ResidualMultiplier float64
	LogitsScaling      float64
	// Llama 4 / Ministral 3 position-dependent query scaling (0 = off): q at
	// absolute position p is multiplied by
	// 1 + Llama4ScalingBeta · log(1 + floor(p / Llama4ScalingOrig)),
	// i.e. exactly 1 below Llama4ScalingOrig.
	Llama4ScalingBeta float64
	Llama4ScalingOrig int
	LayerNorm1P       bool // Nemotron: LayerNorm (mean-centred, with bias) instead of RMSNorm; loaders store gamma as 1 + w
	ReLU2MLP          bool // Nemotron: non-gated MLP down(relu(up(x))²); GateProj is nil
	// Mixture-of-experts FFN (Qwen3-MoE / Mixtral / DeepSeek-style). NumExperts
	// > 0 turns the FFN of every layer selected by DecoderSparseStep /
	// MLPOnlyLayers into an nn.SparseMoE with NumExpertsPerTok routed experts
	// of width MoEIntermediateSize; other layers keep the dense SwiGLU MLP.
	NumExperts          int
	NumExpertsPerTok    int
	MoEIntermediateSize int
	NormTopKProb        bool
	DecoderSparseStep   int   // layer i is MoE when (i+1) % step == 0 (HF semantics; 0 or 1 = every layer)
	MLPOnlyLayers       []int // layers forced dense regardless of the step
}

// IsMoELayer reports whether layer i uses the routed-expert FFN.
func (cfg LlamaConfig) IsMoELayer(i int) bool {
	if cfg.NumExperts <= 0 {
		return false
	}
	for _, l := range cfg.MLPOnlyLayers {
		if l == i {
			return false
		}
	}
	step := cfg.DecoderSparseStep
	if step <= 1 {
		return true
	}
	return (i+1)%step == 0
}

// slidingWindowFor returns the attention window for one layer (0 = full
// causal attention).
func (cfg LlamaConfig) slidingWindowFor(layer int) int {
	if cfg.SlidingWindow <= 0 {
		return 0
	}
	if cfg.SlidingLayers != nil && layer < len(cfg.SlidingLayers) && !cfg.SlidingLayers[layer] {
		return 0
	}
	return cfg.SlidingWindow
}

// ropeFor returns the RoPE base and scaling for one layer, and which cached
// table set (0 = full / default, 1 = sliding) it uses.
func (cfg LlamaConfig) ropeFor(layer int) (base float64, rs *RopeScaling, slot int) {
	if cfg.SlidingRopeBase > 0 && cfg.slidingWindowFor(layer) > 0 {
		return cfg.SlidingRopeBase, cfg.SlidingRopeScaling, 1
	}
	return cfg.RopeBase, cfg.RopeScaling, 0
}

// Qwen2_0_5B returns the public Qwen2-0.5B-Instruct config.
func Qwen2_0_5B() LlamaConfig {
	return LlamaConfig{
		VocabSize: 151936, HiddenSize: 896, IntermediateSize: 4864,
		NumLayers: 24, NumHeads: 14, NumKVHeads: 2, HeadDim: 64,
		MaxPosition: 32768, RopeBase: 1000000, RmsEps: 1e-6,
		TieEmbed: true, QKVBias: true, WeightDType: dtype.F32,
	}
}

// Qwen2_1_5B returns the Qwen2-1.5B-Instruct config (~1.5B params).
func Qwen2_1_5B() LlamaConfig {
	return LlamaConfig{
		VocabSize: 151936, HiddenSize: 1536, IntermediateSize: 8960,
		NumLayers: 28, NumHeads: 12, NumKVHeads: 2, HeadDim: 128,
		MaxPosition: 32768, RopeBase: 1000000, RmsEps: 1e-6,
		TieEmbed: true, QKVBias: true, WeightDType: dtype.F32,
	}
}

// llamaBlock is one pre-norm decoder layer:
//
//	x' = x + OProj( Attn( RMSNorm(x) ) )
//	y  = x' + DownProj( SiLU(GateProj(RMSNorm(x'))) ⊙ UpProj(RMSNorm(x')) )
type llamaBlock struct {
	N1                         *nn.RMSNorm
	QProj, KProj, VProj, OProj *nn.Linear
	QNorm, KNorm               *nn.RMSNorm   // Qwen3 QK-norm (HeadDim); nil otherwise
	PostAttnNorm, PostMLPNorm  *nn.RMSNorm   // GLM-4-0414 post-norms (Cfg.PostNorms); nil otherwise
	LN1, LN2                   *nn.LayerNorm // Nemotron (Cfg.LayerNorm1P): used instead of N1 / N2
	N2                         *nn.RMSNorm
	GateProj, UpProj, DownProj *nn.Linear    // dense SwiGLU MLP; nil on MoE layers
	MoE                        *nn.SparseMoE // routed-expert FFN; nil on dense layers
}

func newLlamaBlock(cfg LlamaConfig, layer int, dev device.Device) (*llamaBlock, error) {
	D, NH, KH, HD, IM := cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.IntermediateSize
	if KH <= 0 || NH%KH != 0 {
		return nil, fmt.Errorf("llama: NumHeads (%d) not divisible by NumKVHeads (%d)", NH, KH)
	}
	// LlamaConfig's zero-value is F16 (iota=0 in the dtype enum), but we
	// don't support F16 matmul yet — and a user constructing a bare
	// LlamaConfig{} almost certainly means F32. Treat F16 as "unset" until
	// the F16 kernel path lands; explicit BF16/F32 selections pass through.
	wdt := cfg.WeightDType
	if wdt == dtype.F16 {
		wdt = dtype.F32
	}
	mkLinear := nn.NewLinearWithDType
	if cfg.SkipInit {
		mkLinear = nn.NewLinearUninit
	}
	if cfg.LazyLinear {
		mkLinear = func(in, out int, bias bool, _ dtype.DType, dev device.Device) (*nn.Linear, error) {
			return nn.NewLinearPlaceholder(in, out, bias, dev)
		}
	}
	n1, err := nn.NewRMSNormWithDType(D, cfg.RmsEps, wdt, dev)
	if err != nil {
		return nil, err
	}
	q, err := mkLinear(D, NH*HD, cfg.QKVBias, wdt, dev)
	if err != nil {
		return nil, err
	}
	k, err := mkLinear(D, KH*HD, cfg.QKVBias, wdt, dev)
	if err != nil {
		return nil, err
	}
	v, err := mkLinear(D, KH*HD, cfg.QKVBias, wdt, dev)
	if err != nil {
		return nil, err
	}
	o, err := mkLinear(NH*HD, D, false, wdt, dev)
	if err != nil {
		return nil, err
	}
	n2, err := nn.NewRMSNormWithDType(D, cfg.RmsEps, wdt, dev)
	if err != nil {
		return nil, err
	}
	blk := &llamaBlock{
		N1: n1, QProj: q, KProj: k, VProj: v, OProj: o,
		N2: n2,
	}
	if cfg.IsMoELayer(layer) {
		var moe *nn.SparseMoE
		if cfg.LazyLinear {
			moe, err = nn.NewSparseMoELazy(D, cfg.NumExperts, cfg.NumExpertsPerTok, cfg.NormTopKProb, dev)
		} else {
			moe, err = nn.NewSparseMoEUninit(D, cfg.MoEIntermediateSize, cfg.NumExperts, cfg.NumExpertsPerTok, cfg.NormTopKProb, wdt, dev)
		}
		if err != nil {
			return nil, err
		}
		blk.MoE = moe
	} else {
		if !cfg.ReLU2MLP {
			if blk.GateProj, err = mkLinear(D, IM, false, wdt, dev); err != nil {
				return nil, err
			}
		}
		if blk.UpProj, err = mkLinear(D, IM, false, wdt, dev); err != nil {
			return nil, err
		}
		if blk.DownProj, err = mkLinear(IM, D, false, wdt, dev); err != nil {
			return nil, err
		}
	}
	if cfg.QKNorm {
		if blk.QNorm, err = nn.NewRMSNormWithDType(HD, cfg.RmsEps, wdt, dev); err != nil {
			return nil, err
		}
		if blk.KNorm, err = nn.NewRMSNormWithDType(HD, cfg.RmsEps, wdt, dev); err != nil {
			return nil, err
		}
	}
	if cfg.LayerNorm1P {
		if blk.LN1, err = nn.NewLayerNormWithDType(D, cfg.RmsEps, dtype.F32, dev); err != nil {
			return nil, err
		}
		if blk.LN2, err = nn.NewLayerNormWithDType(D, cfg.RmsEps, dtype.F32, dev); err != nil {
			return nil, err
		}
	}
	if cfg.PostNorms {
		if blk.PostAttnNorm, err = nn.NewRMSNormWithDType(D, cfg.RmsEps, wdt, dev); err != nil {
			return nil, err
		}
		if blk.PostMLPNorm, err = nn.NewRMSNormWithDType(D, cfg.RmsEps, wdt, dev); err != nil {
			return nil, err
		}
	}
	return blk, nil
}

func (b *llamaBlock) parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, b.N1.Parameters()...)
	ps = append(ps, b.QProj.Parameters()...)
	ps = append(ps, b.KProj.Parameters()...)
	ps = append(ps, b.VProj.Parameters()...)
	ps = append(ps, b.OProj.Parameters()...)
	if b.QNorm != nil {
		ps = append(ps, b.QNorm.Parameters()...)
		ps = append(ps, b.KNorm.Parameters()...)
	}
	ps = append(ps, b.N2.Parameters()...)
	if b.PostAttnNorm != nil {
		ps = append(ps, b.PostAttnNorm.Parameters()...)
		ps = append(ps, b.PostMLPNorm.Parameters()...)
	}
	if b.LN1 != nil {
		ps = append(ps, b.LN1.Parameters()...)
		ps = append(ps, b.LN2.Parameters()...)
	}
	if b.MoE != nil {
		ps = append(ps, b.MoE.Gate.Parameters()...)
		if b.MoE.GateUpQ4 != nil {
			for e := range b.MoE.GateUpQ4 {
				for _, w := range []quant.Int4{b.MoE.GateUpQ4[e], b.MoE.DownQ4[e]} {
					ps = append(ps, w.Codes, w.Scales, w.Zeros)
				}
			}
			return ps
		}
		ps = append(ps, b.MoE.GateUp, b.MoE.Down)
		return ps
	}
	if b.GateProj != nil {
		ps = append(ps, b.GateProj.Parameters()...)
	}
	ps = append(ps, b.UpProj.Parameters()...)
	ps = append(ps, b.DownProj.Parameters()...)
	return ps
}

// Llama is a LLaMA / Qwen2 / Mistral decoder LLM. Forward consumes token IDs
// of shape (B, T) and returns logits (B, T, V).
type Llama struct {
	Cfg       LlamaConfig
	TokenEmb  *nn.Embedding
	Blocks    []*llamaBlock
	FinalNorm *nn.RMSNorm
	FinalLN   *nn.LayerNorm // Nemotron (Cfg.LayerNorm1P): used instead of FinalNorm
	// MTP is a multi-token-prediction draft head (MiMo-7B's mtp_layers.0),
	// nil when the checkpoint has none; LastHidden is the final normed
	// hidden state of the latest forward (B, T, D), which it consumes.
	MTP        *MTPHead
	LastHidden *tensor.Tensor
	LMHead     *nn.Linear // (V, D); shares storage with TokenEmb when TieEmbed.

	scale      *tensor.Tensor // (1,) F32 = 1/sqrt(HeadDim) on dev
	scaleDev   device.Device
	maskTotal  int
	maskPrior  int
	maskT      int
	maskWindow int
	maskBlock  int
	maskDev    device.Device
	maskCache  *tensor.Tensor

	// Block-diffusion decoding state for the current ForwardDiffusion call
	// (SDAR): diffBlock > 0 selects the block-bidirectional mask, diffStore
	// false computes against the cache without appending to it.
	diffBlock int
	diffStore bool
	posS      int
	posOff    int
	posDev    device.Device
	posCache  *tensor.Tensor

	rope       [2]ropeTables // scaled RoPE tables per layer type (Cfg.ropeFor slot); built lazily per device
	invFreqDev device.Device
}

// ropeTables caches one RoPE scheme's tables on a device.
type ropeTables struct {
	invFreq     *tensor.Tensor // (rotDim/2,) F32 scaled RoPE table
	invFreqLong *tensor.Tensor // LongRoPE long-factor table (seqLen > OriginalMaxPositions)
	attn        *tensor.Tensor // (1,) F32 attention factor (LongRoPE / YaRN)
}

// resetRoPETables drops the cached RoPE tables, e.g. after changing
// Cfg.RopeScaling or Cfg.PartialRotary on a live model.
func (m *Llama) resetRoPETables() {
	m.rope = [2]ropeTables{}
	m.invFreqDev = nil
}

// KVCache stores per-layer K and V tensors so we don't recompute past tokens
// during autoregressive generation. Allocate with NewKVCache(cfg.NumLayers)
// and pass to Llama.ForwardWithCache. Cache grows by T_new on each call.
//
// Layout: K[i] and V[i] both have shape (B, KH, kvLen, HD). Storage is grown
// via concat each step — simple but O(N²) memcpy across N tokens. Good enough
// to validate correctness; an in-place ring-buffer is a future optimisation.
//
// Len is the absolute number of tokens seen (drives RoPE positions). With
// sliding-window attention a layer keeps only its last `window` entries, so
// K[i].Dim(2) may be smaller than Len (Candle's RotatingKvCache).
//
// Linear-attention layers (Gated DeltaNet, Qwen3.5) keep fixed-size state
// instead of K/V: Conv[i] is the short-conv state (B, C, kernel−1) and
// Rec[i] the delta-rule state (B, H, dk, dv), both F32.
type KVCache struct {
	K    []*tensor.Tensor
	V    []*tensor.Tensor
	Conv []*tensor.Tensor
	Rec  []*tensor.Tensor
	Len  int
	// ring marks K / V as fixed-size ring buffers (Moshi's one-frame decode
	// path) rather than position-ordered concatenations.
	ring bool
}

// NewKVCache allocates a fresh cache for a model with `numLayers` blocks.
func NewKVCache(numLayers int) *KVCache {
	return &KVCache{
		K:    make([]*tensor.Tensor, numLayers),
		V:    make([]*tensor.Tensor, numLayers),
		Conv: make([]*tensor.Tensor, numLayers),
		Rec:  make([]*tensor.Tensor, numLayers),
	}
}

// Truncate drops every cached position from n on (speculative decoding
// rolls back a rejected draft this way). Only key / value caches can be
// truncated; recurrent state (Conv / Rec) cannot be un-stepped.
func (c *KVCache) Truncate(n int) error {
	if n < 0 || n > c.Len {
		return fmt.Errorf("KVCache.Truncate: %d outside [0, %d]", n, c.Len)
	}
	if c.ring {
		return fmt.Errorf("KVCache.Truncate: ring-buffer cache")
	}
	for i := range c.Conv {
		if c.Conv[i] != nil || c.Rec[i] != nil {
			return fmt.Errorf("KVCache.Truncate: layer %d carries recurrent state", i)
		}
	}
	for i, k := range c.K {
		if k == nil {
			continue
		}
		if k.Dim(2) != c.Len {
			return fmt.Errorf("KVCache.Truncate: layer %d holds %d positions of %d (sliding window)", i, k.Dim(2), c.Len)
		}
		var err error
		if c.K[i], err = narrowContig(k, 2, 0, n); err != nil {
			return err
		}
		if c.V[i], err = narrowContig(c.V[i], 2, 0, n); err != nil {
			return err
		}
	}
	c.Len = n
	return nil
}

// Reset empties the cache (frees stored K/V and linear-attention state).
// Call between independent prompts to reuse the same cache struct.
func (c *KVCache) Reset() {
	for i := range c.K {
		c.K[i] = nil
		c.V[i] = nil
	}
	for i := range c.Conv {
		c.Conv[i] = nil
		c.Rec[i] = nil
	}
	c.Len = 0
	c.ring = false
}

// NewLlama constructs a fresh model with the given config on dev. Loaders
// overwrite parameter values in place.
func NewLlama(cfg LlamaConfig, dev device.Device) (*Llama, error) {
	if cfg.HeadDim%2 != 0 {
		return nil, fmt.Errorf("llama: HeadDim must be even for RoPE (got %d)", cfg.HeadDim)
	}
	embedDT := dtype.F32
	if cfg.WeightDType == dtype.BF16 {
		// Embedding lives at BF16 in the all-BF16 path so the lookup
		// produces BF16 directly and the tied LM-head stays BF16×BF16.
		embedDT = dtype.BF16
	}
	var tokEmb *nn.Embedding
	var err error
	if cfg.SkipInit {
		tokEmb, err = nn.NewEmbeddingUninit(cfg.VocabSize, cfg.HiddenSize, embedDT, dev)
		if err != nil {
			return nil, err
		}
	} else {
		tokEmb, err = nn.NewEmbedding(cfg.VocabSize, cfg.HiddenSize, dev)
		if err != nil {
			return nil, err
		}
		if embedDT != dtype.F32 {
			bf16W, err := ops.Cast(tokEmb.Weight, embedDT)
			if err != nil {
				return nil, fmt.Errorf("llama: cast embedding to %v: %w", embedDT, err)
			}
			tokEmb.Weight = bf16W
		}
	}
	blocks := make([]*llamaBlock, cfg.NumLayers)
	for i := range blocks {
		b, err := newLlamaBlock(cfg, i, dev)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
		blocks[i] = b
	}
	rmsDT := cfg.WeightDType
	if rmsDT == dtype.F16 {
		rmsDT = dtype.F32
	}
	var finalLN *nn.LayerNorm
	if cfg.LayerNorm1P {
		if finalLN, err = nn.NewLayerNormWithDType(cfg.HiddenSize, cfg.RmsEps, dtype.F32, dev); err != nil {
			return nil, err
		}
	}
	fn, err := nn.NewRMSNormWithDType(cfg.HiddenSize, cfg.RmsEps, rmsDT, dev)
	if err != nil {
		return nil, err
	}
	headDT := cfg.WeightDType
	if headDT == dtype.F16 { // see newLlamaBlock — F16 zero-value means "default to F32".
		headDT = dtype.F32
	}
	var head *nn.Linear
	if cfg.TieEmbed {
		// Reuse the embedding's storage — no fresh (V, D) allocation. Saves
		// ~300 MB of throwaway BF16 weight on Qwen2.5-3B that otherwise sits
		// in pending-finalizer state and inflates the post-load GPU reading.
		head, err = nn.NewLinearTied(tokEmb.Weight, false, dev)
	} else if cfg.LazyLinear {
		head, err = nn.NewLinearPlaceholder(cfg.HiddenSize, cfg.VocabSize, false, dev)
	} else if cfg.SkipInit {
		head, err = nn.NewLinearUninit(cfg.HiddenSize, cfg.VocabSize, false, headDT, dev)
	} else {
		head, err = nn.NewLinearWithDType(cfg.HiddenSize, cfg.VocabSize, false, headDT, dev)
	}
	if err != nil {
		return nil, err
	}
	scaleVal := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	if cfg.AttnScale != 0 {
		scaleVal = float32(cfg.AttnScale)
	}
	scaleT, err := tensor.New([]float32{scaleVal}, []int{1}, dev)
	if err != nil {
		return nil, err
	}
	if cfg.WeightDType == dtype.BF16 {
		// In the all-BF16 forward, every multiply against `scale` happens at
		// BF16. Pre-cast so each layer doesn't pay the cost.
		scaleT, err = ops.Cast(scaleT, dtype.BF16)
		if err != nil {
			return nil, err
		}
	}
	return &Llama{
		Cfg: cfg, TokenEmb: tokEmb, Blocks: blocks, FinalNorm: fn, FinalLN: finalLN, LMHead: head,
		scale: scaleT, scaleDev: dev,
	}, nil
}

// Parameters returns the full list of trainable tensors. Honors the tied LM
// head (the shared embedding appears once).
func (m *Llama) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, m.TokenEmb.Parameters()...)
	for _, b := range m.Blocks {
		ps = append(ps, b.parameters()...)
	}
	if m.FinalLN != nil {
		ps = append(ps, m.FinalLN.Parameters()...)
	} else {
		ps = append(ps, m.FinalNorm.Parameters()...)
	}
	if !m.Cfg.TieEmbed {
		ps = append(ps, m.LMHead.Parameters()...)
	}
	return ps
}

// ParamCount returns the total number of trainable elements.
func (m *Llama) ParamCount() int {
	total := 0
	for _, p := range m.Parameters() {
		total += p.NumElements()
	}
	return total
}

// Forward consumes tokens (B, T) I64 and returns logits (B, T, V) F32. No
// caching — every call recomputes the full attention over the input.
func (m *Llama) Forward(tokens *tensor.Tensor) (*tensor.Tensor, error) {
	return m.ForwardWithCache(tokens, nil)
}

// ForwardWithCache runs the model with an optional KV cache. Pass `cache=nil`
// for a stateless forward (equivalent to Forward); pass a non-nil cache to
// accumulate K/V across calls so the next call can score the new tokens
// against everything seen so far.
//
// Modes:
//
//   - cache.Len == 0  : "prefill". Tokens carry the full prompt; positions
//     are [0..T-1] and a (T, T) causal mask is applied.
//
//   - cache.Len  > 0  : "decode". Tokens carry only the newly generated
//     suffix (typically T=1). Positions are [Len..Len+T-1]; for T=1 no mask
//     is needed since the new token may attend to all cached + itself; for
//     T>1 the new tokens get a causal mask among themselves but attend
//     freely to cached tokens.
//
// After a successful call, cache.Len is incremented by T and cache.K/V hold
// the post-RoPE K and pre-projection V for every layer.
func (m *Llama) ForwardWithCache(tokens *tensor.Tensor, cache *KVCache) (*tensor.Tensor, error) {
	if tokens.Rank() != 2 {
		return nil, fmt.Errorf("Llama.Forward: tokens must be (B, T), got %v", tokens.Shape())
	}
	if tokens.DType() != dtype.I64 {
		return nil, fmt.Errorf("Llama.Forward: tokens must be I64, got %v", tokens.DType())
	}
	h, err := m.TokenEmb.Forward(tokens) // (B, T, D)
	if err != nil {
		return nil, err
	}
	if m.Cfg.EmbedMultiplier != 0 {
		if h, err = mulScalarF32(h, m.Cfg.EmbedMultiplier); err != nil {
			return nil, err
		}
	}
	return m.ForwardEmbeds(h, cache)
}

// Embed returns the token embeddings (B, T, D) for tokens (B, T) I64, in
// the activation dtype. Multimodal front ends splice their own vectors
// into this before calling ForwardEmbeds.
func (m *Llama) Embed(tokens *tensor.Tensor) (*tensor.Tensor, error) {
	return m.TokenEmb.Forward(tokens)
}

// ForwardEmbeds is ForwardWithCache on precomputed input embeddings
// (B, T, D) instead of token ids (HF `inputs_embeds`). Positions, cache
// handling and the returned F32 logits (B, T, V) are identical.
func (m *Llama) ForwardEmbeds(h *tensor.Tensor, cache *KVCache) (*tensor.Tensor, error) {
	if h.Rank() != 3 || h.Dim(2) != m.Cfg.HiddenSize {
		return nil, fmt.Errorf("Llama.ForwardEmbeds: want (B, T, %d), got %v", m.Cfg.HiddenSize, h.Shape())
	}
	B := h.Dim(0)
	T := h.Dim(1)
	dev := h.Device()
	prior := 0
	if cache != nil {
		if len(cache.K) != m.Cfg.NumLayers {
			return nil, fmt.Errorf("Llama.ForwardWithCache: cache has %d layers, model has %d", len(cache.K), m.Cfg.NumLayers)
		}
		prior = cache.Len
	}
	if prior+T > m.Cfg.MaxPosition {
		return nil, fmt.Errorf("Llama.Forward: sequence length %d > MaxPosition %d", prior+T, m.Cfg.MaxPosition)
	}
	var err error
	for i, blk := range m.Blocks {
		h, err = m.runBlock(blk, h, B, T, prior, dev, cache, i)
		if err != nil {
			return nil, err
		}
	}
	if m.FinalLN != nil {
		h, err = m.FinalLN.Forward(h)
	} else {
		h, err = m.FinalNorm.Forward(h)
	}
	if err != nil {
		return nil, err
	}
	if cache != nil && !(m.diffBlock > 0 && !m.diffStore) {
		cache.Len = prior + T
	}
	m.LastHidden = h                   // the LM head's input; an MTP draft head reads it
	logits, err := m.LMHead.Forward(h) // (B, T, V)
	if err != nil {
		return nil, err
	}
	if m.Cfg.LogitsScaling != 0 {
		if logits, err = divScalarF32(logits, m.Cfg.LogitsScaling); err != nil {
			return nil, err
		}
	}
	// In BF16 mode the LM-head output is BF16; cast to F32 so callers (which
	// sample on CPU via MustSlice[float32]) can read it directly. The cost
	// is one (B, T, V) cast per forward — for 31-token prefill at vocab
	// 151936 that's ~5 MB, negligible vs the matmul itself.
	if logits.DType() == dtype.BF16 {
		logits, err = ops.Cast(logits, dtype.F32)
		if err != nil {
			return nil, err
		}
	}
	return logits, nil
}

func (m *Llama) runBlock(blk *llamaBlock, x *tensor.Tensor, B, T, prior int, dev device.Device, cache *KVCache, layerIdx int) (*tensor.Tensor, error) {
	cfg := m.Cfg
	NH, KH, HD := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	g := NH / KH

	var h *tensor.Tensor
	var err error
	if blk.LN1 != nil {
		h, err = blk.LN1.Forward(x)
	} else {
		h, err = blk.N1.Forward(x)
	}
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
	q, err = headsForward(q, B, T, NH, HD)
	if err != nil {
		return nil, err
	}
	k, err = headsForward(k, B, T, KH, HD)
	if err != nil {
		return nil, err
	}
	v, err = headsForward(v, B, T, KH, HD)
	if err != nil {
		return nil, err
	}
	if blk.QNorm != nil { // Qwen3: per-head RMSNorm over HD, before RoPE
		if q, err = blk.QNorm.Forward(q); err != nil {
			return nil, err
		}
		if k, err = blk.KNorm.Forward(k); err != nil {
			return nil, err
		}
	}

	// RoPE positions are [prior..prior+T-1]. The per-token rotation depends
	// only on its absolute index, so cached past K (already rotated) is left
	// alone; only the new K, Q get RoPE.
	pos, err := m.positionsFor(T, prior, dev)
	if err != nil {
		return nil, err
	}
	q, err = m.applyRoPE(q, pos, prior+T, layerIdx, dev)
	if err != nil {
		return nil, err
	}
	if q, err = m.llama4QScale(q, T, prior, dev); err != nil {
		return nil, err
	}
	k, err = m.applyRoPE(k, pos, prior+T, layerIdx, dev)
	if err != nil {
		return nil, err
	}

	// Append-then-attend. With a cache, K and V grow by T along the seqlen
	// axis. Without a cache, they stay at T. kvPrior is the number of cached
	// keys actually present (== prior unless a sliding window trimmed them).
	window := cfg.slidingWindowFor(layerIdx)
	totalT := T
	kvPrior := 0
	if cache != nil {
		if cache.K[layerIdx] != nil {
			kvPrior = cache.K[layerIdx].Dim(2)
			k, err = ops.Concat([]*tensor.Tensor{cache.K[layerIdx], k}, 2)
			if err != nil {
				return nil, err
			}
			v, err = ops.Concat([]*tensor.Tensor{cache.V[layerIdx], v}, 2)
			if err != nil {
				return nil, err
			}
		}
		totalT = kvPrior + T
		if m.diffBlock > 0 && !m.diffStore {
			// Denoising pass: attend over the cache, but leave it untouched.
			goto attend
		}
		cache.K[layerIdx], cache.V[layerIdx] = k, v
		// Rotating cache: keep only the last `window` keys for next time.
		// Attention below still sees the full concat, since the earliest
		// query of this step may reach further back than the last one.
		if window > 0 && totalT > window {
			cache.K[layerIdx], err = trimKV(k, totalT-window, window)
			if err != nil {
				return nil, err
			}
			cache.V[layerIdx], err = trimKV(v, totalT-window, window)
			if err != nil {
				return nil, err
			}
		}
	}

	// GQA: expand K, V to NH heads.
attend:
	k, err = repeatKV(k, g, B, KH, totalT, HD)
	if err != nil {
		return nil, err
	}
	v, err = repeatKV(v, g, B, KH, totalT, HD)
	if err != nil {
		return nil, err
	}

	// Collapse the leading B*NH so the batched matmul stays rank-3.
	q, err = autograd.Reshape(q, B*NH, T, HD)
	if err != nil {
		return nil, err
	}
	k, err = autograd.Reshape(k, B*NH, totalT, HD)
	if err != nil {
		return nil, err
	}
	v, err = autograd.Reshape(v, B*NH, totalT, HD)
	if err != nil {
		return nil, err
	}

	kT, err := autograd.Transpose(k, 1, 2)
	if err != nil {
		return nil, err
	}
	kTC, err := autograd.Contiguous(kT)
	if err != nil {
		return nil, err
	}
	scores, err := autograd.MatMul(q, kTC) // (B*NH, T, totalT)
	if err != nil {
		return nil, err
	}
	scores, err = autograd.Mul(scores, m.scale)
	if err != nil {
		return nil, err
	}
	// Mask: only when there's potential ambiguity among new queries (T > 1)
	// or a sliding window excludes some cached keys. In plain single-token
	// decode (T=1) we simply attend to everything cached.
	if T > 1 || (window > 0 && totalT > window) {
		mask, err := m.queryKeyMask(T, kvPrior, window, dev)
		if err != nil {
			return nil, err
		}
		scores, err = autograd.Add(scores, mask)
		if err != nil {
			return nil, err
		}
	}
	attn, err := autograd.Softmax(scores, -1)
	if err != nil {
		return nil, err
	}
	ctx, err := autograd.MatMul(attn, v) // (B*NH, T, HD)
	if err != nil {
		return nil, err
	}

	ctx, err = autograd.Reshape(ctx, B, NH, T, HD)
	if err != nil {
		return nil, err
	}
	ctx, err = autograd.Permute(ctx, 0, 2, 1, 3)
	if err != nil {
		return nil, err
	}
	ctx, err = autograd.Contiguous(ctx)
	if err != nil {
		return nil, err
	}
	// NumHeads·HeadDim need not equal HiddenSize (Qwen3, Mellum 2.1:
	// 32 × 128 heads on a 2304-wide model); OProj maps back to HiddenSize.
	ctx, err = autograd.Reshape(ctx, B, T, NH*HD)
	if err != nil {
		return nil, err
	}
	attnOut, err := blk.OProj.Forward(ctx)
	if err != nil {
		return nil, err
	}
	if blk.PostAttnNorm != nil {
		if attnOut, err = blk.PostAttnNorm.Forward(attnOut); err != nil {
			return nil, err
		}
	}
	x, err = m.residual(x, attnOut)
	if err != nil {
		return nil, err
	}

	// SwiGLU MLP, or the routed-expert FFN on MoE layers.
	if blk.LN2 != nil {
		h, err = blk.LN2.Forward(x)
	} else {
		h, err = blk.N2.Forward(x)
	}
	if err != nil {
		return nil, err
	}
	if blk.MoE != nil {
		mlpOut, err := blk.MoE.Forward(h)
		if err != nil {
			return nil, err
		}
		if blk.PostMLPNorm != nil {
			if mlpOut, err = blk.PostMLPNorm.Forward(mlpOut); err != nil {
				return nil, err
			}
		}
		return m.residual(x, mlpOut)
	}
	if blk.GateProj == nil { // Nemotron: down(relu(up(h))²)
		up, err := blk.UpProj.Forward(h)
		if err != nil {
			return nil, err
		}
		if up, err = ops.Relu(up); err != nil {
			return nil, err
		}
		if up, err = ops.Sqr(up); err != nil {
			return nil, err
		}
		mlpOut, err := blk.DownProj.Forward(up)
		if err != nil {
			return nil, err
		}
		return m.residual(x, mlpOut)
	}
	gate, err := blk.GateProj.Forward(h)
	if err != nil {
		return nil, err
	}
	gate, err = autograd.SiLU(gate)
	if err != nil {
		return nil, err
	}
	up, err := blk.UpProj.Forward(h)
	if err != nil {
		return nil, err
	}
	mlpIn, err := autograd.Mul(gate, up)
	if err != nil {
		return nil, err
	}
	mlpOut, err := blk.DownProj.Forward(mlpIn)
	if err != nil {
		return nil, err
	}
	if blk.PostMLPNorm != nil {
		if mlpOut, err = blk.PostMLPNorm.Forward(mlpOut); err != nil {
			return nil, err
		}
	}
	return m.residual(x, mlpOut)
}

// llama4QScale applies the Llama 4 query scaling to q (B, NH, T, HD) at
// positions prior … prior+T−1: the factor is formed in F32 and cast to q's
// dtype before the multiply, as transformers' get_llama_4_attn_scale(…)
// .to(query_states.dtype) does. A no-op while every position is below
// Llama4ScalingOrig (the factor is exactly 1 there).
func (m *Llama) llama4QScale(q *tensor.Tensor, T, prior int, dev device.Device) (*tensor.Tensor, error) {
	beta, orig := m.Cfg.Llama4ScalingBeta, m.Cfg.Llama4ScalingOrig
	if beta == 0 || orig <= 0 || prior+T <= orig {
		return q, nil
	}
	f := make([]float32, T)
	for t := range T {
		f[t] = float32(1 + beta*math.Log(1+math.Floor(float64(prior+t)/float64(orig))))
	}
	ft, err := tensor.New(f, []int{1, 1, T, 1}, dev)
	if err != nil {
		return nil, err
	}
	if q.DType() != dtype.F32 {
		if ft, err = ops.Cast(ft, q.DType()); err != nil {
			return nil, err
		}
	}
	return ops.Mul(q, ft)
}

// residual is x + branch, the branch scaled by ResidualMultiplier first
// when set (Granite: residual + hidden · multiplier, rounded per op as
// transformers does in BF16).
func (m *Llama) residual(x, branch *tensor.Tensor) (*tensor.Tensor, error) {
	if r := m.Cfg.ResidualMultiplier; r != 0 {
		var err error
		if branch, err = mulScalarF32(branch, r); err != nil {
			return nil, err
		}
	}
	return autograd.Add(x, branch)
}

// applyRoPE rotates x by position: plain RoPE on Cfg.RopeBase, or the scaled
// table when Cfg.RopeScaling is set (tables built once per device and
// cached). seqLen is the total sequence length after this step (prior + T);
// LongRoPE picks its short or long factor table from it and additionally
// scales the rotated output by its attention factor.
//
// layer selects the per-layer-type scheme (Cfg.ropeFor): sliding-window
// layers use SlidingRopeBase / SlidingRopeScaling when set.
func (m *Llama) applyRoPE(x, pos *tensor.Tensor, seqLen, layer int, dev device.Device) (*tensor.Tensor, error) {
	base, rs, slot := m.Cfg.ropeFor(layer)
	rotDim := m.rotaryDim()
	if rotDim < m.Cfg.HeadDim {
		// Partial rotary (Phi-4-mini): rotate_half runs over the first rotDim
		// dims only; the rest pass through untouched.
		rot, err := x.Narrow(3, 0, rotDim)
		if err != nil {
			return nil, err
		}
		if rot, err = autograd.Contiguous(rot); err != nil {
			return nil, err
		}
		if rot, err = m.applyRoPEDim(rot, pos, seqLen, rotDim, base, rs, slot, dev); err != nil {
			return nil, err
		}
		pass, err := x.Narrow(3, rotDim, m.Cfg.HeadDim-rotDim)
		if err != nil {
			return nil, err
		}
		return ops.Concat([]*tensor.Tensor{rot, pass}, 3)
	}
	return m.applyRoPEDim(x, pos, seqLen, rotDim, base, rs, slot, dev)
}

// rotaryDim is the number of head dims RoPE rotates (even).
func (m *Llama) rotaryDim() int {
	if f := m.Cfg.PartialRotary; f > 0 && f < 1 {
		return int(f*float64(m.Cfg.HeadDim)) &^ 1
	}
	return m.Cfg.HeadDim
}

func (m *Llama) applyRoPEDim(x, pos *tensor.Tensor, seqLen, rotDim int, base float64, rs *RopeScaling, slot int, dev device.Device) (*tensor.Tensor, error) {
	if rs == nil {
		return ops.RoPE(x, pos, base)
	}
	if m.invFreqDev == nil || !m.invFreqDev.SameAs(dev) {
		m.rope = [2]ropeTables{}
		m.invFreqDev = dev
	}
	tb := &m.rope[slot]
	table := &tb.invFreq
	long := rs.Type == "longrope" && seqLen > rs.OriginalMaxPositions
	if long {
		table = &tb.invFreqLong
	}
	if *table == nil {
		t, err := rs.invFreqTensor(rotDim, base, long, dev)
		if err != nil {
			return nil, err
		}
		*table = t
	}
	out, err := ops.RoPEWithInvFreq(x, pos, *table)
	if err != nil {
		return nil, err
	}
	af := rs.AttentionFactor()
	if af == 1 {
		return out, nil
	}
	// The factor multiplies in F32. transformers folds it into cos / sin in
	// F32 before casting them; a BF16 scalar would round it (Mellum 2.1:
	// 1.27726 → 1.2734, 0.3 % low on q and on k, so every full-layer score
	// came out 0.6 % small).
	if tb.attn == nil {
		t, err := tensor.New([]float32{float32(af)}, []int{1}, dev)
		if err != nil {
			return nil, err
		}
		tb.attn = t
	}
	if out.DType() == dtype.F32 {
		return ops.Mul(out, tb.attn)
	}
	dt := out.DType()
	o32, err := ops.Cast(out, dtype.F32)
	if err != nil {
		return nil, err
	}
	if o32, err = ops.Mul(o32, tb.attn); err != nil {
		return nil, err
	}
	return ops.Cast(o32, dt)
}

// headsForward reshapes (B, T, H*HD) to (B, H, T, HD) and contiguifies.
func headsForward(x *tensor.Tensor, B, T, H, HD int) (*tensor.Tensor, error) {
	r, err := autograd.Reshape(x, B, T, H, HD)
	if err != nil {
		return nil, err
	}
	p, err := autograd.Permute(r, 0, 2, 1, 3)
	if err != nil {
		return nil, err
	}
	return autograd.Contiguous(p)
}

// repeatKV expands x of shape (B, KH, T, HD) to (B, KH*g, T, HD) by repeating
// each KV head g times so it can match the query-head count under GQA.
func repeatKV(x *tensor.Tensor, g, B, KH, T, HD int) (*tensor.Tensor, error) {
	if g == 1 {
		return x, nil
	}
	x5, err := autograd.Reshape(x, B, KH, 1, T, HD)
	if err != nil {
		return nil, err
	}
	parts := make([]*tensor.Tensor, g)
	for i := range parts {
		parts[i] = x5
	}
	cat, err := ops.Concat(parts, 2) // (B, KH, g, T, HD)
	if err != nil {
		return nil, err
	}
	return autograd.Reshape(cat, B, KH*g, T, HD)
}

// trimKV returns a contiguous copy of the last `length` entries along the
// sequence axis of a (B, KH, L, HD) tensor.
func trimKV(x *tensor.Tensor, start, length int) (*tensor.Tensor, error) {
	n, err := x.Narrow(2, start, length)
	if err != nil {
		return nil, err
	}
	return autograd.Contiguous(n)
}

// queryKeyMask returns a (T, prior+T) mask where entry (i, j) is -∞ when key
// position j > prior+i (i.e. future tokens) or, with window > 0, when
// j <= prior+i-window (sliding-window attention). When prior == 0 and
// window == 0 this collapses to the standard square causal mask. Cached on
// the model. Allocated at the activation dtype (F32 for the regular path,
// BF16 for full BF16 inference).
func (m *Llama) queryKeyMask(T, prior, window int, dev device.Device) (*tensor.Tensor, error) {
	totalT := prior + T
	block := m.diffBlock
	if m.maskCache != nil && m.maskTotal == totalT && m.maskPrior == prior && m.maskT == T && m.maskWindow == window && m.maskBlock == block && m.maskDev != nil && m.maskDev.SameAs(dev) {
		return m.maskCache, nil
	}
	negInf := float32(math.Inf(-1))
	data := make([]float32, T*totalT)
	for i := range T {
		pos := prior + i
		if block > 0 {
			// Block diffusion: key j is visible iff its block is not after the
			// query's block (bidirectional inside a block, causal across).
			for j := (pos/block + 1) * block; j < totalT; j++ {
				data[i*totalT+j] = negInf
			}
			continue
		}
		for j := pos + 1; j < totalT; j++ {
			data[i*totalT+j] = negInf
		}
		if window > 0 {
			for j := 0; j <= pos-window; j++ {
				data[i*totalT+j] = negInf
			}
		}
	}
	mask, err := tensor.New(data, []int{T, totalT}, dev)
	if err != nil {
		return nil, err
	}
	if m.Cfg.WeightDType == dtype.BF16 {
		mask, err = ops.Cast(mask, dtype.BF16)
		if err != nil {
			return nil, err
		}
	}
	m.maskCache = mask
	m.maskTotal = totalT
	m.maskPrior = prior
	m.maskT = T
	m.maskWindow = window
	m.maskBlock = block
	m.maskDev = dev
	return mask, nil
}

// positionsFor returns (and caches) [offset..offset+S-1] as an (S,) I64
// tensor on dev. The cache hits whenever the same (S, offset, dev) repeats.
func (m *Llama) positionsFor(S, offset int, dev device.Device) (*tensor.Tensor, error) {
	if m.posCache != nil && m.posS == S && m.posOff == offset && m.posDev != nil && m.posDev.SameAs(dev) {
		return m.posCache, nil
	}
	data := make([]int64, S)
	for i := range S {
		data[i] = int64(offset + i)
	}
	pos, err := tensor.New(data, []int{S}, dev)
	if err != nil {
		return nil, err
	}
	m.posCache = pos
	m.posS = S
	m.posOff = offset
	m.posDev = dev
	return pos, nil
}
