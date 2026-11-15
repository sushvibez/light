package models

import (
	"fmt"
	"math"
	"sort"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// DeepseekV41Config is DeepSeek-V4.1's text config (the released
// config.json's text_config; the field names are transformers').
type DeepseekV41Config struct {
	VocabSize       int     `json:"vocab_size"`
	HiddenSize      int     `json:"hidden_size"`
	MoEIntermediate int     `json:"moe_intermediate_size"`
	NumLayers       int     `json:"num_hidden_layers"`
	NumHeads        int     `json:"num_attention_heads"`
	HeadDim         int     `json:"head_dim"`
	RopeHeadDim     int     `json:"qk_rope_head_dim"`
	QLoraRank       int     `json:"q_lora_rank"`
	OLoraRank       int     `json:"o_lora_rank"`
	OGroups         int     `json:"o_groups"`
	SwiGLULimit     float64 `json:"swiglu_limit"`
	RMSNormEps      float64 `json:"rms_norm_eps"`
	RopeTheta       float64 `json:"rope_theta"`
	RopeScaling     *struct {
		Type        string  `json:"rope_type"`
		Factor      float64 `json:"factor"`
		BetaFast    float64 `json:"beta_fast"`
		BetaSlow    float64 `json:"beta_slow"`
		OriginalMax int     `json:"original_max_position_embeddings"`
	} `json:"rope_scaling"`
	NRoutedExperts        int     `json:"n_routed_experts"`
	NSharedExperts        int     `json:"n_shared_experts"`
	ExpertsPerTok         int     `json:"num_experts_per_tok"`
	ScoringFunc           string  `json:"scoring_func"`
	NormTopkProb          bool    `json:"norm_topk_prob"`
	RoutedScaling         float64 `json:"routed_scaling_factor"`
	SlidingWindow         int     `json:"sliding_window"`
	CompressRatios        []int   `json:"compress_ratios"`
	CompressRopeTheta     float64 `json:"compress_rope_theta"`
	KVSourceLayers        []int   `json:"kv_source_layer_ids"`
	IndexSourceLayers     []int   `json:"index_source_layer_ids"`
	IndexNHeads           int     `json:"index_n_heads"`
	IndexHeadDim          int     `json:"index_head_dim"`
	IndexTopk             int     `json:"index_topk"`
	CandidateSourceLayer  *int    `json:"candidate_source_layer_id"`
	CandidateTopkBlocks   int     `json:"candidate_topk_blocks"`
	CandidateBlockSize    int     `json:"candidate_block_size"`
	HCMult                int     `json:"hc_mult"`
	HCSinkhornIters       int     `json:"hc_sinkhorn_iters"`
	HCEps                 float64 `json:"hc_eps"`
	EngramLayerIDs        []int   `json:"engram_layer_ids"`
	EngramNumEmbeddings   []int64 `json:"engram_num_embeddings"`
	EngramMaxNgram        int     `json:"engram_max_ngram_size"`
	EngramVocabSize       int64   `json:"engram_vocab_size"`
	EngramNHeads          int     `json:"engram_n_heads"`
	EngramHeadDim         int     `json:"engram_head_dim"`
	EngramPadTokenID      int     `json:"engram_pad_token_id"`
	EngramCompressedVocab int64   `json:"engram_compressed_vocab_size"`
	TieEmbeddings         bool    `json:"tie_word_embeddings"`
}

// DeepseekV41 is DeepSeek-V4.1-Flash's language model (DeepSeek's
// inference model.py Transformer, text path): hc_mult hyper-connection
// streams; latent single-KV attention with a low-rank query, attention
// sinks, RoPE on the last qk_rope_head_dim dims (the output de-rotated) and
// a grouped low-rank output projection, attending over a sliding window plus
// - at compress_ratio > 0 - compressed KV positions chosen by a ReLU indexer
// (two-level from the candidate source layer on: candidate blocks, then the
// top index_topk positions), the compressed KV and index keys produced by
// the kv_source layers and shared by the layers after them; sqrtsoftplus-
// routed experts plus a shared expert (SwiGLU clamped at swiglu_limit); and
// engram n-gram hash lookups gated into the residual streams.
//
// ForwardWithCache decodes incrementally (DeepseekV41Cache). Light computes in F32 from the dequantized
// FP8 / FP4 weights; the released kernels' activation fake-quantization
// (FP8 activations into the GEMMs, FP8 / FP4 KV) is not reproduced. Not
// ported: the vision encoder and the DSpark speculative heads (mtp.*).
type DeepseekV41 struct {
	Cfg    DeepseekV41Config
	Embed  *tensor.Tensor // (V, D)
	Layers []*v41Layer
	NormW  *tensor.Tensor
	Head   *nn.Linear
	engram *v41Engram
	hc     *GLM5 // the hyper-connection helpers (hcWeights / hcCombine / rms)
	dev    device.Device
}

type v41Expert struct{ W1, W2, W3 *nn.Linear }

type v41Layer struct {
	ID, Ratio             int
	KVSource, IndexSource bool
	AttnHC, FFNHC         glm5HC
	AttnNormW, FFNNormW   *tensor.Tensor
	// Attention.
	WqA, WqB, Wkv, WoB *nn.Linear
	WoA                []*nn.Linear // one per output group
	QNormW, KVNormW    *tensor.Tensor
	Sink               []float32
	// Compressor (kv sources).
	CWkv, CWgate *nn.Linear
	CNormW       *tensor.Tensor
	// Indexer (index sources; Wk / k_norm on kv sources).
	IWqB, IWeights, IWk *nn.Linear
	IKNormW             *tensor.Tensor
	// MoE.
	GateW    *tensor.Tensor // (E, D)
	GateBias []float32
	Shared   *v41Expert
	Experts  []*v41Expert
	// Engram (EngramIdx ≥ 0): the table stays on the host.
	EngramIdx      int
	EWkv           *nn.Linear
	EWeight        *tensor.Tensor // q_weight · k_weight, (1, 1, hc, D)
	ETable, EScale *tensor.Tensor
}

// v41Shared is what attention layers hand down the stack (the reference's
// SharedAttentionRuntime): the compressed KV and index keys of the last kv
// source, the compressed positions each query attends to from the last
// index source, and the candidate positions of the candidate source.
type v41Shared struct {
	compKV, indexK *tensor.Tensor
	idxs           [][]int // per (b, s): compressed positions
	idxRatio       int
	cand           []bool // (B·S·C)
	candC          int
}

func (m *DeepseekV41) candidateSource() int {
	if m.Cfg.CandidateSourceLayer == nil {
		return -1
	}
	return *m.Cfg.CandidateSourceLayer
}

// ropeFreqs is precompute_freqs_cis's frequencies (F32 like the reference):
// compressed layers rotate at compress_rope_theta with YaRN, pure sliding-
// window layers at rope_theta without.
func (m *DeepseekV41) ropeFreqs(compressed bool) []float32 {
	cfg := m.Cfg
	dim := cfg.RopeHeadDim
	base, orig := cfg.RopeTheta, 0
	factor, betaFast, betaSlow := 1.0, 32.0, 1.0
	if rs := cfg.RopeScaling; rs != nil {
		if rs.Type == "yarn" {
			orig = rs.OriginalMax
		}
		factor, betaFast, betaSlow = rs.Factor, rs.BetaFast, rs.BetaSlow
	}
	if compressed {
		base = cfg.CompressRopeTheta
	} else {
		orig = 0
	}
	f := make([]float32, dim/2)
	for i := range f {
		e := float32(float32(2*i) / float32(dim))
		f[i] = float32(1 / float32(math.Pow(base, float64(e))))
	}
	if orig > 0 {
		corr := func(rot float64) float64 {
			return float64(dim) * math.Log(float64(orig)/(rot*2*math.Pi)) / (2 * math.Log(base))
		}
		low := max(math.Floor(corr(betaFast)), 0)
		high := min(math.Ceil(corr(betaSlow)), float64(dim-1))
		for i := range f {
			ramp := float32((float64(i) - low) / max(high-low, 1e-3))
			ramp = min(max(ramp, 0), 1)
			smooth := 1 - ramp
			f[i] = f[i]/float32(factor)*(1-smooth) + f[i]*smooth
		}
	}
	return f
}

// rope rotates the last qk_rope_head_dim dims of x (B, S, d) or
// (B, S, H, d) as adjacent (real, imaginary) pairs at positions pos (one
// per S); inverse conjugates the rotation.
func (m *DeepseekV41) rope(o *iop, x *tensor.Tensor, pos []int, freqs []float32, inverse bool) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	dims := x.Shape().Dims()
	B, S, d := dims[0], dims[1], dims[len(dims)-1]
	M := 1
	if len(dims) == 4 {
		M = dims[2]
	}
	rd, half := m.Cfg.RopeHeadDim, m.Cfg.RopeHeadDim/2
	cs, sn := make([]float32, S*M*half), make([]float32, S*M*half)
	for s, p := range pos {
		for i := range half {
			a := float64(float32(p) * freqs[i])
			c, si := float32(math.Cos(a)), float32(math.Sin(a))
			if inverse {
				si = -si
			}
			for h := range M {
				cs[(s*M+h)*half+i], sn[(s*M+h)*half+i] = c, si
			}
		}
	}
	cosT := o.do(func() (*tensor.Tensor, error) { return tensor.New(cs, []int{1, S, M * half, 1}, m.dev) })
	sinT := o.do(func() (*tensor.Tensor, error) { return tensor.New(sn, []int{1, S, M * half, 1}, m.dev) })
	x4 := o.reshape(x, B, S, M, d)
	rot := o.reshape(o.narrow(x4, 3, d-rd, rd), B, S, M*half, 2)
	x0, x1 := o.narrow(rot, 3, 0, 1), o.narrow(rot, 3, 1, 1)
	out0 := o.sub(o.mul(x0, cosT), o.mul(x1, sinT))
	out1 := o.add(o.mul(x0, sinT), o.mul(x1, cosT))
	y := o.reshape(o.cat(3, out0, out1), B, S, M, rd)
	if d > rd {
		y = o.cat(3, o.narrow(x4, 3, 0, d-rd), y)
	}
	return o.reshape(y, dims...)
}

func (m *DeepseekV41) rms(o *iop, x, w *tensor.Tensor) *tensor.Tensor { return m.hc.rms(o, x, w) }

func v41Positions(n, stride int) []int { return v41Span(0, n, stride) }

// v41Span is the positions (start+i)·stride for i < n.
func v41Span(start, n, stride int) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = (start + i) * stride
	}
	return p
}

func (o *iop) sum(t *tensor.Tensor, axis int) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return ops.Sum(t, ops.ReduceOptions{Axes: []int{axis}}) })
}

func (o *iop) matmul(a, b *tensor.Tensor) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return ops.MatMul(a, b) })
}

func (o *iop) toHost(t *tensor.Tensor) []float32 {
	if o.err != nil {
		return nil
	}
	c, err := t.To(device.CPU())
	if err == nil {
		c, err = c.Contiguous()
	}
	if err != nil {
		o.err = err
		return nil
	}
	return tensor.MustSlice[float32](c)
}

// compress pools ratio consecutive tokens into one latent (B, C, head_dim)
// with a softmax gate (a plain projection at ratio 1), RMS-normed, before
// RoPE: the groups x (B, S, D) - after P earlier tokens, whose incomplete
// group's projections kc holds - completes, and the index of the first.
// Nil when none completes.
func (m *DeepseekV41) compress(o *iop, ly *v41Layer, x *tensor.Tensor, kc *v41LayerCache, P int) (*tensor.Tensor, int) {
	B, S, hd, r := x.Dim(0), x.Dim(1), m.Cfg.HeadDim, ly.Ratio
	if o.err != nil {
		return nil, 0
	}
	if r == 1 {
		return m.rms(o, o.lin(ly.CWkv, x), ly.CNormW), P
	}
	kv, g := o.lin(ly.CWkv, x), o.lin(ly.CWgate, x)
	if kc != nil && kc.pendKV != nil {
		kv, g = o.cat(1, kc.pendKV, kv), o.cat(1, kc.pendGate, g)
	}
	if o.err != nil {
		return nil, 0
	}
	n := kv.Dim(1)
	C0, C := (P+S-n)/r, n/r
	if kc != nil {
		kc.pendKV, kc.pendGate = nil, nil
		if rest := n - C*r; rest > 0 {
			kc.pendKV, kc.pendGate = o.narrow(kv, 1, C*r, rest), o.narrow(g, 1, C*r, rest)
		}
	}
	if C == 0 {
		return nil, C0
	}
	group := func(t *tensor.Tensor) *tensor.Tensor {
		g := o.reshape(o.narrow(t, 1, 0, C*r), B, C, r, hd)
		return o.reshape(o.perm(g, 0, 1, 3, 2), B, C, hd, r)
	}
	kv, sc := group(kv), group(g)
	p := o.do(func() (*tensor.Tensor, error) { return ops.Softmax(sc, 3) })
	pooled := o.sum(o.mul(kv, p), 3)
	return m.rms(o, o.reshape(pooled, B, C, hd), ly.CNormW), C0
}

// v41Append is old ++ new along the sequence (either may be nil).
func v41Append(o *iop, old, new *tensor.Tensor) *tensor.Tensor {
	switch {
	case old == nil:
		return new
	case new == nil:
		return old
	}
	return o.cat(1, old, new)
}

// indexer scores the compressed positions for every query and keeps the
// best index_topk visible ones (two-level after the candidate source).
// latent is this layer's own (pre-RoPE) newly complete latents from
// compressed position C0 on when it is a kv source; x holds the tokens
// after P earlier ones.
func (m *DeepseekV41) indexer(o *iop, ly *v41Layer, x, qr, latent *tensor.Tensor, C0 int, st *v41Shared, kc *v41LayerCache, P int) [][]int {
	if o.err != nil {
		return nil
	}
	cfg := m.Cfg
	B, S, r := x.Dim(0), x.Dim(1), ly.Ratio
	nh, dh, C := cfg.IndexNHeads, cfg.IndexHeadDim, (P+S)/ly.Ratio
	freqs := m.ropeFreqs(true)
	if ly.IWk != nil { // owns the index keys
		var ik *tensor.Tensor
		if latent != nil {
			ik = m.rms(o, o.lin(ly.IWk, latent), ly.IKNormW)
			ik = m.rope(o, ik, v41Span(C0, latent.Dim(1), r), freqs, false)
		}
		if kc != nil {
			ik = v41Append(o, kc.indexK, ik)
			kc.indexK = ik
		}
		if ik != nil {
			st.indexK = ik
		}
	}
	if o.err != nil {
		return nil
	}
	if st.indexK == nil || st.indexK.Dim(1) != C {
		o.err = fmt.Errorf("deepseek_v41 layer %d: no index keys for %d compressed positions", ly.ID, C)
		return nil
	}
	q := m.rope(o, o.reshape(o.lin(ly.IWqB, qr), B, S, nh, dh), v41Span(P, S, 1), freqs, false)
	w := o.affine(o.lin(ly.IWeights, x), 1/math.Sqrt(float64(dh))/math.Sqrt(float64(nh)), 0)
	kT := o.do(func() (*tensor.Tensor, error) {
		t, err := st.indexK.Transpose(1, 2)
		if err != nil {
			return nil, err
		}
		return t.Contiguous()
	})
	sc := o.matmul(o.reshape(q, B, S*nh, dh), kT)
	sc = o.sum(o.mul(o.reshape(o.un(ops.Relu, sc), B, S, nh, C), o.reshape(w, B, S, nh, 1)), 2)
	v := o.toHost(sc)
	if o.err != nil {
		return nil
	}
	ninf := float32(math.Inf(-1))
	cs := m.candidateSource()
	isSrc, usesCand := ly.ID == cs, cs >= 0 && cs < ly.ID
	if isSrc {
		st.cand, st.candC = make([]bool, B*S*C), C
	} else if usesCand && (st.cand == nil || st.candC != C) {
		o.err = fmt.Errorf("deepseek_v41 layer %d: no candidate blocks for %d compressed positions", ly.ID, C)
		return nil
	}
	k := min(cfg.IndexTopk, C)
	out := make([][]int, B*S)
	order := make([]int, C)
	for q := range B * S {
		row := v[q*C : (q+1)*C]
		clen := (P + q%S + 1) / r
		for t := clen; t < C; t++ {
			row[t] = ninf
		}
		if isSrc {
			m.candidates(row, clen, st.cand[q*C:(q+1)*C])
		} else if usesCand {
			for t, keep := range st.cand[q*C : (q+1)*C] {
				if !keep {
					row[t] = ninf
				}
			}
		}
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool { return row[order[a]] > row[order[b]] })
		var sel []int
		for _, t := range order[:k] {
			if row[t] > ninf { // the reference's leftover -inf picks fall outside compress_lens
				sel = append(sel, t)
			}
		}
		sort.Ints(sel)
		out[q] = sel
	}
	return out
}

// candidates is select_candidate_blocks for one query: the
// candidate_topk_blocks blocks with the best position (the block holding
// the newest visible position pinned in), as a per-position mask.
func (m *DeepseekV41) candidates(row []float32, clen int, keep []bool) {
	bs, C := m.Cfg.CandidateBlockSize, len(row)
	nb := (C + bs - 1) / bs
	sc := make([]float32, nb)
	for b := range sc {
		sc[b] = float32(math.Inf(-1))
		for t := b * bs; t < min((b+1)*bs, C); t++ {
			sc[b] = max(sc[b], row[t])
		}
	}
	if clen > 0 {
		sc[(clen-1)/bs] = float32(math.Inf(1))
	}
	order := make([]int, nb)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return sc[order[a]] > sc[order[b]] })
	kept := make([]bool, nb)
	for _, b := range order[:min(m.Cfg.CandidateTopkBlocks, nb)] {
		kept[b] = sc[b] > float32(math.Inf(-1))
	}
	for t := range keep {
		keep[t] = kept[t/bs]
	}
}

// attention is one layer's attention on x (B, S, D), the normed sublayer
// input, for the tokens after P earlier ones (kc: their state, or nil).
func (m *DeepseekV41) attention(o *iop, ly *v41Layer, x *tensor.Tensor, st *v41Shared, kc *v41LayerCache, P int) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	cfg := m.Cfg
	B, S := x.Dim(0), x.Dim(1)
	H, hd := cfg.NumHeads, cfg.HeadDim
	freqs := m.ropeFreqs(ly.Ratio > 0)
	pos := v41Span(P, S, 1)
	qr := m.rms(o, o.lin(ly.WqA, x), ly.QNormW)
	q := m.rope(o, o.reshape(o.lin(ly.WqB, qr), B, S, H, hd), pos, freqs, false)
	keys := m.rope(o, m.rms(o, o.lin(ly.Wkv, x), ly.KVNormW), pos, freqs, false)
	K0 := P // the position of keys' first row
	if kc != nil {
		if kc.keys != nil {
			keys, K0 = o.cat(1, kc.keys, keys), kc.start
		}
		// Later queries see only the last sliding_window - 1 of these.
		kc.keys = nil
		if keep := min(P+S-K0, cfg.SlidingWindow-1); keep > 0 && o.err == nil {
			kc.keys, kc.start = o.narrow(keys, 1, P+S-K0-keep, keep), P+S-keep
		}
	}
	if o.err != nil {
		return nil
	}
	W := keys.Dim(1)
	C := 0
	if r := ly.Ratio; r > 0 {
		C = (P + S) / r
		var latent *tensor.Tensor
		C0 := 0
		if ly.KVSource {
			latent, C0 = m.compress(o, ly, x, kc, P)
		}
		if ly.IndexSource {
			st.idxs, st.idxRatio = make([][]int, B*S), r
			if C > 0 {
				st.idxs = m.indexer(o, ly, x, qr, latent, C0, st, kc, P)
			}
		}
		if ly.KVSource {
			var ckv *tensor.Tensor
			if latent != nil {
				ckv = m.rope(o, latent, v41Span(C0, latent.Dim(1), r), freqs, false)
			}
			if kc != nil {
				ckv = v41Append(o, kc.compKV, ckv)
				kc.compKV = ckv
			}
			if ckv != nil {
				st.compKV = ckv
			}
		}
		if o.err != nil {
			return nil
		}
		if st.idxRatio != r || len(st.idxs) != B*S {
			o.err = fmt.Errorf("deepseek_v41 layer %d: no compressed top-k from an index source at ratio %d", ly.ID, r)
			return nil
		}
		if C > 0 {
			if st.compKV == nil || st.compKV.Dim(1) != C {
				o.err = fmt.Errorf("deepseek_v41 layer %d: no compressed KV for %d positions", ly.ID, C)
				return nil
			}
			keys = o.cat(1, keys, st.compKV)
		}
	}
	T := W + C
	ninf := float32(math.Inf(-1))
	mv := make([]float32, B*S*T)
	for i := range mv {
		mv[i] = ninf
	}
	for b := range B {
		for s := range S {
			row := mv[(b*S+s)*T : (b*S+s+1)*T]
			for p := max(K0, P+s-cfg.SlidingWindow+1); p <= P+s; p++ {
				row[p-K0] = 0
			}
			if C > 0 {
				for _, t := range st.idxs[b*S+s] {
					row[W+t] = 0
				}
			}
		}
	}
	sv := make([]float32, B*S*H)
	for i := range sv {
		sv[i] = ly.Sink[i%H]
	}
	mask := o.do(func() (*tensor.Tensor, error) { return tensor.New(mv, []int{B, S, 1, T}, m.dev) })
	sink := o.do(func() (*tensor.Tensor, error) { return tensor.New(sv, []int{B, S, H, 1}, m.dev) })
	kT := o.do(func() (*tensor.Tensor, error) {
		t, err := keys.Transpose(1, 2)
		if err != nil {
			return nil, err
		}
		return t.Contiguous()
	})
	sc := o.matmul(o.reshape(q, B, S*H, hd), kT)
	sc = o.cat(3, o.add(o.affine(o.reshape(sc, B, S, H, T), 1/math.Sqrt(float64(hd)), 0), mask), sink)
	p := o.do(func() (*tensor.Tensor, error) { return ops.Softmax(sc, 3) })
	p = o.reshape(o.narrow(p, 3, 0, T), B, S*H, T)
	att := o.matmul(p, o.reshape(keys, B, T, hd))
	att = m.rope(o, o.reshape(att, B, S, H, hd), pos, freqs, true)
	G := cfg.OGroups
	dg := H * hd / G
	og := o.reshape(att, B, S, G, dg)
	parts := make([]*tensor.Tensor, G)
	for g := range G {
		parts[g] = o.lin(ly.WoA[g], o.reshape(o.narrow(og, 2, g, 1), B, S, dg))
	}
	return o.lin(ly.WoB, o.cat(2, parts...))
}

// expert is w2(silu(clamp(w1·x, max=L)) · clamp(w3·x, ±L) · weight).
func (m *DeepseekV41) expert(o *iop, e *v41Expert, x, weight *tensor.Tensor) *tensor.Tensor {
	L := m.Cfg.SwiGLULimit
	g, u := o.lin(e.W1, x), o.lin(e.W3, x)
	if L > 0 {
		g = o.do(func() (*tensor.Tensor, error) { return ops.Clamp(g, math.Inf(-1), L) })
		u = o.do(func() (*tensor.Tensor, error) { return ops.Clamp(u, -L, L) })
	}
	a := o.mul(o.un(ops.SiLU, g), u)
	if weight != nil {
		a = o.mul(a, weight)
	}
	return o.lin(e.W2, a)
}

// moe is the routed experts (host routing: scores by scoring_func, the
// correction bias picks, the unbiased scores weigh) plus the shared expert.
func (m *DeepseekV41) moe(o *iop, ly *v41Layer, x *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	cfg := m.Cfg
	B, S, D := x.Dim(0), x.Dim(1), x.Dim(2)
	T := B * S
	flat := o.reshape(x, T, D)
	logits := o.do(func() (*tensor.Tensor, error) {
		wt, err := ly.GateW.Transpose(0, 1)
		if err != nil {
			return nil, err
		}
		return ops.MatMul(flat, wt)
	})
	lv := o.toHost(logits)
	if o.err != nil {
		return nil
	}
	E, K := cfg.NRoutedExperts, cfg.ExpertsPerTok
	type hit struct {
		tok int
		w   float32
	}
	byExpert := make([][]hit, E)
	sc, idx := make([]float32, E), make([]int, E)
	for t := range T {
		row := lv[t*E : (t+1)*E]
		switch cfg.ScoringFunc {
		case "softmax":
			mx := math.Inf(-1)
			for _, l := range row {
				mx = max(mx, float64(l))
			}
			var s float64
			for e, l := range row {
				sc[e] = float32(math.Exp(float64(l) - mx))
				s += float64(sc[e])
			}
			for e := range sc {
				sc[e] = float32(float64(sc[e]) / s)
			}
		case "sigmoid":
			for e, l := range row {
				sc[e] = float32(1 / (1 + math.Exp(-float64(l))))
			}
		default: // sqrtsoftplus (torch softplus: linear above 20)
			for e, l := range row {
				sp := float64(l)
				if sp <= 20 {
					sp = math.Log1p(math.Exp(sp))
				}
				sc[e] = float32(math.Sqrt(sp))
			}
		}
		for e := range idx {
			idx[e] = e
		}
		sort.SliceStable(idx, func(a, b int) bool { return sc[idx[a]]+ly.GateBias[idx[a]] > sc[idx[b]]+ly.GateBias[idx[b]] })
		var sum float32
		for _, e := range idx[:K] {
			sum += sc[e]
		}
		for _, e := range idx[:K] {
			w := sc[e]
			if cfg.NormTopkProb && K > 1 {
				w /= sum + 1e-20
			}
			byExpert[e] = append(byExpert[e], hit{t, w * float32(cfg.RoutedScaling)})
		}
	}
	out := m.expert(o, ly.Shared, flat, nil)
	for e, hs := range byExpert {
		if len(hs) == 0 || o.err != nil {
			continue
		}
		tok := make([]int64, len(hs))
		wv := make([]float32, len(hs))
		for i, h := range hs {
			tok[i], wv[i] = int64(h.tok), h.w
		}
		ti := o.do(func() (*tensor.Tensor, error) { return tensor.New(tok, []int{len(tok)}, m.dev) })
		wt := o.do(func() (*tensor.Tensor, error) { return tensor.New(wv, []int{len(wv), 1}, m.dev) })
		xs := o.do(func() (*tensor.Tensor, error) { return ops.IndexSelect(flat, ti) })
		y := m.expert(o, ly.Experts[e], xs, wt)
		// Scatter back with a one-hot (T, n) product.
		oh := o.do(func() (*tensor.Tensor, error) {
			v := make([]float32, T*len(tok))
			for i, t := range tok {
				v[int(t)*len(tok)+i] = 1
			}
			return tensor.New(v, []int{T, len(tok)}, m.dev)
		})
		out = o.add(out, o.do(func() (*tensor.Tensor, error) { return ops.MatMul(oh, y) }))
	}
	return o.reshape(out, B, S, D)
}

// engramApply adds the gated n-gram value into the streams (B, S, N, D):
// the hashed rows project to one key per stream plus a shared value; each
// stream's gate is sigmoid(signed √) of its normalised dot with its key.
func (m *DeepseekV41) engramApply(o *iop, ly *v41Layer, streams *tensor.Tensor, ids []int64) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	B, S, N, D := streams.Dim(0), streams.Dim(1), streams.Dim(2), streams.Dim(3)
	rows, err := v41EngramRows(ly.ETable, ly.EScale, ids)
	if err != nil {
		o.err = err
		return nil
	}
	emb := o.do(func() (*tensor.Tensor, error) { return tensor.New(rows, []int{B, S, len(rows) / (B * S)}, m.dev) })
	kv := o.lin(ly.EWkv, emb)
	key := o.reshape(o.narrow(kv, 2, 0, N*D), B, S, N, D)
	val := o.reshape(o.narrow(kv, 2, N*D, D), B, S, 1, D)
	ones, err := m.hc.onesOf(D)
	if err != nil {
		o.err = err
		return nil
	}
	eps := m.Cfg.RMSNormEps
	hn := o.do(func() (*tensor.Tensor, error) { return ops.RMSNorm(streams, ones, eps) })
	kn := o.do(func() (*tensor.Tensor, error) { return ops.RMSNorm(key, ones, eps) })
	dot := o.sum(o.mul(o.mul(hn, kn), ly.EWeight), 3)
	dv := o.toHost(dot)
	if o.err != nil {
		return nil
	}
	gv := make([]float32, len(dv))
	for i, d := range dv {
		d64 := float64(d) / math.Sqrt(float64(D))
		g := math.Copysign(math.Sqrt(max(math.Abs(d64), 1e-6)), d64)
		gv[i] = float32(1 / (1 + math.Exp(-g)))
	}
	gate := o.do(func() (*tensor.Tensor, error) { return tensor.New(gv, []int{B, S, N, 1}, m.dev) })
	return o.add(streams, o.mul(gate, val))
}

// hcPre collapses the streams (B, S, N, D) with per-token weights (B, S, N, 1).
func (m *DeepseekV41) hcPre(o *iop, streams, pre *tensor.Tensor) *tensor.Tensor {
	return o.sum(o.mul(streams, pre), 2)
}

// DeepseekV41Cache is the decode state: per layer the last
// sliding_window - 1 keys, a kv source's compressed KV, index keys and the
// projections of its incomplete group, plus the token history (engram
// n-grams).
type DeepseekV41Cache struct {
	pos    int
	tokens [][]int64 // per batch row
	layers []v41LayerCache
}

type v41LayerCache struct {
	keys             *tensor.Tensor // (B, n, hd) at positions start..
	start            int
	pendKV, pendGate *tensor.Tensor
	compKV, indexK   *tensor.Tensor
}

// NewCache returns an empty decode cache.
func (m *DeepseekV41) NewCache() *DeepseekV41Cache {
	return &DeepseekV41Cache{layers: make([]v41LayerCache, len(m.Layers))}
}

// Pos is the number of tokens the cache has seen.
func (c *DeepseekV41Cache) Pos() int { return c.pos }

// Forward returns the logits (B, S, V) of every position for tokens (B, S).
func (m *DeepseekV41) Forward(tokens *tensor.Tensor) (*tensor.Tensor, error) {
	return m.ForwardWithCache(tokens, nil)
}

// ForwardWithCache returns the logits (B, S, V) of tokens (B, S) following
// the cache's (nil: none), and extends the cache with them.
func (m *DeepseekV41) ForwardWithCache(tokens *tensor.Tensor, cache *DeepseekV41Cache) (*tensor.Tensor, error) {
	o := &iop{}
	cfg := m.Cfg
	if tokens.Rank() != 2 {
		return nil, fmt.Errorf("deepseek_v41: tokens %v, want (B, S)", tokens.Shape())
	}
	B, S := tokens.Dim(0), tokens.Dim(1)
	D, N := cfg.HiddenSize, cfg.HCMult
	P := 0
	if cache != nil {
		P = cache.pos
		if P > 0 && len(cache.tokens) != B {
			return nil, fmt.Errorf("deepseek_v41: batch %d, the cache holds %d", B, len(cache.tokens))
		}
	}
	var hashes [][]int64
	if m.engram != nil || cache != nil {
		t64, err := ops.Cast(tokens, dtype.I64)
		if err == nil {
			t64, err = t64.To(device.CPU())
		}
		if err == nil {
			t64, err = t64.Contiguous()
		}
		if err != nil {
			return nil, err
		}
		ids := tensor.MustSlice[int64](t64)
		T := S
		if cache != nil { // the n-grams reach back into the history
			if P == 0 {
				cache.tokens = make([][]int64, B)
			}
			for b := range B {
				cache.tokens[b] = append(cache.tokens[b], ids[b*S:(b+1)*S]...)
			}
			T = P + S
			ids = make([]int64, 0, B*T)
			for b := range B {
				ids = append(ids, cache.tokens[b]...)
			}
		}
		if m.engram != nil {
			if hashes, err = m.engram.hashIDs(ids, B, T); err != nil {
				return nil, err
			}
			if T > S { // keep the new positions' rows
				for li, h := range hashes {
					cols := len(h) / (B * T)
					nh := make([]int64, 0, B*S*cols)
					for b := range B {
						nh = append(nh, h[(b*T+P)*cols:(b+1)*T*cols]...)
					}
					hashes[li] = nh
				}
			}
		}
	}
	flatTok := o.do(func() (*tensor.Tensor, error) { return tokens.Reshape(B * S) })
	h := o.reshape(o.do(func() (*tensor.Tensor, error) { return ops.IndexSelect(m.Embed, flatTok) }), B, S, 1, D)
	copies := make([]*tensor.Tensor, N)
	for i := range copies {
		copies[i] = h
	}
	streams := o.cat(2, copies...) // (B, S, N, D)
	preMix := o.do(func() (*tensor.Tensor, error) {
		v := make([]float32, B*S*N)
		for i := 0; i < len(v); i += N {
			v[i] = 1
		}
		return tensor.New(v, []int{B, S, N, 1}, m.dev)
	})
	st := &v41Shared{}
	for li, ly := range m.Layers {
		if ly.EngramIdx >= 0 {
			streams = m.engramApply(o, ly, streams, hashes[ly.EngramIdx])
		}
		// Each sub-block's mixes feed the next sub-block's collapse: the
		// attention input uses the previous FFN's pre weights, the FFN
		// input this attention's.
		attnPre, post, combT := m.hc.hcWeights(o, &ly.AttnHC, streams)
		x := m.rms(o, m.hcPre(o, streams, preMix), ly.AttnNormW)
		var kc *v41LayerCache
		if cache != nil {
			kc = &cache.layers[li]
		}
		x = m.attention(o, ly, x, st, kc, P)
		streams = m.hc.hcCombine(o, x, streams, post, combT)
		ffnPre, post, combT := m.hc.hcWeights(o, &ly.FFNHC, streams)
		x = m.rms(o, m.hcPre(o, streams, attnPre), ly.FFNNormW)
		x = m.moe(o, ly, x)
		streams = m.hc.hcCombine(o, x, streams, post, combT)
		preMix = ffnPre
		if o.err != nil {
			return nil, fmt.Errorf("deepseek_v41 layer %d: %w", ly.ID, o.err)
		}
	}
	logits := o.lin(m.Head, m.rms(o, m.hcPre(o, streams, preMix), m.NormW))
	if o.err == nil && cache != nil {
		cache.pos += S
	}
	return logits, o.err
}
