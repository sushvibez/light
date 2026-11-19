package models

import (
	"encoding/json"
	"fmt"
	"light/nn"
	"os"
	"path/filepath"
	"slices"
	"strconv"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/safetensors"
	"light/tensor"
)

// LoadDeepseekV41 loads DeepSeek-V4.1's language model from a folder with
// the released config.json and safetensors (DeepSeek's own tensor names:
// embed, layers.N.attn.wq_a, …, head), onto dev at F32. FP8 projections
// (E4M3 with 32 × 32 UE8M0 block scales) stay block-scaled FP8 on CPU and
// CUDA (widened at load elsewhere); the FP4 routed experts (e2m1 packed in
// I8, 1 × 32 UE8M0 scales) stay packed in the checkpoint and are widened to
// F32 the first time the router picks them (LIGHT_EXPERT_SLOTS bounds how
// many stay resident per layer; unset, every expert stays once loaded). The engram tables
// stay in the (memory-mapped) checkpoint on the host; the engram hashing
// also needs engram_token_map.json next to config.json. vision.*, aligner.*,
// image_* and the DSpark heads (mtp.*) are not loaded.
func LoadDeepseekV41(dir string, dev device.Device) (*DeepseekV41, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var top struct {
		TextConfig json.RawMessage `json:"text_config"`
		Quant      *struct {
			Method    string `json:"quant_method"`
			BlockSize []int  `json:"weight_block_size"`
		} `json:"quantization_config"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, err
	}
	cfg := DeepseekV41Config{RMSNormEps: 1e-6, HCEps: 1e-6, HCSinkhornIters: 20, HCMult: 4, ScoringFunc: "sqrtsoftplus",
		RoutedScaling: 1, NormTopkProb: true}
	src := raw
	if len(top.TextConfig) > 0 {
		src = top.TextConfig
	}
	if err := json.Unmarshal(src, &cfg); err != nil {
		return nil, fmt.Errorf("deepseek_v41 config: %w", err)
	}
	L := cfg.NumLayers
	if len(cfg.CompressRatios) < L {
		return nil, fmt.Errorf("deepseek_v41: %d compress ratios for %d layers", len(cfg.CompressRatios), L)
	}
	if cfg.NSharedExperts != 1 {
		return nil, fmt.Errorf("deepseek_v41: n_shared_experts %d (want 1)", cfg.NSharedExperts)
	}
	if cfg.RopeHeadDim%2 != 0 || cfg.RopeHeadDim > cfg.HeadDim || cfg.RopeHeadDim > cfg.IndexHeadDim && len(cfg.IndexSourceLayers) > 0 {
		return nil, fmt.Errorf("deepseek_v41: qk_rope_head_dim %d with head_dim %d / index_head_dim %d", cfg.RopeHeadDim, cfg.HeadDim, cfg.IndexHeadDim)
	}
	if cfg.OGroups < 1 || cfg.NumHeads*cfg.HeadDim%cfg.OGroups != 0 {
		return nil, fmt.Errorf("deepseek_v41: o_groups %d", cfg.OGroups)
	}
	if err := v41CheckSharing(cfg); err != nil {
		return nil, err
	}
	paths, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	hf, _, err := safetensors.LoadFiles(paths, device.CPU())
	if err != nil {
		return nil, err
	}
	l := &idolLoader{hf: hf, dev: dev, fp8Block: [2]int{32, 32}}
	if q := top.Quant; q != nil && len(q.BlockSize) == 2 {
		l.fp8Block = [2]int{q.BlockSize[0], q.BlockSize[1]}
	}
	m := &DeepseekV41{Cfg: cfg, dev: dev}
	m.hc = &GLM5{Cfg: GLM5Config{RMSNormEps: cfg.RMSNormEps, HCEps: cfg.HCEps, HCSinkhornIters: cfg.HCSinkhornIters},
		ones: map[int]*tensor.Tensor{}, dev: dev}
	if len(cfg.EngramLayerIDs) > 0 {
		if m.engram, err = newV41Engram(cfg, dir); err != nil {
			return nil, err
		}
	}
	lin := func(name string) *nn.Linear {
		v41Prep(l, name)
		return l.linear(name)
	}
	m.Embed = l.param("embed.weight")
	m.NormW = l.param("norm.weight")
	if cfg.TieEmbeddings {
		hf["head.weight"] = hf["embed.weight"]
	}
	m.Head = lin("head")
	hc := func(p string) glm5HC {
		h := glm5HC{Fn: l.param(p + "_fn"), Base: l.host(p + "_base")}
		if s := l.host(p + "_scale"); len(s) == 3 {
			copy(h.Scale[:], s)
		}
		return h
	}
	expert := func(p string) *v41Expert {
		return &v41Expert{W1: lin(p + ".w1"), W2: lin(p + ".w2"), W3: lin(p + ".w3")}
	}
	for i := range L {
		p := fmt.Sprintf("layers.%d.", i)
		ly := &v41Layer{ID: i, Ratio: cfg.CompressRatios[i], EngramIdx: slices.Index(cfg.EngramLayerIDs, i),
			KVSource: slices.Contains(cfg.KVSourceLayers, i), IndexSource: slices.Contains(cfg.IndexSourceLayers, i),
			AttnHC: hc(p + "hc_attn"), FFNHC: hc(p + "hc_ffn"),
			AttnNormW: l.param(p + "attn_norm.weight"), FFNNormW: l.param(p + "ffn_norm.weight")}
		a := p + "attn."
		ly.WqA, ly.WqB, ly.Wkv, ly.WoB = lin(a+"wq_a"), lin(a+"wq_b"), lin(a+"wkv"), lin(a+"wo_b")
		ly.QNormW, ly.KVNormW = l.param(a+"q_norm.weight"), l.param(a+"kv_norm.weight")
		ly.Sink = l.host(a + "attn_sink")
		v41Prep(l, a+"wo_a")
		ly.WoA = v41SplitRows(l, a+"wo_a", cfg.OGroups)
		if ly.Ratio > 0 && ly.KVSource {
			ly.CWkv, ly.CNormW = lin(a+"compressor.wkv"), l.param(a+"compressor.norm.weight")
			if ly.Ratio > 1 {
				ly.CWgate = lin(a + "compressor.wgate")
			}
		}
		if ly.Ratio > 0 && ly.IndexSource {
			ly.IWqB, ly.IWeights = lin(a+"indexer.wq_b"), lin(a+"indexer.weights_proj")
			if ly.KVSource {
				ly.IWk, ly.IKNormW = lin(a+"indexer.wk"), l.param(a+"indexer.k_norm.weight")
			}
		}
		f := p + "ffn."
		ly.GateW, ly.GateBias = l.param(f+"gate.weight"), l.host(f+"gate.bias")
		ly.Shared = expert(f + "shared_experts")
		for e := range cfg.NRoutedExperts { // loaded on first use
			name := fmt.Sprintf("%sexperts.%d", f, e)
			for _, w := range []string{".w1.weight", ".w2.weight", ".w3.weight"} {
				if _, ok := hf[name+w]; !ok && l.err == nil {
					l.err = fmt.Errorf("missing %s%s", name, w)
				}
			}
		}
		slots := cfg.NRoutedExperts
		if n := envExpertSlots(); n > 0 {
			slots = min(n, slots)
		}
		prefix := f + "experts."
		if ly.Experts, err = nn.NewExpertCache(slots, func(e int) (*v41Expert, error) { return v41LoadExpert(l, prefix+strconv.Itoa(e)) }); err != nil {
			return nil, err
		}
		if ly.EngramIdx >= 0 {
			en := p + "engram."
			ly.EWkv = lin(en + "wkv")
			ly.ETable, ly.EScale = hf[en+"embed.weight"], hf[en+"embed.scale"]
			if ly.ETable == nil || ly.EScale == nil {
				return nil, fmt.Errorf("deepseek_v41: missing %sembed.weight / .scale", en)
			}
			if n := cfg.EngramNumEmbeddings; ly.EngramIdx < len(n) && int64(ly.ETable.Dim(0)) < n[ly.EngramIdx] {
				return nil, fmt.Errorf("deepseek_v41: %sembed has %d rows, config says %d", en, ly.ETable.Dim(0), n[ly.EngramIdx])
			}
			qw, kw := l.host(en+"q_weight"), l.host(en+"k_weight")
			if l.err == nil {
				w := make([]float32, len(qw))
				for j := range w {
					w[j] = qw[j] * kw[j]
				}
				ly.EWeight, l.err = tensor.New(w, []int{1, 1, cfg.HCMult, cfg.HiddenSize}, dev)
			}
		}
		if l.err != nil {
			return nil, fmt.Errorf("deepseek_v41 layer %d: %w", i, l.err)
		}
		m.Layers = append(m.Layers, ly)
	}
	if l.err != nil {
		return nil, fmt.Errorf("deepseek_v41: %w", l.err)
	}
	if m.engram != nil {
		for li := range m.engram.layerIDs {
			last := m.engram.offsets[li][len(m.engram.offsets[li])-1] + m.engram.primes[li][len(m.engram.primes[li])-1]
			if t := m.Layers[m.engram.layerIDs[li]].ETable; int64(t.Dim(0)) < last {
				return nil, fmt.Errorf("deepseek_v41: engram layer %d table has %d rows, the hash needs %d", m.engram.layerIDs[li], t.Dim(0), last)
			}
		}
	}
	for i := 1; i < len(m.Layers); i++ {
		m.Layers[i-1].next = m.Layers[i]
	}
	return m, nil
}

// v41LoadExpert widens routed expert name's weights from the checkpoint
// (through a loader of its own, so the shared map keeps only the packed
// tensors and loads may run concurrently).
func v41LoadExpert(src *idolLoader, name string) (*v41Expert, error) {
	sub := src.sub(name+".w1", name+".w2", name+".w3")
	lin := func(n string) *nn.Linear {
		v41Prep(sub, n)
		return sub.linear(n)
	}
	e := &v41Expert{W1: lin(name + ".w1"), W2: lin(name + ".w2"), W3: lin(name + ".w3")}
	if sub.err != nil {
		return nil, fmt.Errorf("deepseek_v41 %s: %w", name, sub.err)
	}
	return e, nil
}

// v41CheckSharing checks that every compressing layer reads a compressed KV
// and a top-k produced at its own ratio by an earlier (or its own) source.
func v41CheckSharing(cfg DeepseekV41Config) error {
	kvR, idxR := 0, 0
	for i := range cfg.NumLayers {
		r := cfg.CompressRatios[i]
		if r == 0 {
			continue
		}
		if slices.Contains(cfg.KVSourceLayers, i) {
			kvR = r
		}
		if slices.Contains(cfg.IndexSourceLayers, i) {
			idxR = r
		}
		if kvR != r || idxR != r {
			return fmt.Errorf("deepseek_v41: layer %d (compress ratio %d) has no kv / index source at its ratio", i, r)
		}
	}
	if cs := cfg.CandidateSourceLayer; cs != nil && *cs >= 0 {
		if !slices.Contains(cfg.IndexSourceLayers, *cs) || cfg.CandidateBlockSize < 1 {
			return fmt.Errorf("deepseek_v41: candidate source %d is not an index source (block size %d)", *cs, cfg.CandidateBlockSize)
		}
	}
	return nil
}

// v41Prep turns the released <name>.scale into what idolLoader.linear
// reads: an FP8 weight gets its UE8M0 block scales as F32
// <name>.weight_scale_inv; an FP4 weight (e2m1 pairs in I8, 1 × blk
// scales) is widened to F32 on the host.
func v41Prep(l *idolLoader, name string) {
	if l.err != nil {
		return
	}
	w, ok := l.hf[name+".weight"]
	sc, hasS := l.hf[name+".scale"]
	if !ok || !hasS {
		return
	}
	scF, err := v41ScaleF32(sc)
	if err != nil {
		l.err = fmt.Errorf("%s.scale: %w", name, err)
		return
	}
	switch w.DType() {
	case dtype.F8E4M3:
		l.hf[name+".weight_scale_inv"] = scF
	case dtype.I8, dtype.U8:
		d, err := dequantFP4Host(w, scF)
		if err != nil {
			l.err = fmt.Errorf("%s: %w", name, err)
			return
		}
		l.hf[name+".weight"] = d
	default:
		l.err = fmt.Errorf("%s: %v weight with a block scale", name, w.DType())
		return
	}
	delete(l.hf, name+".scale")
}

// v41ScaleF32 decodes UE8M0 scale bytes (U8) to F32, or casts float scales.
func v41ScaleF32(sc *tensor.Tensor) (*tensor.Tensor, error) {
	if sc.DType() != dtype.U8 {
		return ops.Cast(sc, dtype.F32)
	}
	c, err := sc.Contiguous()
	if err != nil {
		return nil, err
	}
	b := tensor.MustSlice[uint8](c)
	out := make([]float32, len(b))
	for i, v := range b {
		out[i] = e8m0ToF32(v)
	}
	return tensor.New(out, sc.Shape().Dims(), device.CPU())
}

// fp4E2M1 is the e2m1 code table (sign in bit 3).
var fp4E2M1 = [16]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}

// dequantFP4Host widens a packed FP4 weight (N, K/2) — two e2m1 values per
// byte along K, the low nibble first — with its (N, K/blk) scales to F32.
func dequantFP4Host(w, scale *tensor.Tensor) (*tensor.Tensor, error) {
	if w.Rank() != 2 || scale.Rank() != 2 || scale.Dim(0) != w.Dim(0) {
		return nil, fmt.Errorf("fp4 weight %v with scale %v", w.Shape(), scale.Shape())
	}
	N, K := w.Dim(0), 2*w.Dim(1)
	nb := scale.Dim(1)
	if nb == 0 || K%nb != 0 {
		return nil, fmt.Errorf("fp4 weight %v with scale %v", w.Shape(), scale.Shape())
	}
	blk := K / nb
	wc, err := w.Contiguous()
	if err != nil {
		return nil, err
	}
	sc, err := scale.Contiguous()
	if err != nil {
		return nil, err
	}
	pb := wc.Storage().CPUBytes()[wc.Offset() : wc.Offset()+N*K/2]
	sv := tensor.MustSlice[float32](sc)
	out := make([]float32, N*K)
	for i := range N {
		for j := 0; j < K; j += 2 {
			b := pb[(i*K+j)/2]
			out[i*K+j] = fp4E2M1[b&0xF] * sv[i*nb+j/blk]
			out[i*K+j+1] = fp4E2M1[b>>4] * sv[i*nb+(j+1)/blk]
		}
	}
	return tensor.New(out, []int{N, K}, device.CPU())
}

// v41SplitRows splits the grouped output projection wo_a (G·r, d) into G
// linears (r, d), keeping FP8 block scales when r is a whole number of
// blocks (widening on the host otherwise).
func v41SplitRows(l *idolLoader, name string, G int) []*nn.Linear {
	if l.err != nil {
		return nil
	}
	w, ok := l.hf[name+".weight"]
	if !ok {
		l.err = fmt.Errorf("deepseek_v41: missing %s.weight", name)
		return nil
	}
	if w.Dim(0)%G != 0 {
		l.err = fmt.Errorf("deepseek_v41: %s %v not divisible into %d groups", name, w.Shape(), G)
		return nil
	}
	r := w.Dim(0) / G
	sc, scaled := l.hf[name+".weight_scale_inv"]
	if scaled && (w.DType() != dtype.F8E4M3 || r%l.fp8Block[0] != 0) {
		d, err := dequantFP8BlockHost(w, sc, l.fp8Block)
		if err != nil {
			l.err = err
			return nil
		}
		w, scaled = d, false
		delete(l.hf, name+".weight_scale_inv")
	}
	out := make([]*nn.Linear, G)
	for g := range G {
		part := func(t *tensor.Tensor, start, n int) *tensor.Tensor {
			v, err := t.Narrow(0, start, n)
			if err == nil {
				v, err = v.Contiguous()
			}
			if err != nil && l.err == nil {
				l.err = err
			}
			return v
		}
		gn := fmt.Sprintf("%s.g%d", name, g)
		l.hf[gn+".weight"] = part(w, g*r, r)
		if scaled {
			rb := r / l.fp8Block[0]
			l.hf[gn+".weight_scale_inv"] = part(sc, g*rb, rb)
		}
		if l.err != nil {
			return nil
		}
		out[g] = l.linear(gn)
	}
	delete(l.hf, name+".weight")
	delete(l.hf, name+".weight_scale_inv")
	return out
}
