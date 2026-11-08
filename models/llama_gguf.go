package models

import (
	"fmt"
	"os"
	"strings"

	"light/device"
	"light/dtype"
	"light/gguf"
	"light/nn"
	"light/ops"
	"light/quant"
	"light/tensor"
)

// LoadLlamaGGUF loads a llama.cpp GGUF checkpoint (architectures llama,
// qwen2, mistral and anything else that is the Llama decoder) into a *Llama
// whose Linear weights stay in their GGUF quantized format and are consumed
// by ops.MatMulQ at forward time. Activations are F32. Norm weights and
// biases are F32 as stored; the token embedding keeps its block format so
// the lookup (and a tied LM head) run on the ordinary path.
//
// For architecture "llama" the converter stores attn_q / attn_k rows in the
// interleaved order ggml's RoPE expects; they are un-permuted here back to
// HF rotate-half order. "rope_freqs.weight" (Llama-3.x frequency factors)
// is honoured as a per-pair divisor table.
//
// GPU memory ≈ file size (the embedding is looked up in its quantized form;
// a tied LM head on CPU or in a non-fused format adds an F32 copy). The tokenizer lives in the same file: `tokenizer.LoadGGUF(path)`
// builds it from the tokenizer.ggml.* metadata (llama.cpp semantics).
func LoadLlamaGGUF(path string, dev device.Device) (*Llama, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	g, err := gguf.Open(fh)
	if err != nil {
		return nil, fmt.Errorf("LoadLlamaGGUF: %w", err)
	}
	arch, ok := g.String("general.architecture")
	if !ok {
		return nil, fmt.Errorf("LoadLlamaGGUF: general.architecture missing")
	}
	cfg, err := llamaConfigFromGGUF(g, arch)
	if err != nil {
		return nil, err
	}
	m, err := NewLlama(cfg, dev)
	if err != nil {
		return nil, err
	}

	// Token embedding → F32 (dequantized on the target device).
	embRaw, err := g.TensorRowMajor("token_embd.weight", dev)
	if err != nil {
		return nil, err
	}
	// Quantized tables stay in their block format: nn.Embedding gathers the
	// rows' blocks and dequantizes only those (quantized embedding lookup).
	emb := embRaw
	if !embRaw.DType().IsQuantized() && embRaw.DType() != dtype.F32 {
		if emb, err = ops.Cast(embRaw, dtype.F32); err != nil {
			return nil, err
		}
	}
	m.TokenEmb.Weight = emb
	if cfg.TieEmbed {
		// The tied head runs MatMulQ on the same storage where a fused kernel
		// exists; otherwise (CPU, or a format that would dequantize on every
		// call) it gets a one-time F32 copy.
		headW := emb
		if emb.DType().IsQuantized() && !ops.MatMulQFusedOn(dev.Kind(), emb.DType()) {
			if headW, err = quant.Dequantize(emb); err != nil {
				return nil, fmt.Errorf("token_embd: %w", err)
			}
		}
		head, err := nn.NewLinearTied(headW, false, dev)
		if err != nil {
			return nil, err
		}
		m.LMHead = head
	} else {
		w, err := loadGGUFLinearWeight(g, "output.weight", dev)
		if err != nil {
			return nil, err
		}
		if err := m.LMHead.SetWeight(w); err != nil {
			return nil, fmt.Errorf("output.weight: %w", err)
		}
	}
	if err := copyGGUFF32(g, "output_norm.weight", m.FinalNorm.Weight); err != nil {
		return nil, err
	}

	permuteQK := arch == "llama"
	for i, blk := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", i)
		lin := map[string]*nn.Linear{
			"attn_q": blk.QProj, "attn_k": blk.KProj, "attn_v": blk.VProj, "attn_output": blk.OProj,
			"ffn_gate": blk.GateProj, "ffn_up": blk.UpProj, "ffn_down": blk.DownProj,
		}
		names := []string{"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"}
		if blk.MoE != nil {
			names = names[:4]
			if err := loadGGUFMoE(g, p, blk.MoE, dev); err != nil {
				return nil, err
			}
		}
		for _, name := range names {
			l := lin[name]
			var w *tensor.Tensor
			if permuteQK && (name == "attn_q" || name == "attn_k") {
				heads := cfg.NumHeads
				if name == "attn_k" {
					heads = cfg.NumKVHeads
				}
				w, err = loadGGUFLinearWeightUnpermuted(g, p+name+".weight", heads, dev)
			} else {
				w, err = loadGGUFLinearWeight(g, p+name+".weight", dev)
			}
			if err != nil {
				return nil, err
			}
			if err := l.SetWeight(w); err != nil {
				return nil, fmt.Errorf("%s%s: %w", p, name, err)
			}
			if l.B != nil {
				if err := copyGGUFF32(g, p+name+".bias", l.B); err != nil {
					return nil, err
				}
			}
		}
		if err := copyGGUFF32(g, p+"attn_norm.weight", blk.N1.Weight); err != nil {
			return nil, err
		}
		if err := copyGGUFF32(g, p+"ffn_norm.weight", blk.N2.Weight); err != nil {
			return nil, err
		}
		if blk.QNorm != nil {
			if err := copyGGUFF32(g, p+"attn_q_norm.weight", blk.QNorm.Weight); err != nil {
				return nil, err
			}
			if err := copyGGUFF32(g, p+"attn_k_norm.weight", blk.KNorm.Weight); err != nil {
				return nil, err
			}
		}
	}
	return m, nil
}

// loadGGUFMoE installs one layer's routed experts: `ffn_gate_inp.weight`
// (E, H) router, and the stacked `ffn_gate_exps` / `ffn_up_exps` (E, I, H)
// and `ffn_down_exps` (E, H, I) tensors. Gate and up are fused into the
// (E, 2I, H) layout nn.SparseMoE expects by concatenating each expert's
// rows at the byte level, which works unchanged for quantized blocks (a row
// is a whole number of blocks). Quantized experts stay quantized
// (ops.MoEGemmQ); F16/BF16 experts are cast to F32.
func loadGGUFMoE(g *gguf.File, p string, m *nn.SparseMoE, dev device.Device) error {
	router, err := loadGGUFLinearWeight(g, p+"ffn_gate_inp.weight", dev)
	if err != nil {
		return err
	}
	if router.DType().IsQuantized() {
		if router, err = quant.Dequantize(router); err != nil {
			return err
		}
	}
	if err := m.Gate.SetWeight(router); err != nil {
		return fmt.Errorf("%sffn_gate_inp: %w", p, err)
	}
	gi, ok := g.Tensors[p+"ffn_gate_exps.weight"]
	if !ok {
		return fmt.Errorf("%sffn_gate_exps.weight missing", p)
	}
	ui, ok := g.Tensors[p+"ffn_up_exps.weight"]
	if !ok || ui.DType != gi.DType || len(gi.Dims) != 3 {
		return fmt.Errorf("%sffn_up_exps.weight missing or mismatched", p)
	}
	H, I, E := gi.Dims[0], gi.Dims[1], gi.Dims[2] // file order innermost-first
	gb, err := g.TensorBytes(p + "ffn_gate_exps.weight")
	if err != nil {
		return err
	}
	ub, err := g.TensorBytes(p + "ffn_up_exps.weight")
	if err != nil {
		return err
	}
	slab := gi.DType.ByteSize(I * H) // one expert's (I, H) block
	fused := make([]byte, 0, 2*E*slab)
	for e := range E {
		fused = append(fused, gb[e*slab:(e+1)*slab]...)
		fused = append(fused, ub[e*slab:(e+1)*slab]...)
	}
	gu, err := tensor.FromBytes(fused, gi.DType, []int{E, 2 * I, H}, device.CPU())
	if err != nil {
		return err
	}
	down, err := g.TensorRowMajor(p+"ffn_down_exps.weight", device.CPU())
	if err != nil {
		return err
	}
	for _, t := range []**tensor.Tensor{&gu, &down} {
		if dt := (*t).DType(); !dt.IsQuantized() && dt != dtype.F32 {
			if *t, err = ops.Cast(*t, dtype.F32); err != nil {
				return err
			}
		}
		if *t, err = (*t).To(dev); err != nil {
			return err
		}
	}
	m.GateUp, m.Down = gu, down
	return nil
}

// llamaConfigFromGGUF maps the `<arch>.*` metadata keys onto LlamaConfig.
func llamaConfigFromGGUF(g *gguf.File, arch string) (LlamaConfig, error) {
	key := func(k string) string { return arch + "." + k }
	needInt := func(k string) (int, error) {
		v, ok := g.Int64(key(k))
		if !ok {
			return 0, fmt.Errorf("LoadLlamaGGUF: metadata %s missing", key(k))
		}
		return int(v), nil
	}
	optFloat := func(k string, def float64) float64 {
		if v, ok := g.Float64(key(k)); ok {
			return v
		}
		return def
	}
	var cfg LlamaConfig
	var err error
	if cfg.HiddenSize, err = needInt("embedding_length"); err != nil {
		return cfg, err
	}
	if v, ok := g.Int64(key("feed_forward_length")); ok {
		cfg.IntermediateSize = int(v)
	}
	// Mixture of experts (qwen3moe, qwen2moe, mixtral converters).
	if v, ok := g.Int64(key("expert_count")); ok && v > 0 {
		cfg.NumExperts = int(v)
		if v, ok := g.Int64(key("expert_used_count")); ok {
			cfg.NumExpertsPerTok = int(v)
		}
		if v, ok := g.Int64(key("expert_feed_forward_length")); ok {
			cfg.MoEIntermediateSize = int(v)
		} else {
			cfg.MoEIntermediateSize = cfg.IntermediateSize
		}
		if cfg.IntermediateSize == 0 {
			cfg.IntermediateSize = cfg.MoEIntermediateSize
		}
		cfg.NormTopKProb = arch == "qwen3moe" // llama.cpp renormalises top-k for qwen3moe, not qwen2moe/mixtral
	} else if cfg.IntermediateSize == 0 {
		return cfg, fmt.Errorf("LoadLlamaGGUF: metadata %s missing", key("feed_forward_length"))
	}
	if cfg.NumLayers, err = needInt("block_count"); err != nil {
		return cfg, err
	}
	if cfg.NumHeads, err = needInt("attention.head_count"); err != nil {
		return cfg, err
	}
	if v, ok := g.Int64(key("attention.head_count_kv")); ok {
		cfg.NumKVHeads = int(v)
	} else {
		cfg.NumKVHeads = cfg.NumHeads
	}
	if v, ok := g.Int64(key("rope.dimension_count")); ok {
		cfg.HeadDim = int(v)
	} else {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	if v, ok := g.Int64(key("attention.sliding_window")); ok && v > 0 {
		cfg.SlidingWindow = int(v)
	}
	if v, ok := g.Int64(key("context_length")); ok {
		cfg.MaxPosition = int(v)
	} else {
		cfg.MaxPosition = 4096
	}
	cfg.RopeBase = optFloat("rope.freq_base", 10000)
	cfg.RmsEps = optFloat("attention.layer_norm_rms_epsilon", 1e-6)
	emb, ok := g.Tensors["token_embd.weight"]
	if !ok {
		return cfg, fmt.Errorf("LoadLlamaGGUF: token_embd.weight missing")
	}
	cfg.VocabSize = emb.Dims[1] // file order [D, V]
	_, hasOut := g.Tensors["output.weight"]
	cfg.TieEmbed = !hasOut
	_, cfg.QKVBias = g.Tensors["blk.0.attn_q.bias"]
	_, cfg.QKNorm = g.Tensors["blk.0.attn_q_norm.weight"]
	// Layers without stacked expert tensors stay dense (qwen3moe mlp_only_layers).
	if cfg.NumExperts > 0 {
		for i := range cfg.NumLayers {
			if _, ok := g.Tensors[fmt.Sprintf("blk.%d.ffn_gate_exps.weight", i)]; !ok {
				cfg.MLPOnlyLayers = append(cfg.MLPOnlyLayers, i)
			}
		}
	}
	cfg.WeightDType = dtype.F32
	cfg.SkipInit = true
	cfg.LazyLinear = true
	// Llama-3.x: the converter precomputes the llama3 frequency divisors.
	if rf, ok := g.Tensors["rope_freqs.weight"]; ok {
		raw, err := g.TensorBytes("rope_freqs.weight")
		if err != nil {
			return cfg, err
		}
		t, err := tensor.FromBytes(raw, rf.DType, rf.Dims, device.CPU())
		if err != nil {
			return cfg, err
		}
		if t.DType() != dtype.F32 {
			if t, err = ops.Cast(t, dtype.F32); err != nil {
				return cfg, err
			}
		}
		fac := tensor.MustSlice[float32](t)
		f64 := make([]float64, len(fac))
		for i, v := range fac {
			f64[i] = float64(v)
		}
		cfg.RopeScaling = &RopeScaling{Type: "longrope", ShortFactor: f64, LongFactor: f64}
	}
	return cfg, nil
}

// loadGGUFLinearWeight reads a weight in (out, in) layout. Quantized
// formats stay quantized (ops.MatMulQ); F16/BF16 are cast to F32.
func loadGGUFLinearWeight(g *gguf.File, name string, dev device.Device) (*tensor.Tensor, error) {
	w, err := g.TensorRowMajor(name, dev)
	if err != nil {
		return nil, err
	}
	if w.DType().IsQuantized() || w.DType() == dtype.F32 {
		return w, nil
	}
	return ops.Cast(w, dtype.F32)
}

// loadGGUFLinearWeightUnpermuted undoes convert_hf_to_gguf's `permute` on
// llama-arch attn_q / attn_k: the converter reshapes (heads, 2, hd/2, K) →
// swap(1,2), i.e. file row h*hd + 2i + s holds HF row h*hd + s*hd/2 + i.
// Rows are whole runs of blocks, so the gather works on raw bytes.
func loadGGUFLinearWeightUnpermuted(g *gguf.File, name string, heads int, dev device.Device) (*tensor.Tensor, error) {
	info, ok := g.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	raw, err := g.TensorBytes(name)
	if err != nil {
		return nil, err
	}
	K, N := info.Dims[0], info.Dims[1]
	rowBytes := info.DType.ByteSize(K)
	if rowBytes*N != len(raw) {
		return nil, fmt.Errorf("%s: %d bytes, want %d rows × %d", name, len(raw), N, rowBytes)
	}
	hd := N / heads
	half := hd / 2
	out := make([]byte, len(raw))
	for h := 0; h < heads; h++ {
		for s := 0; s < 2; s++ {
			for i := 0; i < half; i++ {
				dst := h*hd + s*half + i
				src := h*hd + 2*i + s
				copy(out[dst*rowBytes:(dst+1)*rowBytes], raw[src*rowBytes:(src+1)*rowBytes])
			}
		}
	}
	w, err := tensor.FromBytes(out, info.DType, []int{N, K}, dev)
	if err != nil {
		return nil, err
	}
	if w.DType().IsQuantized() || w.DType() == dtype.F32 {
		return w, nil
	}
	return ops.Cast(w, dtype.F32)
}

// copyGGUFF32 copies an F32 (or F16/BF16) vector from the file into dst.
func copyGGUFF32(g *gguf.File, name string, dst *tensor.Tensor) error {
	t, err := g.TensorRowMajor(name, device.CPU())
	if err != nil {
		return err
	}
	if t.DType() != dst.DType() {
		if t, err = ops.Cast(t, dst.DType()); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	if !t.Shape().Equal(dst.Shape()) {
		return fmt.Errorf("%s: shape %v, model wants %v", name, t.Shape(), dst.Shape())
	}
	return dst.CopyFrom(t)
}

// LoadChatModelGGUF loads a chat GGUF by its general.architecture: qwen35
// (LoadQwen35GGUF), muse-glimmer (LoadMuseGlimmerGGUF), and every
// Llama-decoder architecture LoadLlamaGGUF reads.
func LoadChatModelGGUF(path string, dev device.Device) (CausalLM, error) {
	arch, err := GGUFArchitecture(path)
	if err != nil {
		return nil, err
	}
	switch arch {
	case "qwen35":
		return LoadQwen35GGUF(path, dev)
	case "muse-glimmer":
		return LoadMuseGlimmerGGUF(path, dev)
	}
	return LoadLlamaGGUF(path, dev)
}

// GGUFArchitecture returns general.architecture of a GGUF file, for
// callers that want to pick a loader or tokenizer by it.
func GGUFArchitecture(path string) (string, error) {
	fh, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer fh.Close()
	g, err := gguf.Open(fh)
	if err != nil {
		return "", err
	}
	a, _ := g.String("general.architecture")
	return strings.TrimSpace(a), nil
}
