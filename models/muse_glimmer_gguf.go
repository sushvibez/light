package models

import (
	"fmt"
	"os"

	"light/device"
	"light/dtype"
	"light/gguf"
	"light/nn"
	"light/tensor"
)

// LoadMuseGlimmerGGUF loads a llama.cpp GGUF of Muse Glimmer's text model
// (general.architecture "muse-glimmer", e.g. unsloth/Muse-Glimmer-30B-GGUF)
// onto the Gemma 3 skeleton. Linear weights and the token embedding stay in
// their block formats (ops.MatMulQ, quantized lookup); activations are F32;
// GPU memory ≈ file size.
//
// What llama.cpp's converter (conversion/muse_glimmer.py) and graph
// (src/models/muse-glimmer.cpp) do differently from the HF checkpoint:
//   - the four layer norms have the +1 baked in; the final norm does not
//     (copied as stored, like the HF loader's (1 + w) / w);
//   - attn_q / attn_k rows are permuted to ggml's interleaved RoPE layout:
//     un-permuted here back to rotate-half, as for llama-arch GGUFs;
//   - qk_scale_factor is absorbed into a synthesized attn_q_norm weight
//     (3.87 everywhere; attn_k_norm is ones): used as the q / k norm weights,
//     so Cfg.QScale stays 0;
//   - layer types come from attention.sliding_window_pattern (a period n:
//     every n-th layer, counting from 1, is full attention, the rest
//     sliding); RoPE runs on the sliding layers only (NoPE on the full ones),
//     at rope.freq_base_swa when present, else rope.freq_base;
//   - the post-attention / post-FFN norm eps is fixed at 1e-8 in the graph;
//   - logit_scale is the output multiplier, applied before the softcap.
//
// The tokenizer and chat template are in the same file.
func LoadMuseGlimmerGGUF(path string, dev device.Device) (*Gemma3, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	g, err := gguf.Open(fh)
	if err != nil {
		return nil, fmt.Errorf("LoadMuseGlimmerGGUF: %w", err)
	}
	cfg, err := museGlimmerConfigFromGGUF(g)
	if err != nil {
		return nil, err
	}
	m, err := NewGemma3(cfg, dev)
	if err != nil {
		return nil, err
	}
	emb, err := g.TensorRowMajor("token_embd.weight", dev)
	if err != nil {
		return nil, err
	}
	if !emb.DType().IsQuantized() && emb.DType() != dtype.F32 {
		if emb, err = loadGGUFLinearWeight(g, "token_embd.weight", dev); err != nil {
			return nil, err
		}
	}
	if m.TokenEmb, err = nn.NewEmbeddingFromWeight(emb); err != nil {
		return nil, err
	}
	headName := "output.weight"
	if !cfg.UntiedHead {
		headName = "token_embd.weight"
	}
	if err := setGGUFLinear(g, m.LMHead, headName, dev); err != nil {
		return nil, err
	}
	if err := copyGGUFF32(g, "output_norm.weight", m.FinalNorm.Weight); err != nil {
		return nil, err
	}
	// The weightless embedding norm: ones (NewGemma3 allocates it as ones
	// already, F32 since the activations are F32).
	for i, b := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", i)
		for _, nr := range []struct {
			dst  *tensor.Tensor
			name string
		}{
			{b.InNorm.Weight, "attn_norm.weight"}, {b.PostAttnNorm.Weight, "post_attention_norm.weight"},
			{b.PreFFNorm.Weight, "ffn_norm.weight"}, {b.PostFFNorm.Weight, "post_ffw_norm.weight"},
			{b.QNorm.Weight, "attn_q_norm.weight"}, {b.KNorm.Weight, "attn_k_norm.weight"},
		} {
			if err := copyGGUFF32(g, p+nr.name, nr.dst); err != nil {
				return nil, err
			}
		}
		for _, qk := range []struct {
			l     *nn.Linear
			name  string
			heads int
		}{{b.QProj, "attn_q.weight", cfg.NumHeads}, {b.KProj, "attn_k.weight", cfg.NumKVHeads}} {
			w, err := loadGGUFLinearWeightUnpermuted(g, p+qk.name, qk.heads, dev)
			if err != nil {
				return nil, err
			}
			if err := qk.l.SetWeight(w); err != nil {
				return nil, fmt.Errorf("%s: %w", p+qk.name, err)
			}
		}
		for name, l := range map[string]*nn.Linear{
			"attn_v": b.VProj, "attn_output": b.OProj, "attn_gate": b.AttnGate,
			"ffn_gate": b.GateProj, "ffn_up": b.UpProj, "ffn_down": b.DownProj,
		} {
			if err := setGGUFLinear(g, l, p+name+".weight", dev); err != nil {
				return nil, err
			}
		}
	}
	return m, nil
}

// museGlimmerConfigFromGGUF reads the muse-glimmer.* metadata.
func museGlimmerConfigFromGGUF(g *gguf.File) (Gemma3Config, error) {
	const arch = "muse-glimmer"
	if a, _ := g.String("general.architecture"); a != arch {
		return Gemma3Config{}, fmt.Errorf("LoadMuseGlimmerGGUF: architecture %q, want %s", a, arch)
	}
	var missing []string
	geti := func(k string) int {
		v, ok := g.Int64(arch + "." + k)
		if !ok {
			missing = append(missing, k)
		}
		return int(v)
	}
	n := geti("block_count")
	cfg := Gemma3Config{
		NumLayers: n, HiddenSize: geti("embedding_length"), IntermediateSize: geti("feed_forward_length"),
		NumHeads: geti("attention.head_count"), NumKVHeads: geti("attention.head_count_kv"),
		HeadDim: geti("attention.key_length"), SlidingWindow: geti("attention.sliding_window"),
		WeightDType: dtype.F32, EmbedNorm: true, AttnOutputGate: true, SiLUMLP: true,
		PostNormEps: 1e-8, lazyWeights: true,
	}
	if len(missing) > 0 {
		return Gemma3Config{}, fmt.Errorf("LoadMuseGlimmerGGUF: metadata missing: %s.%v", arch, missing)
	}
	cfg.MaxPosition = 131072
	if v, ok := g.Int64(arch + ".context_length"); ok {
		cfg.MaxPosition = int(v)
	}
	cfg.RmsEps = 1e-5
	if v, ok := g.Float64(arch + ".attention.layer_norm_rms_epsilon"); ok {
		cfg.RmsEps = v
	}
	cfg.FinalLogitSoftcap, _ = g.Float64(arch + ".final_logit_softcapping")
	cfg.OutputMultiplier, _ = g.Float64(arch + ".logit_scale")
	base, ok := g.Float64(arch + ".rope.freq_base")
	if !ok {
		return Gemma3Config{}, fmt.Errorf("LoadMuseGlimmerGGUF: %s.rope.freq_base missing", arch)
	}
	if v, ok := g.Float64(arch + ".rope.freq_base_swa"); ok {
		base = v
	}

	// Sliding layers: an explicit per-layer bool array, or a period.
	cfg.SlidingLayers = make([]bool, n)
	switch pat := g.Meta[arch+".attention.sliding_window_pattern"].(type) {
	case []bool:
		if len(pat) < n {
			return Gemma3Config{}, fmt.Errorf("LoadMuseGlimmerGGUF: sliding_window_pattern has %d entries for %d layers", len(pat), n)
		}
		copy(cfg.SlidingLayers, pat[:n])
	case []any:
		if len(pat) < n {
			return Gemma3Config{}, fmt.Errorf("LoadMuseGlimmerGGUF: sliding_window_pattern has %d entries for %d layers", len(pat), n)
		}
		for i := range cfg.SlidingLayers {
			b, ok := pat[i].(bool)
			if !ok {
				return Gemma3Config{}, fmt.Errorf("LoadMuseGlimmerGGUF: sliding_window_pattern[%d] is %T", i, pat[i])
			}
			cfg.SlidingLayers[i] = b
		}
	default:
		period := 4 // llama.cpp's default for this arch
		if v, ok := g.Int64(arch + ".attention.sliding_window_pattern"); ok && v > 0 {
			period = int(v)
		}
		for i := range cfg.SlidingLayers {
			cfg.SlidingLayers[i] = i%period < period-1
		}
	}
	cfg.LayerRopeBase = make([]float64, n)
	for i, sl := range cfg.SlidingLayers {
		if sl {
			cfg.LayerRopeBase[i] = base // RoPE on sliding layers only; full layers are NoPE
		}
	}

	emb, ok := g.Tensors["token_embd.weight"]
	if !ok || len(emb.Dims) != 2 || emb.Dims[0] != cfg.HiddenSize {
		return Gemma3Config{}, fmt.Errorf("LoadMuseGlimmerGGUF: token_embd.weight missing or not (%d, vocab)", cfg.HiddenSize)
	}
	cfg.VocabSize = emb.Dims[1]
	_, hasHead := g.Tensors["output.weight"]
	cfg.UntiedHead = hasHead
	return cfg, nil
}
