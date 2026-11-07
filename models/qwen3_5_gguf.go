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

// LoadQwen35GGUF loads a llama.cpp GGUF of a Qwen3.5 text model
// (general.architecture "qwen35", e.g. unsloth/Qwen3.5-27B-GGUF). Linear
// weights and the token embedding stay in their block-quantized formats
// (ops.MatMulQ, quantized embedding lookup); activations run in F32. GPU
// memory ≈ file size.
//
// What llama.cpp's converter (conversion/qwen.py, Qwen3_5TextModel) changes
// relative to the HF checkpoint, and how it is handled:
//   - every norm.weight except the gated linear_attn.norm has the
//     zero-centred +1 baked in: copied as stored (the HF loader adds it);
//   - A_log is stored as ssm_a = −exp(A_log), which is NegA as is;
//   - conv1d is squeezed to (C, K), the model's layout;
//   - when there are more value than key heads, the value heads are
//     reordered from grouped to tiled order in attn_qkv's V rows, attn_gate,
//     ssm_alpha / ssm_beta, ssm_a, ssm_dt, the conv's V channels and
//     ssm_out's columns. The model runs in that order (Cfg.VHeadsTiled)
//     instead of undoing it: ssm_out's 128-column head chunks do not align
//     with its 256-wide quant blocks, so its columns cannot be permuted
//     without dequantizing.
//
// The tokenizer and chat template are in the same file
// (tokenizer.LoadGGUF; engine reads tokenizer.chat_template).
func LoadQwen35GGUF(path string, dev device.Device) (*Qwen35, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	g, err := gguf.Open(fh)
	if err != nil {
		return nil, fmt.Errorf("LoadQwen35GGUF: %w", err)
	}
	cfg, err := qwen35ConfigFromGGUF(g)
	if err != nil {
		return nil, err
	}
	m, err := NewQwen35(cfg, dev)
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
	if cfg.TieEmbed {
		headName = "token_embd.weight"
	}
	if err := setGGUFLinear(g, m.LMHead, headName, dev); err != nil {
		return nil, err
	}
	if err := copyGGUFF32(g, "output_norm.weight", m.FinalNorm.Weight); err != nil {
		return nil, err
	}

	// F32 vectors / small matrices that live on dev as plain tensors.
	f32To := func(dst **tensor.Tensor, name string) error {
		t, err := g.TensorRowMajor(name, device.CPU())
		if err != nil {
			return err
		}
		if t.DType() != dtype.F32 {
			return fmt.Errorf("LoadQwen35GGUF: %s is %v, want F32", name, t.DType())
		}
		if t.NumElements() != (*dst).NumElements() {
			return fmt.Errorf("LoadQwen35GGUF: %s shape %v, model %v", name, t.Shape(), (*dst).Shape())
		}
		if t, err = t.Reshape((*dst).Shape().Dims()...); err != nil {
			return err
		}
		*dst, err = t.To(dev)
		return err
	}
	for i, b := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", i)
		if err := copyGGUFF32(g, p+"attn_norm.weight", b.InNorm.Weight); err != nil {
			return nil, err
		}
		if err := copyGGUFF32(g, p+"post_attention_norm.weight", b.PostNorm.Weight); err != nil {
			return nil, err
		}
		lins := map[string]*nn.Linear{"ffn_gate": b.Gate, "ffn_up": b.Up, "ffn_down": b.Down}
		if b.linear {
			lins["attn_qkv"], lins["attn_gate"] = b.InQKV, b.InZ
			lins["ssm_beta"], lins["ssm_alpha"], lins["ssm_out"] = b.InB, b.InA, b.OutProj
			for _, v := range []struct {
				dst  **tensor.Tensor
				name string
			}{{&b.ConvW, "ssm_conv1d.weight"}, {&b.NegA, "ssm_a"}, {&b.DtBias, "ssm_dt.bias"}, {&b.GNorm.Weight, "ssm_norm.weight"}} {
				if err := f32To(v.dst, p+v.name); err != nil {
					return nil, err
				}
			}
		} else {
			lins["attn_q"], lins["attn_k"], lins["attn_v"], lins["attn_output"] = b.QProj, b.KProj, b.VProj, b.OProj
			if err := copyGGUFF32(g, p+"attn_q_norm.weight", b.QNorm.Weight); err != nil {
				return nil, err
			}
			if err := copyGGUFF32(g, p+"attn_k_norm.weight", b.KNorm.Weight); err != nil {
				return nil, err
			}
		}
		for name, l := range lins {
			if err := setGGUFLinear(g, l, p+name+".weight", dev); err != nil {
				return nil, err
			}
		}
	}
	return m, nil
}

// setGGUFLinear installs a GGUF weight (quantized formats kept, F16 / BF16
// cast to F32) into a placeholder Linear.
func setGGUFLinear(g *gguf.File, l *nn.Linear, name string, dev device.Device) error {
	w, err := loadGGUFLinearWeight(g, name, dev)
	if err != nil {
		return err
	}
	if err := l.SetWeight(w); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	return nil
}

// qwen35ConfigFromGGUF reads the qwen35.* metadata.
func qwen35ConfigFromGGUF(g *gguf.File) (Qwen35Config, error) {
	if arch, _ := g.String("general.architecture"); arch != "qwen35" {
		return Qwen35Config{}, fmt.Errorf("LoadQwen35GGUF: architecture %q, want qwen35", arch)
	}
	var missing []string
	geti := func(k string) int {
		v, ok := g.Int64("qwen35." + k)
		if !ok {
			missing = append(missing, k)
		}
		return int(v)
	}
	n := geti("block_count")
	cfg := Qwen35Config{
		NumLayers:        n,
		HiddenSize:       geti("embedding_length"),
		IntermediateSize: geti("feed_forward_length"),
		NumHeads:         geti("attention.head_count"),
		NumKVHeads:       geti("attention.head_count_kv"),
		HeadDim:          geti("attention.key_length"),
		ConvKernel:       geti("ssm.conv_kernel"),
		LinearKDim:       geti("ssm.state_size"),
		LinearKHeads:     geti("ssm.group_count"),
		LinearVHeads:     geti("ssm.time_step_rank"),
		WeightDType:      dtype.F32,
		VHeadsTiled:      true,
		lazyWeights:      true,
	}
	inner := geti("ssm.inner_size")
	rotDim := geti("rope.dimension_count")
	if len(missing) > 0 {
		return Qwen35Config{}, fmt.Errorf("LoadQwen35GGUF: metadata missing: qwen35.%v", missing)
	}
	if cfg.LinearVHeads == 0 || inner%cfg.LinearVHeads != 0 {
		return Qwen35Config{}, fmt.Errorf("LoadQwen35GGUF: ssm.inner_size %d not a multiple of %d value heads", inner, cfg.LinearVHeads)
	}
	cfg.LinearVDim = inner / cfg.LinearVHeads
	if cfg.HeadDim > 0 && rotDim > 0 && rotDim < cfg.HeadDim {
		cfg.PartialRotary = float64(rotDim) / float64(cfg.HeadDim)
	}
	cfg.RopeBase = 10000
	if v, ok := g.Float64("qwen35.rope.freq_base"); ok {
		cfg.RopeBase = v
	}
	cfg.RmsEps = 1e-6
	if v, ok := g.Float64("qwen35.attention.layer_norm_rms_epsilon"); ok {
		cfg.RmsEps = v
	}
	if v, ok := g.Int64("qwen35.context_length"); ok {
		cfg.MaxPosition = int(v)
	}

	// Layer types: an explicit per-layer list when the converter wrote one,
	// else every full_attention_interval-th layer is full attention.
	cfg.LinearLayers = make([]bool, n)
	switch rec := g.Meta["qwen35.recurrent_layers"].(type) {
	case []bool:
		if len(rec) < n {
			return Qwen35Config{}, fmt.Errorf("LoadQwen35GGUF: recurrent_layers has %d entries for %d layers", len(rec), n)
		}
		copy(cfg.LinearLayers, rec[:n])
	case []any:
		if len(rec) < n {
			return Qwen35Config{}, fmt.Errorf("LoadQwen35GGUF: recurrent_layers has %d entries for %d layers", len(rec), n)
		}
		for i := range cfg.LinearLayers {
			b, ok := rec[i].(bool)
			if !ok {
				return Qwen35Config{}, fmt.Errorf("LoadQwen35GGUF: recurrent_layers[%d] is %T", i, rec[i])
			}
			cfg.LinearLayers[i] = b
		}
	default:
		every, ok := g.Int64("qwen35.full_attention_interval")
		if !ok || every <= 0 {
			return Qwen35Config{}, fmt.Errorf("LoadQwen35GGUF: neither recurrent_layers nor full_attention_interval")
		}
		for i := range cfg.LinearLayers {
			cfg.LinearLayers[i] = (i+1)%int(every) != 0
		}
	}

	emb, ok := g.Tensors["token_embd.weight"]
	if !ok || len(emb.Dims) != 2 || emb.Dims[0] != cfg.HiddenSize {
		return Qwen35Config{}, fmt.Errorf("LoadQwen35GGUF: token_embd.weight missing or not (%d, vocab)", cfg.HiddenSize)
	}
	cfg.VocabSize = emb.Dims[1]
	_, hasHead := g.Tensors["output.weight"]
	cfg.TieEmbed = !hasHead

	// The output gate is packed into attn_q: twice NumHeads·HeadDim rows.
	for i, lin := range cfg.LinearLayers {
		if lin {
			continue
		}
		q, ok := g.Tensors[fmt.Sprintf("blk.%d.attn_q.weight", i)]
		if !ok || len(q.Dims) != 2 {
			return Qwen35Config{}, fmt.Errorf("LoadQwen35GGUF: blk.%d.attn_q.weight missing", i)
		}
		switch q.Dims[1] {
		case 2 * cfg.NumHeads * cfg.HeadDim:
			cfg.AttnOutputGate = true
		case cfg.NumHeads * cfg.HeadDim:
		default:
			return Qwen35Config{}, fmt.Errorf("LoadQwen35GGUF: attn_q has %d rows for %d heads of %d", q.Dims[1], cfg.NumHeads, cfg.HeadDim)
		}
		break
	}
	return cfg, nil
}
