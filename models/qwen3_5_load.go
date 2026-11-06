package models

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/ops"
	"light/tensor"
)

// qwen35HFConfig is the text part of a qwen3_5 / qwen3_5_text config.json.
// Multimodal checkpoints (Qwen3_5ForConditionalGeneration, the published
// Qwen3.5 / 3.8 repos) nest it under text_config.
type qwen35HFConfig struct {
	ModelType            string   `json:"model_type"`
	VocabSize            int      `json:"vocab_size"`
	HiddenSize           int      `json:"hidden_size"`
	IntermediateSize     int      `json:"intermediate_size"`
	NumHiddenLayers      int      `json:"num_hidden_layers"`
	LayerTypes           []string `json:"layer_types"`
	FullAttnInterval     int      `json:"full_attention_interval"`
	NumAttentionHeads    int      `json:"num_attention_heads"`
	NumKeyValueHeads     int      `json:"num_key_value_heads"`
	HeadDim              int      `json:"head_dim"`
	AttnOutputGate       *bool    `json:"attn_output_gate"`
	LinearNumKeyHeads    int      `json:"linear_num_key_heads"`
	LinearNumValueHeads  int      `json:"linear_num_value_heads"`
	LinearKeyHeadDim     int      `json:"linear_key_head_dim"`
	LinearValueHeadDim   int      `json:"linear_value_head_dim"`
	LinearConvKernelDim  int      `json:"linear_conv_kernel_dim"`
	RmsNormEps           float64  `json:"rms_norm_eps"`
	MaxPositionEmbedding int      `json:"max_position_embeddings"`
	TieWordEmbeddings    *bool    `json:"tie_word_embeddings"`
	PartialRotaryFactor  float64  `json:"partial_rotary_factor"`
	RopeTheta            float64  `json:"rope_theta"`
	RopeParameters       *struct {
		RopeTheta           float64 `json:"rope_theta"`
		PartialRotaryFactor float64 `json:"partial_rotary_factor"`
		RopeType            string  `json:"rope_type"`
	} `json:"rope_parameters"`
	TextConfig json.RawMessage `json:"text_config"`
}

// fetchQwen35Config reads config.json (descending into text_config) into a
// Qwen35Config.
func fetchQwen35Config(repo string) (Qwen35Config, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return Qwen35Config{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return Qwen35Config{}, err
	}
	var top, hf qwen35HFConfig
	if err := json.Unmarshal(raw, &top); err != nil {
		return Qwen35Config{}, fmt.Errorf("qwen3.5 config: %w", err)
	}
	hf = top
	if len(top.TextConfig) > 0 {
		hf = qwen35HFConfig{}
		if err := json.Unmarshal(top.TextConfig, &hf); err != nil {
			return Qwen35Config{}, fmt.Errorf("qwen3.5 text_config: %w", err)
		}
	}
	tie := false
	switch {
	case hf.TieWordEmbeddings != nil:
		tie = *hf.TieWordEmbeddings
	case top.TieWordEmbeddings != nil:
		tie = *top.TieWordEmbeddings
	}
	theta, partial := hf.RopeTheta, hf.PartialRotaryFactor
	if rp := hf.RopeParameters; rp != nil {
		if rp.RopeType != "" && rp.RopeType != "default" {
			return Qwen35Config{}, fmt.Errorf("qwen3.5: rope_type %q not supported", rp.RopeType)
		}
		if rp.RopeTheta > 0 {
			theta = rp.RopeTheta
		}
		if rp.PartialRotaryFactor > 0 {
			partial = rp.PartialRotaryFactor
		}
	}
	if theta == 0 {
		theta = 10000
	}
	n := hf.NumHiddenLayers
	linear := make([]bool, n)
	switch {
	case len(hf.LayerTypes) == n:
		for i, t := range hf.LayerTypes {
			switch t {
			case "linear_attention":
				linear[i] = true
			case "full_attention":
			default:
				return Qwen35Config{}, fmt.Errorf("qwen3.5: layer %d type %q not supported", i, t)
			}
		}
	case hf.FullAttnInterval > 0:
		for i := range linear {
			linear[i] = (i+1)%hf.FullAttnInterval != 0
		}
	default:
		return Qwen35Config{}, fmt.Errorf("qwen3.5: config has neither layer_types nor full_attention_interval")
	}
	gate := hf.AttnOutputGate == nil || *hf.AttnOutputGate
	eps := hf.RmsNormEps
	if eps == 0 {
		eps = 1e-6
	}
	conv := hf.LinearConvKernelDim
	if conv == 0 {
		conv = 4
	}
	return Qwen35Config{
		VocabSize: hf.VocabSize, HiddenSize: hf.HiddenSize, IntermediateSize: hf.IntermediateSize,
		NumLayers: n, LinearLayers: linear,
		NumHeads: hf.NumAttentionHeads, NumKVHeads: hf.NumKeyValueHeads, HeadDim: hf.HeadDim,
		PartialRotary: partial, RopeBase: theta, AttnOutputGate: gate,
		LinearKHeads: hf.LinearNumKeyHeads, LinearVHeads: hf.LinearNumValueHeads,
		LinearKDim: hf.LinearKeyHeadDim, LinearVDim: hf.LinearValueHeadDim, ConvKernel: conv,
		RmsEps: eps, MaxPosition: hf.MaxPositionEmbedding, TieEmbed: tie,
	}, nil
}

// LoadQwen35FromHF loads the text model of a qwen3_5 checkpoint (Hub repo
// or local directory), e.g. `Qwen/Qwen3.5-0.8B`. The vision tower
// (`model.visual.*`) and the multi-token-prediction head (`mtp.*`) are not
// loaded. weightDT is dtype.F32 or dtype.BF16.
func LoadQwen35FromHF(repo string, dev device.Device, weightDT dtype.DType) (*Qwen35, error) {
	cfg, err := fetchQwen35Config(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadQwen35FromHF: %w", err)
	}
	m, err := NewQwen35(cfg, dev)
	if err != nil {
		return nil, err
	}
	prefix := ""
	for _, p := range []string{"model.language_model.", "model."} {
		if _, ok := hf[p+"embed_tokens.weight"]; ok {
			prefix = p
			break
		}
	}
	if prefix == "" {
		return nil, fmt.Errorf("LoadQwen35FromHF: no embed_tokens.weight under model. or model.language_model.")
	}
	act := cfg.activationDType()
	if err := copyExactCast(m.TokenEmb.Weight, hf, prefix+"embed_tokens.weight"); err != nil {
		return nil, err
	}
	// Zero-centred RMSNorm: store 1 + w, at the activation dtype.
	normPlusOne := func(dst **tensor.Tensor, name string) error {
		src, ok := hf[name]
		if !ok {
			return fmt.Errorf("LoadQwen35FromHF: missing tensor %q", name)
		}
		w, err := ops.Cast(src, dtype.F32)
		if err != nil {
			return err
		}
		one, err := tensor.New([]float32{1}, []int{1}, w.Device())
		if err != nil {
			return err
		}
		if w, err = ops.Add(w, one); err != nil {
			return err
		}
		if act != dtype.F32 {
			if w, err = ops.Cast(w, act); err != nil {
				return err
			}
		}
		if !w.Shape().Equal((*dst).Shape()) {
			return fmt.Errorf("%s: shape %v, model %v", name, w.Shape(), (*dst).Shape())
		}
		if *dst, err = w.To(dev); err != nil {
			return err
		}
		return nil
	}
	// F32 parameters of the linear layers: conv weight (C, 1, K) → (C, K),
	// −exp(A_log), dt_bias, gated-norm weight.
	f32 := func(dst **tensor.Tensor, name string, fn func(*tensor.Tensor) (*tensor.Tensor, error)) error {
		src, ok := hf[name]
		if !ok {
			return fmt.Errorf("LoadQwen35FromHF: missing tensor %q", name)
		}
		w, err := ops.Cast(src, dtype.F32)
		if err != nil {
			return err
		}
		if fn != nil {
			if w, err = fn(w); err != nil {
				return err
			}
		}
		if w.Shape().NumElements() != (*dst).Shape().NumElements() {
			return fmt.Errorf("%s: shape %v, model %v", name, w.Shape(), (*dst).Shape())
		}
		if w, err = w.Reshape((*dst).Shape().Dims()...); err != nil {
			return err
		}
		*dst, err = w.To(dev)
		return err
	}
	negExp := func(x *tensor.Tensor) (*tensor.Tensor, error) {
		e, err := ops.Exp(x)
		if err != nil {
			return nil, err
		}
		return ops.Neg(e)
	}
	for i, b := range m.Blocks {
		p := prefix + "layers." + strconv.Itoa(i) + "."
		if err := normPlusOne(&b.InNorm.Weight, p+"input_layernorm.weight"); err != nil {
			return nil, err
		}
		if err := normPlusOne(&b.PostNorm.Weight, p+"post_attention_layernorm.weight"); err != nil {
			return nil, err
		}
		type pair struct {
			dst  *tensor.Tensor
			name string
		}
		pairs := []pair{
			{b.Gate.W, p + "mlp.gate_proj.weight"},
			{b.Up.W, p + "mlp.up_proj.weight"},
			{b.Down.W, p + "mlp.down_proj.weight"},
		}
		if b.linear {
			la := p + "linear_attn."
			pairs = append(pairs,
				pair{b.InQKV.W, la + "in_proj_qkv.weight"}, pair{b.InZ.W, la + "in_proj_z.weight"},
				pair{b.InB.W, la + "in_proj_b.weight"}, pair{b.InA.W, la + "in_proj_a.weight"},
				pair{b.OutProj.W, la + "out_proj.weight"})
			if err := f32(&b.ConvW, la+"conv1d.weight", nil); err != nil {
				return nil, err
			}
			if err := f32(&b.NegA, la+"A_log", negExp); err != nil {
				return nil, err
			}
			if err := f32(&b.DtBias, la+"dt_bias", nil); err != nil {
				return nil, err
			}
			if err := f32(&b.GNorm.Weight, la+"norm.weight", nil); err != nil {
				return nil, err
			}
		} else {
			sa := p + "self_attn."
			pairs = append(pairs,
				pair{b.QProj.W, sa + "q_proj.weight"}, pair{b.KProj.W, sa + "k_proj.weight"},
				pair{b.VProj.W, sa + "v_proj.weight"}, pair{b.OProj.W, sa + "o_proj.weight"})
			if err := normPlusOne(&b.QNorm.Weight, sa+"q_norm.weight"); err != nil {
				return nil, err
			}
			if err := normPlusOne(&b.KNorm.Weight, sa+"k_norm.weight"); err != nil {
				return nil, err
			}
		}
		for _, pr := range pairs {
			if err := copyExactCast(pr.dst, hf, pr.name); err != nil {
				return nil, err
			}
		}
	}
	if err := normPlusOne(&m.FinalNorm.Weight, prefix+"norm.weight"); err != nil {
		return nil, err
	}
	if !cfg.TieEmbed {
		if err := copyExactCast(m.LMHead.W, hf, "lm_head.weight"); err != nil {
			return nil, err
		}
	}
	return m, nil
}
