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

// museGlimmerTextConfig is text_config of a muse_glimmer config.json.
type museGlimmerTextConfig struct {
	VocabSize         int       `json:"vocab_size"`
	HiddenSize        int       `json:"hidden_size"`
	IntermediateSize  int       `json:"intermediate_size"`
	NumHiddenLayers   int       `json:"num_hidden_layers"`
	NumAttentionHeads int       `json:"num_attention_heads"`
	NumKeyValueHeads  int       `json:"num_key_value_heads"`
	HeadDim           int       `json:"head_dim"`
	HiddenActivation  string    `json:"hidden_activation"`
	MaxPosition       int       `json:"max_position_embeddings"`
	RmsNormEps        float64   `json:"rms_norm_eps"`
	PostNormEps       float64   `json:"post_norm_eps"`
	SlidingWindow     int       `json:"sliding_window"`
	LayerTypes        []string  `json:"layer_types"`
	LayerRopeTheta    []float64 `json:"layer_rope_theta"`
	FinalLogitSoftcap float64   `json:"final_logit_softcapping"`
	QKScaleFactor     float64   `json:"qk_scale_factor"`
	OutputMultiplier  float64   `json:"output_multiplier"`
	TieWordEmbeddings bool      `json:"tie_word_embeddings"`
	AttentionBias     bool      `json:"attention_bias"`
	RopeParameters    *struct {
		RopeType string `json:"rope_type"`
	} `json:"rope_parameters"`
}

// fetchMuseGlimmerConfig maps config.json (text_config of the multimodal
// checkpoint) onto the Gemma 3 skeleton's switches.
func fetchMuseGlimmerConfig(repo string) (Gemma3Config, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return Gemma3Config{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return Gemma3Config{}, err
	}
	var top struct {
		TextConfig json.RawMessage `json:"text_config"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return Gemma3Config{}, fmt.Errorf("muse_glimmer config: %w", err)
	}
	src := raw
	if len(top.TextConfig) > 0 {
		src = top.TextConfig
	}
	var hf museGlimmerTextConfig
	if err := json.Unmarshal(src, &hf); err != nil {
		return Gemma3Config{}, fmt.Errorf("muse_glimmer text_config: %w", err)
	}
	n := hf.NumHiddenLayers
	if len(hf.LayerTypes) != n || len(hf.LayerRopeTheta) != n {
		return Gemma3Config{}, fmt.Errorf("muse_glimmer: %d layers, %d layer_types, %d layer_rope_theta",
			n, len(hf.LayerTypes), len(hf.LayerRopeTheta))
	}
	if hf.AttentionBias {
		return Gemma3Config{}, fmt.Errorf("muse_glimmer: attention_bias not supported")
	}
	if rp := hf.RopeParameters; rp != nil && rp.RopeType != "" && rp.RopeType != "default" {
		return Gemma3Config{}, fmt.Errorf("muse_glimmer: rope_type %q not supported", rp.RopeType)
	}
	if a := hf.HiddenActivation; a != "" && a != "silu" {
		return Gemma3Config{}, fmt.Errorf("muse_glimmer: hidden_activation %q not supported", a)
	}
	sliding := make([]bool, n)
	for i, t := range hf.LayerTypes {
		switch t {
		case "sliding_attention":
			sliding[i] = true
		case "full_attention":
		default:
			return Gemma3Config{}, fmt.Errorf("muse_glimmer: layer %d type %q", i, t)
		}
	}
	eps := hf.RmsNormEps
	if eps == 0 {
		eps = 1e-5
	}
	return Gemma3Config{
		VocabSize: hf.VocabSize, HiddenSize: hf.HiddenSize, IntermediateSize: hf.IntermediateSize,
		NumLayers: n, NumHeads: hf.NumAttentionHeads, NumKVHeads: hf.NumKeyValueHeads, HeadDim: hf.HeadDim,
		MaxPosition: hf.MaxPosition, RmsEps: eps, SlidingWindow: hf.SlidingWindow, SlidingLayers: sliding,
		FinalLogitSoftcap: hf.FinalLogitSoftcap,
		EmbedNorm:         true, LayerRopeBase: hf.LayerRopeTheta, AttnOutputGate: true, SiLUMLP: true,
		QScale: hf.QKScaleFactor, PostNormEps: hf.PostNormEps, OutputMultiplier: hf.OutputMultiplier,
		UntiedHead: !hf.TieWordEmbeddings,
	}, nil
}

// LoadMuseGlimmerFromHF loads the text model of a muse_glimmer checkpoint
// (meta-models/Muse-Glimmer-30B or a local directory) onto the Gemma 3
// skeleton; the vision tower is not loaded. weightDT is F32 or BF16.
//
// Norm conventions (transformers' modeling_muse_glimmer): the four sandwich
// norms per layer are centred, (1 + w); the final norm is a plain RMSNorm
// with weight w; the embedding norm and the q / k norms carry no weight
// (ones here; the query scale factor is Cfg.QScale, applied in F32).
func LoadMuseGlimmerFromHF(repo string, dev device.Device, weightDT dtype.DType) (*Gemma3, error) {
	cfg, err := fetchMuseGlimmerConfig(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadMuseGlimmerFromHF: %w", err)
	}
	m, err := NewGemma3(cfg, dev)
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
		return nil, fmt.Errorf("LoadMuseGlimmerFromHF: no embed_tokens.weight under model.language_model. or model.")
	}
	act := m.Cfg.activationDType()
	// norm installs a norm weight, (1 + w) when plusOne, formed in F32 and
	// stored at the activation dtype (the fused rmsnorm kernel's contract,
	// as for Gemma).
	norm := func(dst **tensor.Tensor, name string, plusOne bool) error {
		src, ok := hf[name]
		if !ok {
			return fmt.Errorf("LoadMuseGlimmerFromHF: missing tensor %q", name)
		}
		w, err := ops.Cast(src, dtype.F32)
		if err != nil {
			return err
		}
		if plusOne {
			one, err := tensor.New([]float32{1}, []int{1}, w.Device())
			if err != nil {
				return err
			}
			if w, err = ops.Add(w, one); err != nil {
				return err
			}
		}
		if act != dtype.F32 {
			if w, err = ops.Cast(w, act); err != nil {
				return err
			}
		}
		if !w.Shape().Equal((*dst).Shape()) {
			return fmt.Errorf("%s: shape %v, model %v", name, w.Shape(), (*dst).Shape())
		}
		*dst, err = w.To(dev)
		return err
	}
	// ones makes a norm weightless (weight 1 at the activation dtype).
	ones := func(dst **tensor.Tensor) error {
		data := make([]float32, (*dst).NumElements())
		for i := range data {
			data[i] = 1
		}
		w, err := tensor.New(data, (*dst).Shape().Dims(), dev)
		if err != nil {
			return err
		}
		if act != dtype.F32 {
			if w, err = ops.Cast(w, act); err != nil {
				return err
			}
		}
		*dst = w
		return nil
	}
	if err := copyExactCast(m.TokenEmb.Weight, hf, prefix+"embed_tokens.weight"); err != nil {
		return nil, err
	}
	if err := ones(&m.embNorm.Weight); err != nil {
		return nil, err
	}
	for i, b := range m.Blocks {
		p := prefix + "layers." + strconv.Itoa(i) + "."
		for _, pr := range []struct {
			n    **tensor.Tensor
			name string
		}{
			{&b.InNorm.Weight, p + "input_layernorm.weight"},
			{&b.PostAttnNorm.Weight, p + "post_attention_layernorm.weight"},
			{&b.PreFFNorm.Weight, p + "pre_feedforward_layernorm.weight"},
			{&b.PostFFNorm.Weight, p + "post_feedforward_layernorm.weight"},
		} {
			if err := norm(pr.n, pr.name, true); err != nil {
				return nil, err
			}
		}
		if err := ones(&b.QNorm.Weight); err != nil {
			return nil, err
		}
		if err := ones(&b.KNorm.Weight); err != nil {
			return nil, err
		}
		for _, pr := range []struct {
			l    *tensor.Tensor
			name string
		}{
			{b.QProj.W, p + "self_attn.q_proj.weight"},
			{b.KProj.W, p + "self_attn.k_proj.weight"},
			{b.VProj.W, p + "self_attn.v_proj.weight"},
			{b.OProj.W, p + "self_attn.o_proj.weight"},
			{b.AttnGate.W, p + "self_attn.gate_proj.weight"},
			{b.GateProj.W, p + "mlp.gate_proj.weight"},
			{b.UpProj.W, p + "mlp.up_proj.weight"},
			{b.DownProj.W, p + "mlp.down_proj.weight"},
		} {
			if err := copyExactCast(pr.l, hf, pr.name); err != nil {
				return nil, err
			}
		}
	}
	if err := norm(&m.FinalNorm.Weight, prefix+"norm.weight", false); err != nil {
		return nil, err
	}
	if cfg.UntiedHead {
		if err := copyExactCast(m.LMHead.W, hf, "lm_head.weight"); err != nil {
			return nil, err
		}
	}
	return m, nil
}
