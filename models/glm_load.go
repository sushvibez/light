package models

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"light/device"
	"light/dtype"
	"light/hub"
)

// glmHFConfig is a glm / glm4 config.json (GLM-4-9B-chat-hf, GLM-4-0414).
type glmHFConfig struct {
	ModelType           string        `json:"model_type"`
	VocabSize           int           `json:"vocab_size"`
	HiddenSize          int           `json:"hidden_size"`
	IntermediateSize    int           `json:"intermediate_size"`
	NumHiddenLayers     int           `json:"num_hidden_layers"`
	NumAttentionHeads   int           `json:"num_attention_heads"`
	NumKeyValueHeads    int           `json:"num_key_value_heads"`
	HeadDim             int           `json:"head_dim"`
	PartialRotaryFactor *float64      `json:"partial_rotary_factor"`
	AttentionBias       *bool         `json:"attention_bias"`
	HiddenAct           string        `json:"hidden_act"`
	RopeTheta           float64       `json:"rope_theta"`
	RopeParameters      *hfRopeParams `json:"rope_parameters"`
	RopeScaling         *hfRopeParams `json:"rope_scaling"`
	RmsNormEps          float64       `json:"rms_norm_eps"`
	MaxPosition         int           `json:"max_position_embeddings"`
	TieWordEmbeddings   bool          `json:"tie_word_embeddings"`
}

// fetchGLMConfig maps a glm / glm4 config onto LlamaConfig. transformers'
// defaults apply to absent keys: partial_rotary_factor 0.5, attention_bias
// true, head_dim 128.
func fetchGLMConfig(repo string) (LlamaConfig, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return LlamaConfig{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return LlamaConfig{}, err
	}
	var hf glmHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return LlamaConfig{}, fmt.Errorf("glm config: %w", err)
	}
	if hf.ModelType != "glm" && hf.ModelType != "glm4" {
		return LlamaConfig{}, fmt.Errorf("glm: model_type %q", hf.ModelType)
	}
	if a := hf.HiddenAct; a != "" && a != "silu" {
		return LlamaConfig{}, fmt.Errorf("glm: hidden_act %q not supported", a)
	}
	partial := 0.5
	if hf.PartialRotaryFactor != nil {
		partial = *hf.PartialRotaryFactor
	}
	rp := hf.RopeParameters
	if rp == nil {
		rp = hf.RopeScaling
	}
	base := hf.RopeTheta
	if rp != nil {
		if rp.RopeTheta > 0 {
			base = rp.RopeTheta
		}
		if t := rp.RopeType; t != "" && t != "default" {
			return LlamaConfig{}, fmt.Errorf("glm: rope_type %q not supported", t)
		}
	}
	if base == 0 {
		base = 10000
	}
	headDim := hf.HeadDim
	if headDim == 0 {
		headDim = 128
	}
	bias := hf.AttentionBias == nil || *hf.AttentionBias
	eps := hf.RmsNormEps
	if eps == 0 {
		eps = 1.5625e-07 // GlmConfig default
	}
	return LlamaConfig{
		VocabSize: hf.VocabSize, HiddenSize: hf.HiddenSize, IntermediateSize: hf.IntermediateSize,
		NumLayers: hf.NumHiddenLayers, NumHeads: hf.NumAttentionHeads, NumKVHeads: hf.NumKeyValueHeads,
		HeadDim: headDim, MaxPosition: hf.MaxPosition, RopeBase: base, RmsEps: eps,
		TieEmbed: hf.TieWordEmbeddings, QKVBias: bias, PartialRotary: partial,
		PostNorms: hf.ModelType == "glm4",
	}, nil
}

// LoadGLMFromHF loads GLM-4 (model_type glm: THUDM/glm-4-9b-chat-hf; glm4:
// GLM-4-0414, which adds a post-norm after attention and after the MLP) as a
// *Llama. Besides the post-norms, GLM-4 differs from Llama in its fused
// gate_up_proj (split on load, gate rows first) and in rotating only the
// first half of each head (partial_rotary_factor 0.5) in interleaved pairs
// (2i, 2i+1). Light's partial RoPE is rotate-half, so the rotary rows of
// q_proj / k_proj (weights and biases) are reordered from pair order to
// half order on load, which leaves every q·k unchanged.
func LoadGLMFromHF(repo string, dev device.Device, weightDT dtype.DType) (*Llama, error) {
	cfg, err := fetchGLMConfig(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	cfg.SkipInit = true
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadGLMFromHF: %w", err)
	}
	if err := splitPhi3Fused(hf, cfg); err != nil {
		return nil, err
	}
	m, err := NewLlama(cfg, dev)
	if err != nil {
		return nil, err
	}
	rot := m.rotaryDim()
	for i := 0; i < cfg.NumLayers; i++ {
		p := "model.layers." + strconv.Itoa(i) + ".self_attn."
		for _, name := range []string{"q_proj.weight", "q_proj.bias", "k_proj.weight", "k_proj.bias"} {
			w, ok := hf[p+name]
			if !ok {
				if name[len(name)-4:] == "bias" && !cfg.QKVBias {
					continue
				}
				return nil, fmt.Errorf("LoadGLMFromHF: missing tensor %q", p+name)
			}
			r, err := deinterleaveRows(w, cfg.HeadDim, 0, rot)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", p+name, err)
			}
			hf[p+name] = r
		}
	}
	if err := mapHFToLlama(m, hf); err != nil {
		return nil, err
	}
	if cfg.PostNorms {
		for i, b := range m.Blocks {
			p := "model.layers." + strconv.Itoa(i) + "."
			if err := copyExactCast(b.PostAttnNorm.Weight, hf, p+"post_self_attn_layernorm.weight"); err != nil {
				return nil, err
			}
			if err := copyExactCast(b.PostMLPNorm.Weight, hf, p+"post_mlp_layernorm.weight"); err != nil {
				return nil, err
			}
		}
	}
	return m, nil
}
