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

type gemmaHFConfig struct {
	ModelType            string         `json:"model_type"`
	TextConfig           *gemmaHFConfig `json:"text_config"` // multimodal checkpoints (model_type gemma3) nest the text model
	VocabSize            int            `json:"vocab_size"`
	HiddenSize           int            `json:"hidden_size"`
	IntermediateSize     int            `json:"intermediate_size"`
	NumHiddenLayers      int            `json:"num_hidden_layers"`
	NumAttentionHeads    int            `json:"num_attention_heads"`
	NumKeyValueHeads     int            `json:"num_key_value_heads"`
	HeadDim              int            `json:"head_dim"`
	MaxPositionEmbed     int            `json:"max_position_embeddings"`
	RopeTheta            float64        `json:"rope_theta"`
	RopeLocalBaseFreq    float64        `json:"rope_local_base_freq"`
	RmsNormEps           float64        `json:"rms_norm_eps"`
	QueryPreAttnScalar   float64        `json:"query_pre_attn_scalar"`
	SlidingWindow        int            `json:"sliding_window"`
	SlidingWindowPattern int            `json:"sliding_window_pattern"`
	LayerTypes           []string       `json:"layer_types"`
	FinalLogitSoftcap    *float64       `json:"final_logit_softcapping"`
	AttnLogitSoftcap     *float64       `json:"attn_logit_softcapping"`
	RopeScaling          *struct {
		Type    string  `json:"rope_type"`
		TypeOld string  `json:"type"`
		Factor  float64 `json:"factor"`
	} `json:"rope_scaling"`
}

func fetchGemma3Config(repo string) (Gemma3Config, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return Gemma3Config{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return Gemma3Config{}, err
	}
	var hf gemmaHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return Gemma3Config{}, fmt.Errorf("fetchGemma3Config: %w", err)
	}
	if hf.ModelType == "gemma3" && hf.TextConfig != nil {
		// Gemma3ForConditionalGeneration (4B/12B/27B): text-only load of the
		// language model; the SigLIP tower and projector are skipped.
		hf = *hf.TextConfig
		hf.ModelType = "gemma3_text"
	}
	if hf.ModelType != "gemma3_text" && hf.ModelType != "gemma2" {
		return Gemma3Config{}, fmt.Errorf("fetchGemma3Config: model_type %q not supported (gemma3, gemma3_text, gemma2)", hf.ModelType)
	}
	ropeScale := 1.0
	if rs := hf.RopeScaling; rs != nil && rs.Factor != 0 && rs.Factor != 1 {
		if t := rs.Type + rs.TypeOld; t != "linear" {
			return Gemma3Config{}, fmt.Errorf("fetchGemma3Config: rope_scaling type %q not supported (linear)", t)
		}
		ropeScale = rs.Factor
	}
	if hf.HeadDim == 0 {
		hf.HeadDim = hf.HiddenSize / hf.NumAttentionHeads
	}
	if hf.NumKeyValueHeads == 0 {
		hf.NumKeyValueHeads = hf.NumAttentionHeads
	}
	if hf.RopeTheta == 0 {
		hf.RopeTheta = 10000
	}
	if hf.RopeLocalBaseFreq == 0 {
		hf.RopeLocalBaseFreq = hf.RopeTheta
	}
	if hf.RmsNormEps == 0 {
		hf.RmsNormEps = 1e-6
	}
	cfg := Gemma3Config{
		VocabSize: hf.VocabSize, HiddenSize: hf.HiddenSize, IntermediateSize: hf.IntermediateSize,
		NumLayers: hf.NumHiddenLayers, NumHeads: hf.NumAttentionHeads, NumKVHeads: hf.NumKeyValueHeads,
		HeadDim: hf.HeadDim, MaxPosition: hf.MaxPositionEmbed, RopeBase: hf.RopeTheta,
		RopeLocalBase: hf.RopeLocalBaseFreq, RmsEps: hf.RmsNormEps, QueryPreAttnScalar: hf.QueryPreAttnScalar,
		SlidingWindow: hf.SlidingWindow, RopeScale: ropeScale,
	}
	cfg.NoQKNorm = hf.ModelType == "gemma2"
	if hf.FinalLogitSoftcap != nil {
		cfg.FinalLogitSoftcap = *hf.FinalLogitSoftcap
	}
	if hf.AttnLogitSoftcap != nil {
		cfg.AttnLogitSoftcap = *hf.AttnLogitSoftcap
	}
	switch {
	case len(hf.LayerTypes) > 0:
		cfg.SlidingLayers = make([]bool, len(hf.LayerTypes))
		for i, t := range hf.LayerTypes {
			cfg.SlidingLayers[i] = t == "sliding_attention"
		}
	case hf.SlidingWindowPattern > 0:
		// Gemma 3: every pattern-th layer is global, the rest local.
		cfg.SlidingLayers = make([]bool, hf.NumHiddenLayers)
		for i := range cfg.SlidingLayers {
			cfg.SlidingLayers[i] = (i+1)%hf.SlidingWindowPattern != 0
		}
	case hf.ModelType == "gemma2":
		// Gemma 2 alternates local (even) / global (odd).
		cfg.SlidingLayers = make([]bool, hf.NumHiddenLayers)
		for i := range cfg.SlidingLayers {
			cfg.SlidingLayers[i] = i%2 == 0
		}
	}
	return cfg, nil
}

// LoadGemma3FromHF loads a Gemma 3 text checkpoint (e.g. `google/gemma-3-1b-it`
// or the ungated mirror `unsloth/gemma-3-1b-it`). weightDT is dtype.F32 or
// dtype.BF16; norm gains are always kept as F32 (1+w) like the reference.
func LoadGemma3FromHF(repo string, dev device.Device, weightDT dtype.DType) (*Gemma3, error) {
	cfg, err := fetchGemma3Config(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadGemma3FromHF: %w", err)
	}
	m, err := NewGemma3(cfg, dev)
	if err != nil {
		return nil, err
	}
	// Multimodal checkpoints prefix the text model: language_model.model.*
	// (hub layout) or model.language_model.* (transformers ≥ 4.52 layout).
	prefix := "model."
	for _, p := range []string{"model.", "language_model.model.", "model.language_model."} {
		if _, ok := hf[p+"embed_tokens.weight"]; ok {
			prefix = p
			break
		}
	}
	if err := copyExactCast(m.TokenEmb.Weight, hf, prefix+"embed_tokens.weight"); err != nil {
		return nil, err
	}
	normPlusOne := func(dst **tensor.Tensor, name string) error {
		if err := copyExactCast(*dst, hf, name); err != nil {
			return err
		}
		one, err := tensor.New([]float32{1}, []int{1}, dev)
		if err != nil {
			return err
		}
		w, err := ops.Add(*dst, one)
		if err != nil {
			return err
		}
		// The fused rmsnorm kernel wants weight and activation at one
		// dtype; HF keeps (1+w) in F32 and casts the product, so BF16 here
		// costs one extra rounding of the gain (≈0.4 %), within BF16 noise.
		if act := cfg.activationDType(); act != dtype.F32 {
			if w, err = ops.Cast(w, act); err != nil {
				return err
			}
		}
		*dst = w
		return nil
	}
	for i, b := range m.Blocks {
		p := prefix + "layers." + strconv.Itoa(i) + "."
		for _, pair := range []struct {
			n    **tensor.Tensor
			name string
		}{
			{&b.InNorm.Weight, p + "input_layernorm.weight"},
			{&b.PostAttnNorm.Weight, p + "post_attention_layernorm.weight"},
			{&b.PreFFNorm.Weight, p + "pre_feedforward_layernorm.weight"},
			{&b.PostFFNorm.Weight, p + "post_feedforward_layernorm.weight"},
		} {
			if err := normPlusOne(pair.n, pair.name); err != nil {
				return nil, err
			}
		}
		if b.QNorm != nil {
			if err := normPlusOne(&b.QNorm.Weight, p+"self_attn.q_norm.weight"); err != nil {
				return nil, err
			}
			if err := normPlusOne(&b.KNorm.Weight, p+"self_attn.k_norm.weight"); err != nil {
				return nil, err
			}
		}
		for _, pair := range []struct {
			l    *tensor.Tensor
			name string
		}{
			{b.QProj.W, p + "self_attn.q_proj.weight"},
			{b.KProj.W, p + "self_attn.k_proj.weight"},
			{b.VProj.W, p + "self_attn.v_proj.weight"},
			{b.OProj.W, p + "self_attn.o_proj.weight"},
			{b.GateProj.W, p + "mlp.gate_proj.weight"},
			{b.UpProj.W, p + "mlp.up_proj.weight"},
			{b.DownProj.W, p + "mlp.down_proj.weight"},
		} {
			if err := copyExactCast(pair.l, hf, pair.name); err != nil {
				return nil, err
			}
		}
	}
	if err := normPlusOne(&m.FinalNorm.Weight, prefix+"norm.weight"); err != nil {
		return nil, err
	}
	return m, nil
}

// LoadChatModelFromHF dispatches on config.json's model_type across every
// decoder family Light supports and returns the CausalLM interface.
func LoadChatModelFromHF(repo string, dev device.Device, weightDT dtype.DType) (CausalLM, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var hf struct {
		ModelType string `json:"model_type"`
	}
	if err := json.Unmarshal(raw, &hf); err != nil {
		return nil, err
	}
	switch hf.ModelType {
	case "gemma3", "gemma3_text", "gemma2":
		return LoadGemma3FromHF(repo, dev, weightDT)
	case "gemma4", "gemma4_text":
		return LoadGemma4FromHF(repo, dev, weightDT)
	case "qwen3_5", "qwen3_5_text":
		return LoadQwen35FromHF(repo, dev, weightDT)
	case "muse_glimmer", "muse_glimmer_text":
		return LoadMuseGlimmerFromHF(repo, dev, weightDT)
	case "deepseek_v2":
		return LoadDeepseekV2FromHF(repo, dev, weightDT)
	case "minimax":
		return LoadMiniMaxFromHF(repo, dev, weightDT)
	case "falcon", "RefinedWebModel", "RefinedWeb":
		return LoadFalconFromHF(repo, dev, weightDT)
	default:
		return LoadCausalLMFromHF(repo, dev, weightDT)
	}
}
