package models

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/tensor"
)

// sparkHFConfig is config.json of `model_type: spark2_5` (XHToken's
// Spark-X 2.5 / 2.6).
type sparkHFConfig struct {
	VocabSize            int      `json:"vocab_size"`
	HiddenSize           int      `json:"hidden_size"`
	IntermediateSize     int      `json:"intermediate_size"`
	NumHiddenLayers      int      `json:"num_hidden_layers"`
	NumAttentionHeads    int      `json:"num_attention_heads"`
	NumKeyValueHeads     int      `json:"num_key_value_heads"`
	HeadDim              int      `json:"head_dim"`
	RmsNormEps           float64  `json:"rms_norm_eps"`
	MaxPositionEmbedding int      `json:"max_position_embeddings"`
	TieWordEmbeddings    *bool    `json:"tie_word_embeddings"`
	SlidingWindow        int      `json:"sliding_window"`
	LayerTypes           []string `json:"layer_types"`
	HiddenAct            string   `json:"hidden_act"`
	AttentionBias        bool     `json:"attention_bias"`
	MLPBias              bool     `json:"mlp_bias"`
	HeadwiseGate         bool     `json:"headwise_attn_output_gate"`
	GateActMode          string   `json:"gate_attn_act_mode"`
	RopeParameters       map[string]struct {
		RopeTheta     float64  `json:"rope_theta"`
		PartialRotary *float64 `json:"partial_rotary_factor"`
		RopeType      string   `json:"rope_type"`
	} `json:"rope_parameters"`
}

func fetchSparkConfig(repo string) (LlamaConfig, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return LlamaConfig{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return LlamaConfig{}, err
	}
	var hf sparkHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return LlamaConfig{}, fmt.Errorf("spark config: %w", err)
	}
	switch {
	case hf.HiddenAct != "gelu":
		return LlamaConfig{}, fmt.Errorf("spark: hidden_act %q not supported (gelu)", hf.HiddenAct)
	case hf.AttentionBias || hf.MLPBias:
		return LlamaConfig{}, fmt.Errorf("spark: attention / MLP biases not supported")
	case hf.HeadwiseGate && hf.GateActMode != "sigmoid":
		return LlamaConfig{}, fmt.Errorf("spark: gate_attn_act_mode %q not supported (sigmoid)", hf.GateActMode)
	}
	full, ok := hf.RopeParameters["full_attention"]
	if !ok {
		return LlamaConfig{}, fmt.Errorf("spark: no full_attention rope_parameters")
	}
	sliding := hf.RopeParameters["sliding_attention"]
	for k, p := range hf.RopeParameters {
		if p.RopeType != "" && p.RopeType != "default" {
			return LlamaConfig{}, fmt.Errorf("spark: %s rope_type %q not supported", k, p.RopeType)
		}
	}
	if sliding.PartialRotary != nil && *sliding.PartialRotary != 1 {
		return LlamaConfig{}, fmt.Errorf("spark: partial rotary on sliding layers not supported")
	}
	cfg := LlamaConfig{
		VocabSize: hf.VocabSize, HiddenSize: hf.HiddenSize, IntermediateSize: hf.IntermediateSize,
		NumLayers: hf.NumHiddenLayers, NumHeads: hf.NumAttentionHeads, NumKVHeads: hf.NumKeyValueHeads,
		HeadDim: hf.HeadDim, MaxPosition: hf.MaxPositionEmbedding, RopeBase: full.RopeTheta,
		RmsEps: hf.RmsNormEps, TieEmbed: true, SlidingFullRotary: true,
		HeadwiseGate: hf.HeadwiseGate, GELUMLP: true,
	}
	if full.PartialRotary != nil {
		cfg.PartialRotary = *full.PartialRotary
	}
	if hf.TieWordEmbeddings != nil {
		cfg.TieEmbed = *hf.TieWordEmbeddings
	}
	if cfg.HeadDim == 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	if hf.SlidingWindow > 0 && len(hf.LayerTypes) > 0 {
		if len(hf.LayerTypes) != cfg.NumLayers {
			return LlamaConfig{}, fmt.Errorf("spark: %d layer_types for %d layers", len(hf.LayerTypes), cfg.NumLayers)
		}
		cfg.SlidingWindow = hf.SlidingWindow
		cfg.SlidingLayers = make([]bool, cfg.NumLayers)
		for i, t := range hf.LayerTypes {
			cfg.SlidingLayers[i] = t == "sliding_attention"
		}
		cfg.SlidingRopeBase = sliding.RopeTheta
	}
	return cfg, nil
}

// LoadSparkFromHF loads a Spark2_5ForCausalLM checkpoint (XHToken/Spark-X*):
// a Llama-family decoder with a fused q_k_v_proj, a per-head sigmoid output
// gate (g_proj), a GELU-gated MLP, and sliding layers interleaved with
// partial-RoPE full layers.
func LoadSparkFromHF(repo string, dev device.Device, weightDT dtype.DType) (*Llama, error) {
	cfg, err := fetchSparkConfig(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	cfg.SkipInit = true
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadSparkFromHF: %w", err)
	}
	rename := func(from, to string) {
		if t, ok := hf[from]; ok {
			hf[to] = t
			delete(hf, from)
		}
	}
	rename("model.embedding.weight", "model.embed_tokens.weight")
	qRows, kvRows := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	for i := range cfg.NumLayers {
		p := "model.layers." + strconv.Itoa(i) + ".self_attn."
		rename(p+"out_proj.weight", p+"o_proj.weight")
		fused, ok := hf[p+"q_k_v_proj.weight"]
		if !ok {
			return nil, fmt.Errorf("LoadSparkFromHF: missing %sq_k_v_proj.weight", p)
		}
		if fused.Rank() != 2 || fused.Dim(0) != qRows+2*kvRows {
			return nil, fmt.Errorf("%sq_k_v_proj.weight: shape %v, want (%d, D)", p, fused.Shape(), qRows+2*kvRows)
		}
		off := 0
		for _, part := range []struct {
			name string
			rows int
		}{{"q_proj", qRows}, {"k_proj", kvRows}, {"v_proj", kvRows}} {
			v, err := fused.Narrow(0, off, part.rows)
			if err != nil {
				return nil, err
			}
			if hf[p+part.name+".weight"], err = v.Contiguous(); err != nil {
				return nil, err
			}
			off += part.rows
		}
		delete(hf, p+"q_k_v_proj.weight")
	}
	m, err := NewLlama(cfg, dev)
	if err != nil {
		return nil, err
	}
	gates := map[string]*tensor.Tensor{}
	for i := range cfg.NumLayers {
		name := "model.layers." + strconv.Itoa(i) + ".self_attn.g_proj.weight"
		if t, ok := hf[name]; ok {
			gates[name] = t
			delete(hf, name)
		}
	}
	if err := mapHFToLlama(m, hf); err != nil {
		return nil, err
	}
	if cfg.HeadwiseGate {
		linDT := cfg.WeightDType
		if linDT != dtype.BF16 {
			linDT = dtype.F32
		}
		for i, b := range m.Blocks {
			name := "model.layers." + strconv.Itoa(i) + ".self_attn.g_proj.weight"
			if err := copyLinearW(b.HeadGate, gates, name, linDT, dev); err != nil {
				return nil, err
			}
		}
	}
	return m, nil
}
