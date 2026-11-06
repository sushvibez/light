package models

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/ops"
	"light/tensor"
)

type embeddingGemma2HFConfig struct {
	ModelType  string `json:"model_type"`
	TextConfig *struct {
		VocabSize        int      `json:"vocab_size"`
		HiddenSize       int      `json:"hidden_size"`
		IntermediateSize int      `json:"intermediate_size"`
		NumHiddenLayers  int      `json:"num_hidden_layers"`
		NumHeads         int      `json:"num_attention_heads"`
		NumKVHeads       int      `json:"num_key_value_heads"`
		HeadDim          int      `json:"head_dim"`
		MaxPosition      int      `json:"max_position_embeddings"`
		RmsEps           float64  `json:"rms_norm_eps"`
		SlidingWindow    int      `json:"sliding_window"`
		LayerTypes       []string `json:"layer_types"`
		PLEDim           int      `json:"hidden_size_per_layer_input"`
		EmbeddingDim     int      `json:"embedding_dim"`
		PerLayerConfig   map[string]struct {
			HeadDim    int `json:"head_dim"`
			NumKVHeads int `json:"num_key_value_heads"`
		} `json:"per_layer_config"`
		RopeParameters map[string]struct {
			RopeTheta float64 `json:"rope_theta"`
			RopeType  string  `json:"rope_type"`
		} `json:"rope_parameters"`
	} `json:"text_config"`
}

func fetchEmbeddingGemma2Config(repo string) (EmbeddingGemma2Config, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return EmbeddingGemma2Config{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return EmbeddingGemma2Config{}, err
	}
	var hf embeddingGemma2HFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return EmbeddingGemma2Config{}, fmt.Errorf("fetchEmbeddingGemma2Config: %w", err)
	}
	if hf.ModelType != "embedding_gemma2" || hf.TextConfig == nil {
		return EmbeddingGemma2Config{}, fmt.Errorf("fetchEmbeddingGemma2Config: model_type %q not supported (embedding_gemma2 with text_config)", hf.ModelType)
	}
	t := hf.TextConfig
	cfg := EmbeddingGemma2Config{
		VocabSize: t.VocabSize, HiddenSize: t.HiddenSize, IntermediateSize: t.IntermediateSize,
		NumLayers: t.NumHiddenLayers, NumHeads: t.NumHeads, NumKVHeads: t.NumKVHeads,
		HeadDim: t.HeadDim, MaxPosition: t.MaxPosition, RopeBase: 1e6, RopeLocalBase: 1e4,
		RmsEps: t.RmsEps, SlidingWindow: t.SlidingWindow, PLEDim: t.PLEDim, EmbeddingDim: t.EmbeddingDim,
	}
	if cfg.HeadDim == 0 {
		cfg.HeadDim = 256
	}
	if cfg.NumKVHeads == 0 {
		cfg.NumKVHeads = cfg.NumHeads
	}
	if cfg.RmsEps == 0 {
		cfg.RmsEps = 1e-6
	}
	if cfg.MaxPosition == 0 {
		cfg.MaxPosition = 2048
	}
	if cfg.SlidingWindow == 0 {
		cfg.SlidingWindow = 512
	}
	if cfg.PLEDim == 0 {
		cfg.PLEDim = 512
	}
	if cfg.EmbeddingDim == 0 {
		cfg.EmbeddingDim = 768
	}
	cfg.SlidingLayers = make([]bool, cfg.NumLayers)
	for i := range cfg.SlidingLayers {
		if len(t.LayerTypes) == cfg.NumLayers {
			cfg.SlidingLayers[i] = t.LayerTypes[i] == "sliding_attention"
		} else {
			cfg.SlidingLayers[i] = (i+1)%6 != 0 // default sliding_window_pattern 6
		}
	}
	// per_layer_config: {"05": {head_dim, num_key_value_heads}, ...} for the
	// full-attention layers; every full-attention layer must agree.
	cfg.GlobalHeadDim, cfg.NumGlobalKVHeads = cfg.HeadDim, cfg.NumKVHeads
	for key, pl := range t.PerLayerConfig {
		i, err := strconv.Atoi(key)
		if err != nil || i < 0 || i >= cfg.NumLayers {
			return EmbeddingGemma2Config{}, fmt.Errorf("fetchEmbeddingGemma2Config: per_layer_config key %q", key)
		}
		if cfg.SlidingLayers[i] {
			return EmbeddingGemma2Config{}, fmt.Errorf("fetchEmbeddingGemma2Config: per_layer_config for sliding layer %d not supported", i)
		}
		hd, kv := pl.HeadDim, pl.NumKVHeads
		if hd == 0 {
			hd = cfg.HeadDim
		}
		if kv == 0 {
			kv = cfg.NumKVHeads
		}
		if cfg.GlobalHeadDim != cfg.HeadDim && cfg.GlobalHeadDim != hd || cfg.NumGlobalKVHeads != cfg.NumKVHeads && cfg.NumGlobalKVHeads != kv {
			return EmbeddingGemma2Config{}, fmt.Errorf("fetchEmbeddingGemma2Config: full-attention layers disagree on head geometry")
		}
		cfg.GlobalHeadDim, cfg.NumGlobalKVHeads = hd, kv
	}
	for lt, rp := range t.RopeParameters {
		if rp.RopeType != "" && rp.RopeType != "default" {
			return EmbeddingGemma2Config{}, fmt.Errorf("fetchEmbeddingGemma2Config: %s rope_type %q not supported", lt, rp.RopeType)
		}
		switch {
		case lt == "full_attention" && rp.RopeTheta != 0:
			cfg.RopeBase = rp.RopeTheta
		case lt == "sliding_attention" && rp.RopeTheta != 0:
			cfg.RopeLocalBase = rp.RopeTheta
		}
	}
	return cfg, nil
}

// LoadEmbeddingGemma2FromHF loads the text encoder of an EmbeddingGemma 2
// checkpoint (the vision and audio towers are skipped) one tensor at a
// time. weightDT is dtype.F32 or dtype.BF16.
func LoadEmbeddingGemma2FromHF(repo string, dev device.Device, weightDT dtype.DType) (*EmbeddingGemma2, error) {
	cfg, err := fetchEmbeddingGemma2Config(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	paths, err := hub.DownloadModelSafetensors(repo, "main")
	if err != nil {
		return nil, err
	}
	idx, err := openSafetensorsIndex(paths)
	if err != nil {
		return nil, err
	}
	defer idx.Close()
	m, err := NewEmbeddingGemma2(cfg, dev)
	if err != nil {
		return nil, err
	}
	prefix := "language_model."
	for _, p := range []string{"language_model.", "model.language_model.", "model."} {
		if idx.has(p + "embed_tokens.weight") {
			prefix = p
			break
		}
	}
	if err := idx.copyInto(m.TokenEmb.Weight, prefix+"embed_tokens.weight"); err != nil {
		return nil, err
	}
	// per_layer_model_projection(h) · D^-½: fold the scale into the weight.
	w, err := idx.load(prefix+"ple.per_layer_model_projection.weight", dtype.F32, dev)
	if err != nil {
		return nil, err
	}
	s, err := tensor.New([]float32{float32(1 / math.Sqrt(float64(cfg.HiddenSize)))}, []int{1}, dev)
	if err != nil {
		return nil, err
	}
	if w, err = ops.Mul(w, s); err != nil {
		return nil, err
	}
	if w.DType() != m.PLEProj.W.DType() {
		if w, err = ops.Cast(w, m.PLEProj.W.DType()); err != nil {
			return nil, err
		}
	}
	if err := m.PLEProj.W.CopyFrom(w); err != nil {
		return nil, fmt.Errorf("ple.per_layer_model_projection: %w", err)
	}
	if err := idx.copyInto(m.PLENorm.Weight, prefix+"ple.per_layer_projection_norm.weight"); err != nil {
		return nil, err
	}
	type pair struct {
		dst  *tensor.Tensor
		name string
	}
	for i, b := range m.Blocks {
		p := prefix + "layers." + strconv.Itoa(i) + "."
		pairs := []pair{
			{b.InNorm.Weight, p + "input_layernorm.weight"},
			{b.PostAttnNorm.Weight, p + "post_attention_layernorm.weight"},
			{b.PreFFNorm.Weight, p + "pre_feedforward_layernorm.weight"},
			{b.PostFFNorm.Weight, p + "post_feedforward_layernorm.weight"},
			{b.QNorm.Weight, p + "self_attn.q_norm.weight"},
			{b.KNorm.Weight, p + "self_attn.k_norm.weight"},
			{b.QProj.W, p + "self_attn.q_proj.weight"},
			{b.KProj.W, p + "self_attn.k_proj.weight"},
			{b.VProj.W, p + "self_attn.v_proj.weight"},
			{b.OProj.W, p + "self_attn.o_proj.weight"},
			{b.GateProj.W, p + "mlp.gate_proj.weight"},
			{b.UpProj.W, p + "mlp.up_proj.weight"},
			{b.DownProj.W, p + "mlp.down_proj.weight"},
			{b.PLEGate.W, p + "ple_block.per_layer_input_gate.weight"},
			{b.PLEProj.W, p + "ple_block.per_layer_projection.weight"},
			{b.PostPLENorm.Weight, p + "ple_block.post_per_layer_input_norm.weight"},
		}
		for _, pr := range pairs {
			if err := idx.copyInto(pr.dst, pr.name); err != nil {
				return nil, err
			}
		}
		if idx.has(p + "layer_scalar") {
			if err := idx.copyInto(b.LayerScalar, p+"layer_scalar"); err != nil {
				return nil, err
			}
		} else {
			one, err := tensor.New([]float32{1}, []int{1}, dev)
			if err != nil {
				return nil, err
			}
			if one.DType() != b.LayerScalar.DType() {
				if one, err = ops.Cast(one, b.LayerScalar.DType()); err != nil {
					return nil, err
				}
			}
			if err := b.LayerScalar.CopyFrom(one); err != nil {
				return nil, err
			}
		}
	}
	if err := idx.copyInto(m.FinalNorm.Weight, prefix+"norm.weight"); err != nil {
		return nil, err
	}
	if err := idx.copyInto(m.OutProj.W, prefix+"embedding_projection.weight"); err != nil {
		return nil, err
	}
	return m, nil
}
