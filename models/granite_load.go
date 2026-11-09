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

// graniteHFConfig is a Granite config.json: `granite` (Granite 3.x dense)
// or `granitemoehybrid` (Granite 4.0). Granite 4's attention-only
// checkpoints (granite-4.0-micro: every layer_types entry "attention",
// num_local_experts 0) are a Llama decoder with Granite's multipliers and a
// fused shared MLP; Mamba2 layers (the -h- models) and routed experts are
// not supported yet.
type graniteHFConfig struct {
	ModelType              string        `json:"model_type"`
	VocabSize              int           `json:"vocab_size"`
	HiddenSize             int           `json:"hidden_size"`
	IntermediateSize       int           `json:"intermediate_size"`
	SharedIntermediateSize int           `json:"shared_intermediate_size"`
	NumHiddenLayers        int           `json:"num_hidden_layers"`
	NumAttentionHeads      int           `json:"num_attention_heads"`
	NumKeyValueHeads       int           `json:"num_key_value_heads"`
	RmsNormEps             float64       `json:"rms_norm_eps"`
	RopeTheta              float64       `json:"rope_theta"`
	RopeParameters         *hfRopeParams `json:"rope_parameters"`
	RopeScaling            *hfRopeParams `json:"rope_scaling"`
	MaxPosition            int           `json:"max_position_embeddings"`
	TieWordEmbeddings      bool          `json:"tie_word_embeddings"`
	AttentionBias          bool          `json:"attention_bias"`
	HiddenAct              string        `json:"hidden_act"`
	EmbeddingMultiplier    float64       `json:"embedding_multiplier"`
	AttentionMultiplier    float64       `json:"attention_multiplier"`
	ResidualMultiplier     float64       `json:"residual_multiplier"`
	LogitsScaling          float64       `json:"logits_scaling"`
	LayerTypes             []string      `json:"layer_types"`
	NumLocalExperts        int           `json:"num_local_experts"`
	PositionEmbeddingType  string        `json:"position_embedding_type"`
}

func fetchGraniteConfig(repo string) (LlamaConfig, *graniteHFConfig, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return LlamaConfig{}, nil, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return LlamaConfig{}, nil, err
	}
	var hf graniteHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return LlamaConfig{}, nil, fmt.Errorf("granite config: %w", err)
	}
	for i, lt := range hf.LayerTypes {
		if lt != "attention" && lt != "full_attention" {
			return LlamaConfig{}, nil, fmt.Errorf("granite: layer %d is %q; Mamba2 layers (granite-4.0-h-*) are not supported yet", i, lt)
		}
	}
	if hf.NumLocalExperts > 0 {
		return LlamaConfig{}, nil, fmt.Errorf("granite: %d routed experts not supported yet (dense / shared-MLP checkpoints only)", hf.NumLocalExperts)
	}
	if a := hf.HiddenAct; a != "" && a != "silu" {
		return LlamaConfig{}, nil, fmt.Errorf("granite: hidden_act %q not supported (silu)", a)
	}
	if p := hf.PositionEmbeddingType; p != "" && p != "rope" {
		return LlamaConfig{}, nil, fmt.Errorf("granite: position_embedding_type %q not supported (rope)", p)
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
			return LlamaConfig{}, nil, fmt.Errorf("granite: rope_type %q not supported", t)
		}
	}
	if base == 0 {
		base = 10000
	}
	inter := hf.IntermediateSize
	if hf.ModelType == "granitemoehybrid" && hf.SharedIntermediateSize > 0 {
		inter = hf.SharedIntermediateSize
	}
	eps := hf.RmsNormEps
	if eps == 0 {
		eps = 1e-6
	}
	cfg := LlamaConfig{
		VocabSize: hf.VocabSize, HiddenSize: hf.HiddenSize, IntermediateSize: inter,
		NumLayers: hf.NumHiddenLayers, NumHeads: hf.NumAttentionHeads, NumKVHeads: hf.NumKeyValueHeads,
		HeadDim: hf.HiddenSize / hf.NumAttentionHeads, MaxPosition: hf.MaxPosition,
		RopeBase: base, RmsEps: eps, TieEmbed: hf.TieWordEmbeddings, QKVBias: hf.AttentionBias,
		EmbedMultiplier: hf.EmbeddingMultiplier, AttnScale: hf.AttentionMultiplier,
		ResidualMultiplier: hf.ResidualMultiplier, LogitsScaling: hf.LogitsScaling,
	}
	return cfg, &hf, nil
}

// LoadGraniteFromHF loads a Granite 3.x (`granite`) or attention-only
// Granite 4.0 (`granitemoehybrid`, e.g. ibm-granite/granite-4.0-micro)
// checkpoint as a *Llama: embeddings × embedding_multiplier, softmax scale
// attention_multiplier, residual branches × residual_multiplier, logits ÷
// logits_scaling. Granite 4's shared MLP (input_linear = [gate; up],
// output_linear) is renamed onto the Llama SwiGLU projections.
func LoadGraniteFromHF(repo string, dev device.Device, weightDT dtype.DType) (*Llama, error) {
	cfg, hfc, err := fetchGraniteConfig(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	cfg.SkipInit = true
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadGraniteFromHF: %w", err)
	}
	if hfc.ModelType == "granitemoehybrid" {
		if err := renameGraniteSharedMLP(hf, cfg.NumLayers, cfg.IntermediateSize); err != nil {
			return nil, err
		}
	}
	m, err := NewLlama(cfg, dev)
	if err != nil {
		return nil, err
	}
	if err := mapHFToLlama(m, hf); err != nil {
		return nil, fmt.Errorf("LoadGraniteFromHF: %w", err)
	}
	return m, nil
}

// renameGraniteSharedMLP splits each layer's shared_mlp.input_linear
// (2I, H) into mlp.gate_proj (rows 0..I) and mlp.up_proj (rows I..2I) —
// the order of transformers' `chunk(2)`: activation on the first half —
// and renames shared_mlp.output_linear to mlp.down_proj.
func renameGraniteSharedMLP(hf map[string]*tensor.Tensor, layers, inter int) error {
	for i := range layers {
		p := "model.layers." + strconv.Itoa(i) + "."
		in, ok := hf[p+"shared_mlp.input_linear.weight"]
		if !ok {
			return fmt.Errorf("granite: missing %sshared_mlp.input_linear.weight", p)
		}
		if in.Rank() != 2 || in.Dim(0) != 2*inter {
			return fmt.Errorf("granite: %sshared_mlp.input_linear.weight is %v, want (%d, H)", p, in.Shape(), 2*inter)
		}
		gate, err := in.Narrow(0, 0, inter)
		if err != nil {
			return err
		}
		up, err := in.Narrow(0, inter, inter)
		if err != nil {
			return err
		}
		if gate, err = gate.Contiguous(); err != nil {
			return err
		}
		if up, err = up.Contiguous(); err != nil {
			return err
		}
		hf[p+"mlp.gate_proj.weight"], hf[p+"mlp.up_proj.weight"] = gate, up
		delete(hf, p+"shared_mlp.input_linear.weight")
		out, ok := hf[p+"shared_mlp.output_linear.weight"]
		if !ok {
			return fmt.Errorf("granite: missing %sshared_mlp.output_linear.weight", p)
		}
		hf[p+"mlp.down_proj.weight"] = out
		delete(hf, p+"shared_mlp.output_linear.weight")
	}
	return nil
}
