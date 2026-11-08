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

// nemotronHFConfig is a nemotron config.json (Nemotron-4, Minitron,
// Nemotron-Mini).
type nemotronHFConfig struct {
	ModelType           string        `json:"model_type"`
	VocabSize           int           `json:"vocab_size"`
	HiddenSize          int           `json:"hidden_size"`
	IntermediateSize    int           `json:"intermediate_size"`
	NumHiddenLayers     int           `json:"num_hidden_layers"`
	NumAttentionHeads   int           `json:"num_attention_heads"`
	NumKeyValueHeads    int           `json:"num_key_value_heads"`
	HeadDim             *int          `json:"head_dim"`
	KVChannels          *int          `json:"kv_channels"` // older configs (Nemotron-Mini): the head dim
	HiddenAct           string        `json:"hidden_act"`
	NormEps             float64       `json:"norm_eps"`
	PartialRotaryFactor *float64      `json:"partial_rotary_factor"`
	RopeTheta           float64       `json:"rope_theta"`
	RopeParameters      *hfRopeParams `json:"rope_parameters"`
	RopeScaling         *hfRopeParams `json:"rope_scaling"`
	AttentionBias       bool          `json:"attention_bias"`
	MLPBias             bool          `json:"mlp_bias"`
	MaxPosition         int           `json:"max_position_embeddings"`
	TieWordEmbeddings   bool          `json:"tie_word_embeddings"`
}

// fetchNemotronConfig maps a nemotron config onto LlamaConfig (defaults as
// transformers' NemotronConfig: partial_rotary_factor 0.5, norm_eps 1e-5,
// head_dim from head_dim, else kv_channels, else hidden / heads).
func fetchNemotronConfig(repo string) (LlamaConfig, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return LlamaConfig{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return LlamaConfig{}, err
	}
	var hf nemotronHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return LlamaConfig{}, fmt.Errorf("nemotron config: %w", err)
	}
	if a := hf.HiddenAct; a != "" && a != "relu2" {
		return LlamaConfig{}, fmt.Errorf("nemotron: hidden_act %q not supported (relu2)", a)
	}
	if hf.AttentionBias || hf.MLPBias {
		return LlamaConfig{}, fmt.Errorf("nemotron: attention_bias / mlp_bias not supported")
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
			return LlamaConfig{}, fmt.Errorf("nemotron: rope_type %q not supported", t)
		}
		var pr struct {
			PartialRotaryFactor *float64 `json:"partial_rotary_factor"`
		}
		var top struct {
			RopeParameters json.RawMessage `json:"rope_parameters"`
		}
		if json.Unmarshal(raw, &top) == nil && len(top.RopeParameters) > 0 &&
			json.Unmarshal(top.RopeParameters, &pr) == nil && pr.PartialRotaryFactor != nil {
			partial = *pr.PartialRotaryFactor
		}
	}
	if base == 0 {
		base = 10000
	}
	headDim := hf.HiddenSize / hf.NumAttentionHeads
	switch {
	case hf.HeadDim != nil && *hf.HeadDim > 0:
		headDim = *hf.HeadDim
	case hf.KVChannels != nil && *hf.KVChannels > 0:
		headDim = *hf.KVChannels
	}
	eps := hf.NormEps
	if eps == 0 {
		eps = 1e-5
	}
	return LlamaConfig{
		VocabSize: hf.VocabSize, HiddenSize: hf.HiddenSize, IntermediateSize: hf.IntermediateSize,
		NumLayers: hf.NumHiddenLayers, NumHeads: hf.NumAttentionHeads, NumKVHeads: hf.NumKeyValueHeads,
		HeadDim: headDim, MaxPosition: hf.MaxPosition, RopeBase: base, RmsEps: eps,
		TieEmbed: hf.TieWordEmbeddings, PartialRotary: partial,
		LayerNorm1P: true, ReLU2MLP: true,
	}, nil
}

// LoadNemotronFromHF loads a nemotron checkpoint (e.g.
// nvidia/Nemotron-Mini-4B-Instruct, nvidia/Minitron-4B-Base) as a *Llama:
// every norm is a LayerNorm with bias whose effective weight is 1 + w
// ("LayerNorm1P", computed in F32 as transformers does), the MLP is the
// non-gated down(relu(up(x))²), and RoPE (rotate-half) covers the first
// partial_rotary_factor of each head.
func LoadNemotronFromHF(repo string, dev device.Device, weightDT dtype.DType) (*Llama, error) {
	cfg, err := fetchNemotronConfig(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	cfg.SkipInit = true
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadNemotronFromHF: %w", err)
	}
	m, err := NewLlama(cfg, dev)
	if err != nil {
		return nil, err
	}
	// ln installs a LayerNorm1P: gamma = 1 + weight and beta = bias, at the
	// activation dtype (the CUDA layernorm kernel wants gamma / beta at the
	// input's dtype). The sum is formed in F32 and rounded once, which is
	// exactly transformers' `self.weight + 1.0` on a BF16 parameter.
	act := m.Cfg.WeightDType
	if act != dtype.BF16 {
		act = dtype.F32
	}
	toAct := func(t *tensor.Tensor) (*tensor.Tensor, error) {
		if act == dtype.F32 {
			return t.To(dev)
		}
		c, err := ops.Cast(t, act)
		if err != nil {
			return nil, err
		}
		return c.To(dev)
	}
	ln := func(gamma, beta **tensor.Tensor, prefix string) error {
		w, ok := hf[prefix+".weight"]
		if !ok {
			return fmt.Errorf("LoadNemotronFromHF: missing tensor %q", prefix+".weight")
		}
		w32, err := ops.Cast(w, dtype.F32)
		if err != nil {
			return err
		}
		one, err := tensor.New([]float32{1}, []int{1}, w32.Device())
		if err != nil {
			return err
		}
		if w32, err = ops.Add(w32, one); err != nil {
			return err
		}
		if *gamma, err = toAct(w32); err != nil {
			return err
		}
		b, ok := hf[prefix+".bias"]
		if !ok {
			return fmt.Errorf("LoadNemotronFromHF: missing tensor %q", prefix+".bias")
		}
		b32, err := ops.Cast(b, dtype.F32)
		if err != nil {
			return err
		}
		*beta, err = toAct(b32)
		return err
	}
	if err := copyExactCast(m.TokenEmb.Weight, hf, "model.embed_tokens.weight"); err != nil {
		return nil, err
	}
	if err := ln(&m.FinalLN.Gamma, &m.FinalLN.Beta, "model.norm"); err != nil {
		return nil, err
	}
	if !cfg.TieEmbed {
		if err := copyExactCast(m.LMHead.W, hf, "lm_head.weight"); err != nil {
			return nil, err
		}
	}
	for i, b := range m.Blocks {
		p := "model.layers." + strconv.Itoa(i) + "."
		if err := ln(&b.LN1.Gamma, &b.LN1.Beta, p+"input_layernorm"); err != nil {
			return nil, err
		}
		if err := ln(&b.LN2.Gamma, &b.LN2.Beta, p+"post_attention_layernorm"); err != nil {
			return nil, err
		}
		for _, pr := range []struct {
			dst  *tensor.Tensor
			name string
		}{
			{b.QProj.W, p + "self_attn.q_proj.weight"}, {b.KProj.W, p + "self_attn.k_proj.weight"},
			{b.VProj.W, p + "self_attn.v_proj.weight"}, {b.OProj.W, p + "self_attn.o_proj.weight"},
			{b.UpProj.W, p + "mlp.up_proj.weight"}, {b.DownProj.W, p + "mlp.down_proj.weight"},
		} {
			if err := copyExactCast(pr.dst, hf, pr.name); err != nil {
				return nil, err
			}
		}
	}
	return m, nil
}
