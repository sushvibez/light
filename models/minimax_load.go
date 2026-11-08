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

// minimaxHFConfig is a minimax config.json (transformers' MiniMaxConfig).
type minimaxHFConfig struct {
	VocabSize         int           `json:"vocab_size"`
	HiddenSize        int           `json:"hidden_size"`
	IntermediateSize  int           `json:"intermediate_size"`
	NumHiddenLayers   int           `json:"num_hidden_layers"`
	NumAttentionHeads int           `json:"num_attention_heads"`
	NumKeyValueHeads  int           `json:"num_key_value_heads"`
	HeadDim           *int          `json:"head_dim"`
	HiddenAct         string        `json:"hidden_act"`
	LayerTypes        []string      `json:"layer_types"`
	NumLocalExperts   int           `json:"num_local_experts"`
	NumExpertsPerTok  int           `json:"num_experts_per_tok"`
	LinearAlpha       *float64      `json:"linear_attn_alpha_factor"`
	LinearBeta        *float64      `json:"linear_attn_beta_factor"`
	FullAlpha         *float64      `json:"full_attn_alpha_factor"`
	FullBeta          *float64      `json:"full_attn_beta_factor"`
	MLPAlpha          *float64      `json:"mlp_alpha_factor"`
	MLPBeta           *float64      `json:"mlp_beta_factor"`
	RopeTheta         float64       `json:"rope_theta"`
	RopeParameters    *hfRopeParams `json:"rope_parameters"`
	SlidingWindow     *int          `json:"sliding_window"`
	RmsNormEps        float64       `json:"rms_norm_eps"`
	MaxPosition       int           `json:"max_position_embeddings"`
	TieWordEmbeddings bool          `json:"tie_word_embeddings"`
}

func fetchMiniMaxConfig(repo string) (MiniMaxConfig, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return MiniMaxConfig{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return MiniMaxConfig{}, err
	}
	var hf minimaxHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return MiniMaxConfig{}, fmt.Errorf("minimax config: %w", err)
	}
	if a := hf.HiddenAct; a != "" && a != "silu" {
		return MiniMaxConfig{}, fmt.Errorf("minimax: hidden_act %q not supported", a)
	}
	if hf.SlidingWindow != nil && *hf.SlidingWindow > 0 {
		return MiniMaxConfig{}, fmt.Errorf("minimax: sliding_window not supported")
	}
	n := hf.NumHiddenLayers
	if len(hf.LayerTypes) != n {
		return MiniMaxConfig{}, fmt.Errorf("minimax: %d layer_types for %d layers", len(hf.LayerTypes), n)
	}
	linear := make([]bool, n)
	for i, t := range hf.LayerTypes {
		switch t {
		case "linear_attention":
			linear[i] = true
		case "full_attention":
		default:
			return MiniMaxConfig{}, fmt.Errorf("minimax: layer %d type %q", i, t)
		}
	}
	base := hf.RopeTheta
	if rp := hf.RopeParameters; rp != nil {
		if rp.RopeTheta > 0 {
			base = rp.RopeTheta
		}
		if t := rp.RopeType; t != "" && t != "default" {
			return MiniMaxConfig{}, fmt.Errorf("minimax: rope_type %q not supported", t)
		}
	}
	if base == 0 {
		base = 10000
	}
	headDim := hf.HiddenSize / hf.NumAttentionHeads
	if hf.HeadDim != nil && *hf.HeadDim > 0 {
		headDim = *hf.HeadDim
	}
	f := func(p *float64) float64 {
		if p == nil {
			return 1
		}
		return *p
	}
	eps := hf.RmsNormEps
	if eps == 0 {
		eps = 1e-5
	}
	return MiniMaxConfig{
		VocabSize: hf.VocabSize, HiddenSize: hf.HiddenSize, IntermediateSize: hf.IntermediateSize, NumLayers: n,
		NumHeads: hf.NumAttentionHeads, NumKVHeads: hf.NumKeyValueHeads, HeadDim: headDim, LinearLayers: linear,
		NumExperts: hf.NumLocalExperts, TopK: hf.NumExpertsPerTok,
		LinearAlpha: f(hf.LinearAlpha), LinearBeta: f(hf.LinearBeta), FullAlpha: f(hf.FullAlpha),
		FullBeta: f(hf.FullBeta), MLPAlpha: f(hf.MLPAlpha), MLPBeta: f(hf.MLPBeta),
		RopeBase: base, RmsEps: eps, MaxPosition: hf.MaxPosition, TieEmbed: hf.TieWordEmbeddings,
	}, nil
}

// LoadMiniMaxFromHF loads a minimax checkpoint (transformers layout, e.g.
// MiniMaxAI/MiniMax-Text-01-hf, or a local directory). Experts are the
// Mixtral-style block_sparse_moe.experts.{e}.w1 (gate) / w3 (up) / w2
// (down). The lightning layers' decay buffers (slope_rate, query_decay, …)
// are derived from the config rather than read. weightDT is F32 or BF16.
func LoadMiniMaxFromHF(repo string, dev device.Device, weightDT dtype.DType) (*MiniMax, error) {
	cfg, err := fetchMiniMaxConfig(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadMiniMaxFromHF: %w", err)
	}
	m, err := NewMiniMax(cfg, dev)
	if err != nil {
		return nil, err
	}
	act := m.Cfg.activationDType()
	toDev := func(name string) (*tensor.Tensor, error) {
		src, ok := hf[name]
		if !ok {
			return nil, fmt.Errorf("LoadMiniMaxFromHF: missing tensor %q", name)
		}
		delete(hf, name)
		if src.DType() != act {
			if src, err = ops.Cast(src, act); err != nil {
				return nil, fmt.Errorf("%s: %w", name, err)
			}
		}
		return src.To(dev)
	}
	type pair struct {
		dst  *tensor.Tensor
		name string
	}
	pairs := []pair{{m.TokenEmb.Weight, "model.embed_tokens.weight"}, {m.FinalNorm.Weight, "model.norm.weight"}}
	if !cfg.TieEmbed {
		pairs = append(pairs, pair{m.LMHead.W, "lm_head.weight"})
	}
	for i, b := range m.Blocks {
		p := "model.layers." + strconv.Itoa(i) + "."
		sa := p + "self_attn."
		pairs = append(pairs, pair{b.InNorm.Weight, p + "input_layernorm.weight"},
			pair{b.PostNorm.Weight, p + "post_attention_layernorm.weight"},
			pair{b.MoE.Gate.W, p + "block_sparse_moe.gate.weight"})
		if b.linear {
			pairs = append(pairs, pair{b.QKV.W, sa + "qkv_proj.weight"}, pair{b.OutGate.W, sa + "output_gate.weight"},
				pair{b.Out.W, sa + "out_proj.weight"}, pair{b.LNorm.Weight, sa + "norm.weight"})
		} else {
			pairs = append(pairs, pair{b.QProj.W, sa + "q_proj.weight"}, pair{b.KProj.W, sa + "k_proj.weight"},
				pair{b.VProj.W, sa + "v_proj.weight"}, pair{b.OProj.W, sa + "o_proj.weight"})
		}
	}
	for _, pr := range pairs {
		if err := copyExactCast(pr.dst, hf, pr.name); err != nil {
			return nil, err
		}
	}
	for i, b := range m.Blocks {
		ep := "model.layers." + strconv.Itoa(i) + ".block_sparse_moe.experts."
		E := cfg.NumExperts
		gate, up, down := make([]*tensor.Tensor, E), make([]*tensor.Tensor, E), make([]*tensor.Tensor, E)
		for e := range E {
			x := ep + strconv.Itoa(e) + "."
			if gate[e], err = toDev(x + "w1.weight"); err != nil {
				return nil, err
			}
			if up[e], err = toDev(x + "w3.weight"); err != nil {
				return nil, err
			}
			if down[e], err = toDev(x + "w2.weight"); err != nil {
				return nil, err
			}
		}
		if err := b.MoE.StackExperts(gate, up, down); err != nil {
			return nil, fmt.Errorf("layer %d experts: %w", i, err)
		}
	}
	return m, nil
}
