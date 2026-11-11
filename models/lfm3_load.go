package models

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/nn"
	"light/ops"
	"light/tensor"
)

type lfm3HFConfig struct {
	VocabSize            int                                   `json:"vocab_size"`
	HiddenSize           int                                   `json:"hidden_size"`
	IntermediateSize     int                                   `json:"intermediate_size"`
	NumHiddenLayers      int                                   `json:"num_hidden_layers"`
	LayerTypes           []string                              `json:"layer_types"`
	NumAttentionHeads    int                                   `json:"num_attention_heads"`
	NumKeyValueHeads     int                                   `json:"num_key_value_heads"`
	HeadDim              int                                   `json:"head_dim"`
	SlidingWindow        int                                   `json:"sliding_window"`
	ConvLCache           int                                   `json:"conv_L_cache"`
	ConvBias             bool                                  `json:"conv_bias"`
	NormEps              float64                               `json:"norm_eps"`
	MaxPositionEmbedding int                                   `json:"max_position_embeddings"`
	TieWordEmbeddings    *bool                                 `json:"tie_word_embeddings"`
	UseSandwichNorm      bool                                  `json:"use_sandwich_norm"`
	RopeParameters       map[string]map[string]json.RawMessage `json:"rope_parameters"`
}

func fetchLFM3Config(repo string) (LFM3Config, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return LFM3Config{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return LFM3Config{}, err
	}
	var hf lfm3HFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return LFM3Config{}, fmt.Errorf("lfm3 config: %w", err)
	}
	if hf.ConvBias {
		return LFM3Config{}, fmt.Errorf("lfm3: conv_bias checkpoints are not supported")
	}
	rope := func(kind string) (base, partial float64, err error) {
		p, ok := hf.RopeParameters[kind]
		if !ok {
			return 0, 0, nil
		}
		var typ string
		if r, ok := p["rope_type"]; ok {
			_ = json.Unmarshal(r, &typ)
		}
		if typ != "" && typ != "default" {
			return 0, 0, fmt.Errorf("lfm3: %s rope_type %q not supported", kind, typ)
		}
		if r, ok := p["rope_theta"]; ok {
			if err := json.Unmarshal(r, &base); err != nil {
				return 0, 0, err
			}
		}
		partial = 1
		if r, ok := p["partial_rotary_factor"]; ok {
			if err := json.Unmarshal(r, &partial); err != nil {
				return 0, 0, err
			}
		}
		return base, partial, nil
	}
	cfg := LFM3Config{
		VocabSize: hf.VocabSize, HiddenSize: hf.HiddenSize, IntermediateSize: hf.IntermediateSize,
		NumLayers: hf.NumHiddenLayers, LayerTypes: hf.LayerTypes,
		NumHeads: hf.NumAttentionHeads, NumKVHeads: hf.NumKeyValueHeads, HeadDim: hf.HeadDim,
		SlidingWindow: hf.SlidingWindow, ConvKernel: hf.ConvLCache, SandwichNorm: hf.UseSandwichNorm,
		NormEps: hf.NormEps, MaxPosition: hf.MaxPositionEmbedding, TieEmbed: true,
	}
	if hf.TieWordEmbeddings != nil {
		cfg.TieEmbed = *hf.TieWordEmbeddings
	}
	if cfg.HeadDim == 0 && cfg.NumHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	if cfg.FullRopeBase, cfg.FullPartialRotary, err = rope("full_attention"); err != nil {
		return LFM3Config{}, err
	}
	if cfg.SlidingRopeBase, cfg.SlidingPartialRotary, err = rope("sliding_attention"); err != nil {
		return LFM3Config{}, err
	}
	return cfg, nil
}

// LoadLFM3FromHF loads an Lfm3ForCausalLM checkpoint (LiquidAI/LFM3-*).
func LoadLFM3FromHF(repo string, dev device.Device, weightDT dtype.DType) (*LFM3, error) {
	cfg, err := fetchLFM3Config(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	cfg.lazyWeights = true
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadLFM3FromHF: %w", err)
	}
	m, err := NewLFM3(cfg, dev)
	if err != nil {
		return nil, err
	}
	act := m.Cfg.activationDType()
	toDev := func(name string, dt dtype.DType) (*tensor.Tensor, error) {
		src, ok := hf[name]
		if !ok {
			return nil, fmt.Errorf("LoadLFM3FromHF: missing tensor %q", name)
		}
		delete(hf, name)
		if src.DType() != dt {
			if src, err = ops.Cast(src, dt); err != nil {
				return nil, fmt.Errorf("%s: %w", name, err)
			}
		}
		return src.To(dev)
	}
	emb, err := toDev("model.embed_tokens.weight", nn.WeightStorageDType(act, cfg.HiddenSize))
	if err != nil {
		return nil, err
	}
	if m.TokenEmb, err = nn.NewEmbeddingFromWeight(emb); err != nil {
		return nil, err
	}
	m.TokenEmb.WidenF32 = emb.DType() != act
	setLinear := func(l *nn.Linear, name string) error {
		w, err := toDev(name, nn.WeightStorageDType(act, l.InFeatures()))
		if err != nil {
			return err
		}
		if err := l.SetWeight(w); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		return nil
	}
	setNorm := func(n *nn.RMSNorm, name string) error {
		w, err := toDev(name, act)
		if err != nil {
			return err
		}
		if !w.Shape().Equal(n.Weight.Shape()) {
			return fmt.Errorf("%s: shape %v, model %v", name, w.Shape(), n.Weight.Shape())
		}
		n.Weight = w
		return nil
	}
	for i, b := range m.Blocks {
		p := "model.layers." + strconv.Itoa(i) + "."
		norms := []struct {
			n    *nn.RMSNorm
			name string
		}{{b.OpNorm, "operator_norm"}, {b.FFNNorm, "ffn_norm"}, {b.PostOpNorm, "post_operator_norm"}, {b.PostFFNNorm, "post_ffn_norm"}}
		for _, nn := range norms {
			if nn.n == nil {
				continue
			}
			if err := setNorm(nn.n, p+nn.name+".weight"); err != nil {
				return nil, err
			}
		}
		lins := []struct {
			l    *nn.Linear
			name string
		}{{b.W1, "feed_forward.w1"}, {b.W3, "feed_forward.w3"}, {b.W2, "feed_forward.w2"}}
		if b.kind == "conv" {
			lins = append(lins, struct {
				l    *nn.Linear
				name string
			}{b.InProj, "conv.in_proj"}, struct {
				l    *nn.Linear
				name string
			}{b.OutProj, "conv.out_proj"})
			w, err := toDev(p+"conv.conv.weight", dtype.F32) // (D, 1, K)
			if err != nil {
				return nil, err
			}
			if w, err = w.Reshape(cfg.HiddenSize, cfg.ConvKernel); err != nil {
				return nil, fmt.Errorf("%sconv.conv.weight: %w", p, err)
			}
			b.ConvW = w
		} else {
			for _, x := range []struct {
				l    *nn.Linear
				name string
			}{{b.QProj, "self_attn.q_proj"}, {b.KProj, "self_attn.k_proj"}, {b.VProj, "self_attn.v_proj"},
				{b.OProj, "self_attn.o_proj"}, {b.GateProj, "self_attn.gate_proj"}} {
				lins = append(lins, x)
			}
			if err := setNorm(b.QNorm, p+"self_attn.q_norm.weight"); err != nil {
				return nil, err
			}
			if err := setNorm(b.KNorm, p+"self_attn.k_norm.weight"); err != nil {
				return nil, err
			}
		}
		for _, x := range lins {
			if err := setLinear(x.l, p+x.name+".weight"); err != nil {
				return nil, err
			}
		}
	}
	if err := setNorm(m.FinalNorm, "model.embedding_norm.weight"); err != nil {
		return nil, err
	}
	if cfg.TieEmbed {
		if err := m.LMHead.SetWeight(m.TokenEmb.Weight); err != nil {
			return nil, err
		}
	} else if err := setLinear(m.LMHead, "lm_head.weight"); err != nil {
		return nil, err
	}
	return m, nil
}
