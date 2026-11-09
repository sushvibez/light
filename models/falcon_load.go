package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/nn"
	"light/pth"
	"light/safetensors"
	"light/tensor"
)

type falconHFConfig struct {
	ModelType         string  `json:"model_type"`
	VocabSize         int     `json:"vocab_size"`
	HiddenSize        int     `json:"hidden_size"`
	FFNHiddenSize     int     `json:"ffn_hidden_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKVHeads        int     `json:"num_kv_heads"`
	MaxPosition       int     `json:"max_position_embeddings"`
	LayerNormEps      float64 `json:"layer_norm_epsilon"`
	RopeTheta         float64 `json:"rope_theta"`
	Alibi             bool    `json:"alibi"`
	MultiQuery        bool    `json:"multi_query"`
	NewDecoderArch    bool    `json:"new_decoder_architecture"`
	ParallelAttn      bool    `json:"parallel_attn"`
	Bias              bool    `json:"bias"`
	NumLNParallel     int     `json:"num_ln_in_parallel_attn"`
	TieWordEmbed      *bool   `json:"tie_word_embeddings"`
	RopeScaling       any     `json:"rope_scaling"`
}

// falconFile resolves a repo file: a local checkpoint directory (tiny
// random models written by scripts/verify/ref_falcon.py) or the HF hub.
func falconFile(repo, name string) (string, error) {
	if st, err := os.Stat(repo); err == nil && st.IsDir() {
		p := filepath.Join(repo, name)
		if _, err := os.Stat(p); err != nil {
			return "", err
		}
		return p, nil
	}
	return hub.Download(repo, "main", name)
}

func fetchFalconConfig(repo string) (FalconConfig, error) {
	path, err := falconFile(repo, "config.json")
	if err != nil {
		return FalconConfig{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return FalconConfig{}, err
	}
	var hf falconHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return FalconConfig{}, fmt.Errorf("fetchFalconConfig: %w", err)
	}
	if hf.ModelType != "falcon" && hf.ModelType != "RefinedWebModel" && hf.ModelType != "RefinedWeb" {
		return FalconConfig{}, fmt.Errorf("fetchFalconConfig: model_type %q not supported (falcon)", hf.ModelType)
	}
	if hf.RopeScaling != nil {
		return FalconConfig{}, fmt.Errorf("fetchFalconConfig: rope_scaling not supported")
	}
	cfg := FalconConfig{
		VocabSize: hf.VocabSize, HiddenSize: hf.HiddenSize, FFNHiddenSize: hf.FFNHiddenSize,
		NumLayers: hf.NumHiddenLayers, NumHeads: hf.NumAttentionHeads, NumKVHeads: hf.NumKVHeads,
		MaxPosition: hf.MaxPosition, LayerNormEps: hf.LayerNormEps, RopeBase: hf.RopeTheta,
		Alibi: hf.Alibi, MultiQuery: hf.MultiQuery, NewDecoderArch: hf.NewDecoderArch,
		ParallelAttn: hf.ParallelAttn, Bias: hf.Bias, NumLNParallel: hf.NumLNParallel, TieEmbed: true,
	}
	if cfg.LayerNormEps == 0 {
		cfg.LayerNormEps = 1e-5
	}
	if cfg.MaxPosition == 0 {
		cfg.MaxPosition = 2048
	}
	if hf.TieWordEmbed != nil {
		cfg.TieEmbed = *hf.TieWordEmbed
	}
	return cfg, nil
}

// loadCheckpointTensors reads a checkpoint (an HF repo or a local directory):
// sharded or single safetensors,
// else pytorch_model.bin (falcon-rw-1b ships only the torch zip).
func loadCheckpointTensors(repo string) (map[string]*tensor.Tensor, error) {
	if st, err := os.Stat(repo); err == nil && st.IsDir() {
		if sts, _ := filepath.Glob(filepath.Join(repo, "*.safetensors")); len(sts) > 0 {
			t, _, err := safetensors.LoadFiles(sts, device.CPU())
			return t, err
		}
		return pth.Load(filepath.Join(repo, "pytorch_model.bin"), "", device.CPU())
	}
	if paths, err := hub.DownloadModelSafetensors(repo, "main"); err == nil {
		t, _, err := safetensors.LoadFiles(paths, device.CPU())
		return t, err
	}
	p, err := hub.Download(repo, "main", "pytorch_model.bin")
	if err != nil {
		return nil, fmt.Errorf("falcon: no safetensors and no pytorch_model.bin: %w", err)
	}
	return pth.Load(p, "", device.CPU())
}

// LoadFalconFromHF loads a FalconForCausalLM checkpoint (falcon-rw-1b,
// falcon-7b/40b/180b layouts). weightDT is dtype.F32 or dtype.BF16.
func LoadFalconFromHF(repo string, dev device.Device, weightDT dtype.DType) (*Falcon, error) {
	cfg, err := fetchFalconConfig(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	hf, err := loadCheckpointTensors(repo)
	if err != nil {
		return nil, err
	}
	if _, ok := hf["lm_head.weight"]; ok {
		cfg.TieEmbed = false
	}
	m, err := NewFalcon(cfg, dev)
	if err != nil {
		return nil, err
	}
	prefix := "transformer."
	if _, ok := hf[prefix+"word_embeddings.weight"]; !ok {
		prefix = ""
	}
	cp := func(dst *tensor.Tensor, name string) error { return copyExactCast(dst, hf, name) }
	lin := func(l *nn.Linear, name string) error {
		if err := cp(l.W, name+".weight"); err != nil {
			return err
		}
		if l.B != nil {
			return cp(l.B, name+".bias")
		}
		return nil
	}
	ln := func(l *nn.LayerNorm, name string) error {
		if err := cp(l.Gamma, name+".weight"); err != nil {
			return err
		}
		return cp(l.Beta, name+".bias")
	}
	if err := cp(m.TokenEmb.Weight, prefix+"word_embeddings.weight"); err != nil {
		return nil, err
	}
	for i, b := range m.Blocks {
		p := prefix + "h." + strconv.Itoa(i) + "."
		attnLN, mlpLN := "input_layernorm", "post_attention_layernorm"
		if m.Cfg.twoLN() { // m.Cfg carries the defaulted NumLNParallel
			attnLN, mlpLN = "ln_attn", "ln_mlp"
		}
		if err := ln(b.LNAttn, p+attnLN); err != nil {
			return nil, err
		}
		if b.LNMLP != nil {
			if err := ln(b.LNMLP, p+mlpLN); err != nil {
				return nil, err
			}
		}
		for _, pr := range []struct {
			l    *nn.Linear
			name string
		}{{b.QKV, "self_attention.query_key_value"}, {b.Dense, "self_attention.dense"}, {b.Up, "mlp.dense_h_to_4h"}, {b.Down, "mlp.dense_4h_to_h"}} {
			if err := lin(pr.l, p+pr.name); err != nil {
				return nil, err
			}
		}
	}
	if err := ln(m.LNF, prefix+"ln_f"); err != nil {
		return nil, err
	}
	if !cfg.TieEmbed {
		if err := cp(m.LMHead.W, "lm_head.weight"); err != nil {
			return nil, err
		}
	}
	return m, nil
}
