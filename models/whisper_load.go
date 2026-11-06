package models

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"light/device"
	"light/hub"
	"light/tensor"
)

type whisperHFConfig struct {
	VocabSize           int    `json:"vocab_size"`
	NumMelBins          int    `json:"num_mel_bins"`
	DModel              int    `json:"d_model"`
	EncoderLayers       int    `json:"encoder_layers"`
	EncoderHeads        int    `json:"encoder_attention_heads"`
	EncoderFFN          int    `json:"encoder_ffn_dim"`
	DecoderLayers       int    `json:"decoder_layers"`
	DecoderHeads        int    `json:"decoder_attention_heads"`
	DecoderFFN          int    `json:"decoder_ffn_dim"`
	MaxSourcePositions  int    `json:"max_source_positions"`
	MaxTargetPositions  int    `json:"max_target_positions"`
	EOSTokenID          int    `json:"eos_token_id"`
	DecoderStartTokenID int    `json:"decoder_start_token_id"`
	ActivationFunction  string `json:"activation_function"`
	SuppressTokens      []int  `json:"suppress_tokens"`
	BeginSuppressTokens []int  `json:"begin_suppress_tokens"`
}

type whisperGenConfig struct {
	LangToID            map[string]int `json:"lang_to_id"`
	TaskToID            map[string]int `json:"task_to_id"`
	NoTimestampsTokenID int            `json:"no_timestamps_token_id"`
	IsMultilingual      bool           `json:"is_multilingual"`
	SuppressTokens      []int          `json:"suppress_tokens"`
	BeginSuppressTokens []int          `json:"begin_suppress_tokens"`
}

func fetchWhisperConfig(repo string) (WhisperConfig, error) {
	var cfg WhisperConfig
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return cfg, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return cfg, err
	}
	var hf whisperHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return cfg, fmt.Errorf("fetchWhisperConfig: %w", err)
	}
	cfg = WhisperConfig{
		VocabSize: hf.VocabSize, NumMelBins: hf.NumMelBins, DModel: hf.DModel,
		EncoderLayers: hf.EncoderLayers, EncoderHeads: hf.EncoderHeads, EncoderFFN: hf.EncoderFFN,
		DecoderLayers: hf.DecoderLayers, DecoderHeads: hf.DecoderHeads, DecoderFFN: hf.DecoderFFN,
		MaxSourcePos: hf.MaxSourcePositions, MaxTargetPos: hf.MaxTargetPositions,
		EOSToken: hf.EOSTokenID, SOTToken: hf.DecoderStartTokenID,
		SuppressTokens: hf.SuppressTokens, BeginSuppress: hf.BeginSuppressTokens,
		ActivationIsErf: hf.ActivationFunction == "" || hf.ActivationFunction == "gelu",
		LangToID:        map[string]int{}, TaskToID: map[string]int{},
	}
	if gp, err := hub.Download(repo, "main", "generation_config.json"); err == nil {
		if raw, err := os.ReadFile(gp); err == nil {
			var g whisperGenConfig
			if json.Unmarshal(raw, &g) == nil {
				for k, v := range g.LangToID {
					cfg.LangToID[strings.Trim(k, "<|>")] = v
				}
				cfg.TaskToID = g.TaskToID
				cfg.NoTimestamps = g.NoTimestampsTokenID
				cfg.IsMultilingual = g.IsMultilingual || len(g.LangToID) > 0
				if len(g.SuppressTokens) > 0 {
					cfg.SuppressTokens = g.SuppressTokens
				}
				if len(g.BeginSuppressTokens) > 0 {
					cfg.BeginSuppress = g.BeginSuppressTokens
				}
			}
		}
	}
	if cfg.NoTimestamps == 0 {
		cfg.NoTimestamps = cfg.SOTToken + 105 // multilingual layout; english-only: sot+3
		if !cfg.IsMultilingual {
			cfg.NoTimestamps = cfg.SOTToken + 3
		}
	}
	return cfg, nil
}

// LoadWhisperMelFilters reads the slaney mel filterbank (nMels × 201) from
// the repo's preprocessor_config.json.
func LoadWhisperMelFilters(repo string) ([][]float32, error) {
	path, err := hub.Download(repo, "main", "preprocessor_config.json")
	if err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var pc struct {
		MelFilters [][]float32 `json:"mel_filters"`
	}
	if err := json.Unmarshal(raw, &pc); err != nil {
		return nil, fmt.Errorf("preprocessor_config.json: %w", err)
	}
	if len(pc.MelFilters) == 0 {
		return nil, fmt.Errorf("preprocessor_config.json: no mel_filters")
	}
	return pc.MelFilters, nil
}

// LoadWhisperFromHF loads any openai/whisper-* checkpoint (or a fine-tune
// with the same layout) at F32.
func LoadWhisperFromHF(repo string, dev device.Device) (*Whisper, error) {
	cfg, err := fetchWhisperConfig(repo)
	if err != nil {
		return nil, err
	}
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadWhisperFromHF: %w", err)
	}
	m, err := NewWhisper(cfg, dev)
	if err != nil {
		return nil, err
	}
	cp := func(dst *tensor.Tensor, name string) error { return copyExactCast(dst, hf, name) }
	for _, p := range []struct {
		t    *tensor.Tensor
		name string
	}{
		{m.Conv1W, "model.encoder.conv1.weight"}, {m.Conv1B, "model.encoder.conv1.bias"},
		{m.Conv2W, "model.encoder.conv2.weight"}, {m.Conv2B, "model.encoder.conv2.bias"},
		{m.EncPos, "model.encoder.embed_positions.weight"},
		{m.EncNorm.Gamma, "model.encoder.layer_norm.weight"}, {m.EncNorm.Beta, "model.encoder.layer_norm.bias"},
		{m.TokEmb.Weight, "model.decoder.embed_tokens.weight"},
		{m.DecPos.Weight, "model.decoder.embed_positions.weight"},
		{m.DecNorm.Gamma, "model.decoder.layer_norm.weight"}, {m.DecNorm.Beta, "model.decoder.layer_norm.bias"},
	} {
		if err := cp(p.t, p.name); err != nil {
			return nil, err
		}
	}
	attn := func(a *whisperAttn, p string) error {
		for _, x := range []struct {
			t    *tensor.Tensor
			name string
		}{
			{a.Q.W, p + "q_proj.weight"}, {a.Q.B, p + "q_proj.bias"},
			{a.K.W, p + "k_proj.weight"},
			{a.V.W, p + "v_proj.weight"}, {a.V.B, p + "v_proj.bias"},
			{a.O.W, p + "out_proj.weight"}, {a.O.B, p + "out_proj.bias"},
		} {
			if err := cp(x.t, x.name); err != nil {
				return err
			}
		}
		return nil
	}
	ln := func(l interface{ Parameters() []*tensor.Tensor }, p string) error {
		ps := l.Parameters()
		if err := cp(ps[0], p+".weight"); err != nil {
			return err
		}
		return cp(ps[1], p+".bias")
	}
	for i, l := range m.EncLayers {
		p := "model.encoder.layers." + strconv.Itoa(i) + "."
		if err := ln(l.LN1, p+"self_attn_layer_norm"); err != nil {
			return nil, err
		}
		if err := attn(l.Attn, p+"self_attn."); err != nil {
			return nil, err
		}
		if err := ln(l.LN2, p+"final_layer_norm"); err != nil {
			return nil, err
		}
		if err := ln(l.FC1, p+"fc1"); err != nil {
			return nil, err
		}
		if err := ln(l.FC2, p+"fc2"); err != nil {
			return nil, err
		}
	}
	for i, l := range m.DecLayers {
		p := "model.decoder.layers." + strconv.Itoa(i) + "."
		if err := ln(l.LN1, p+"self_attn_layer_norm"); err != nil {
			return nil, err
		}
		if err := attn(l.Self, p+"self_attn."); err != nil {
			return nil, err
		}
		if err := ln(l.LN2, p+"encoder_attn_layer_norm"); err != nil {
			return nil, err
		}
		if err := attn(l.Cross, p+"encoder_attn."); err != nil {
			return nil, err
		}
		if err := ln(l.LN3, p+"final_layer_norm"); err != nil {
			return nil, err
		}
		if err := ln(l.FC1, p+"fc1"); err != nil {
			return nil, err
		}
		if err := ln(l.FC2, p+"fc2"); err != nil {
			return nil, err
		}
	}
	return m, nil
}
