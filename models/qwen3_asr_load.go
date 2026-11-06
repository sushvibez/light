package models

import (
	"encoding/json"
	"fmt"
	"os"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/nn"
	"light/tensor"
)

type qwen3ASRHFConfig struct {
	ModelType   string `json:"model_type"`
	AudioConfig struct {
		NumMelBins       int `json:"num_mel_bins"`
		DModel           int `json:"d_model"`
		EncoderLayers    int `json:"encoder_layers"`
		EncoderHeads     int `json:"encoder_attention_heads"`
		EncoderFFN       int `json:"encoder_ffn_dim"`
		DownsampleHidden int `json:"downsample_hidden_size"`
		NWindow          int `json:"n_window"`
		NWindowInfer     int `json:"n_window_infer"`
		OutputDim        int `json:"output_dim"`
		MaxPositions     int `json:"max_position_embeddings"`
	} `json:"audio_config"`
	AudioTokenID int64           `json:"audio_token_id"`
	EOS          json.RawMessage `json:"eos_token_id"`
}

// fetchQwen3ASRConfig reads config.json and the special token ids from
// tokenizer.json (audio start/end, <asr_text>).
func fetchQwen3ASRConfig(repo string) (Qwen3ASRConfig, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return Qwen3ASRConfig{}, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return Qwen3ASRConfig{}, err
	}
	var hf qwen3ASRHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return Qwen3ASRConfig{}, fmt.Errorf("qwen3_asr config: %w", err)
	}
	if hf.ModelType != "qwen3_asr" {
		return Qwen3ASRConfig{}, fmt.Errorf("qwen3_asr: model_type %q", hf.ModelType)
	}
	a := hf.AudioConfig
	cfg := Qwen3ASRConfig{
		NumMelBins: a.NumMelBins, DModel: a.DModel, Layers: a.EncoderLayers, Heads: a.EncoderHeads,
		FFN: a.EncoderFFN, DownsampleHidden: a.DownsampleHidden, NWindow: a.NWindow, NWindowInfer: a.NWindowInfer,
		OutputDim: a.OutputDim, ChunkPositions: a.MaxPositions, AudioTokenID: hf.AudioTokenID,
	}
	if len(hf.EOS) > 0 {
		var one int64
		if json.Unmarshal(hf.EOS, &one) == nil {
			cfg.EOS = []int64{one}
		} else if err := json.Unmarshal(hf.EOS, &cfg.EOS); err != nil {
			return cfg, fmt.Errorf("qwen3_asr eos_token_id: %w", err)
		}
	}
	specials, err := addedTokenIDs(repo, "<|audio_start|>", "<|audio_end|>", "<asr_text>")
	if err != nil {
		return cfg, err
	}
	cfg.AudioStartID, cfg.AudioEndID, cfg.ASRTextID = specials[0], specials[1], specials[2]
	return cfg, nil
}

// addedTokenIDs looks up added-token ids by content in tokenizer.json.
func addedTokenIDs(repo string, contents ...string) ([]int64, error) {
	path, err := hub.Download(repo, "main", "tokenizer.json")
	if err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var tj struct {
		Added []struct {
			ID      int64  `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, fmt.Errorf("tokenizer.json: %w", err)
	}
	out := make([]int64, len(contents))
	for i, c := range contents {
		found := false
		for _, a := range tj.Added {
			if a.Content == c {
				out[i], found = a.ID, true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("tokenizer.json: added token %q not found", c)
		}
	}
	return out, nil
}

// LoadQwen3ASRFromHF loads Qwen/Qwen3-ASR-1.7B-hf (or -0.6B-hf). The
// decoder is built at weightDT (BF16 recommended: 3.4 GB for 1.7B); the
// audio encoder and projector stay F32 (~1.3 GB).
func LoadQwen3ASRFromHF(repo string, dev device.Device, weightDT dtype.DType) (*Qwen3ASR, error) {
	cfg, err := fetchQwen3ASRConfig(repo)
	if err != nil {
		return nil, err
	}
	lmCfg, err := fetchLlamaConfig(repo)
	if err != nil {
		return nil, err
	}
	lmCfg.WeightDType = weightDT
	lmCfg.SkipInit = true
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadQwen3ASRFromHF: %w", err)
	}
	lm, err := NewLlama(lmCfg, dev)
	if err != nil {
		return nil, err
	}
	if err := mapHFToLlamaPrefix(lm, hf, "model.language_model."); err != nil {
		return nil, err
	}
	m, err := NewQwen3ASR(cfg, lm, dev)
	if err != nil {
		return nil, err
	}
	const ap = "model.audio_tower."
	cp := func(dst *tensor.Tensor, name string) error { return copyExactCast(dst, hf, name) }
	for _, c := range []struct {
		w, b *tensor.Tensor
		name string
	}{{m.Conv1W, m.Conv1B, "conv2d1"}, {m.Conv2W, m.Conv2B, "conv2d2"}, {m.Conv3W, m.Conv3B, "conv2d3"}} {
		if err := cp(c.w, ap+c.name+".weight"); err != nil {
			return nil, err
		}
		if err := cp(c.b, ap+c.name+".bias"); err != nil {
			return nil, err
		}
	}
	if err := cp(m.ConvOut.W, ap+"conv_out.weight"); err != nil {
		return nil, err
	}
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
	for i, l := range m.Layers {
		p := fmt.Sprintf("%slayers.%d.", ap, i)
		if err := ln(l.LN1, p+"self_attn_layer_norm"); err != nil {
			return nil, err
		}
		if err := ln(l.LN2, p+"final_layer_norm"); err != nil {
			return nil, err
		}
		for _, a := range []struct {
			l    *nn.Linear
			name string
		}{{l.Attn.Q, "q_proj"}, {l.Attn.K, "k_proj"}, {l.Attn.V, "v_proj"}, {l.Attn.O, "out_proj"}} {
			if err := lin(a.l, p+"self_attn."+a.name); err != nil {
				return nil, err
			}
		}
		if err := lin(l.FC1, p+"fc1"); err != nil {
			return nil, err
		}
		if err := lin(l.FC2, p+"fc2"); err != nil {
			return nil, err
		}
	}
	if err := ln(m.LNPost, ap+"ln_post"); err != nil {
		return nil, err
	}
	if err := lin(m.Proj1, "model.multi_modal_projector.linear_1"); err != nil {
		return nil, err
	}
	if err := lin(m.Proj2, "model.multi_modal_projector.linear_2"); err != nil {
		return nil, err
	}
	return m, nil
}
