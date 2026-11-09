package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// LoadCLIPTextFromDir loads a transformers CLIPTextModel from a directory
// with config.json and its safetensors (a diffusers pipeline's
// text_encoder/ folder) at F32. hidden_act quick_gelu (OpenAI CLIP-L) or
// gelu (exact; OpenCLIP, the HF ports).
func LoadCLIPTextFromDir(dir string, dev device.Device) (*CLIPText, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c struct {
		VocabSize     int     `json:"vocab_size"`
		MaxPosition   int     `json:"max_position_embeddings"`
		HiddenSize    int     `json:"hidden_size"`
		Intermediate  int     `json:"intermediate_size"`
		NumLayers     int     `json:"num_hidden_layers"`
		NumHeads      int     `json:"num_attention_heads"`
		ProjectionDim int     `json:"projection_dim"`
		Eps           float64 `json:"layer_norm_eps"`
		Act           string  `json:"hidden_act"`
		EOS           int64   `json:"eos_token_id"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("clip config: %w", err)
	}
	if c.Act != "quick_gelu" && c.Act != "gelu" {
		return nil, fmt.Errorf("clip: hidden_act %q not supported (quick_gelu, gelu)", c.Act)
	}
	cfg := CLIPTextConfig{VocabSize: c.VocabSize, MaxPosition: c.MaxPosition, HiddenSize: c.HiddenSize,
		NumLayers: c.NumLayers, NumHeads: c.NumHeads, ProjectionDim: c.ProjectionDim, IntermediateSize: c.Intermediate,
		LayerNormEps: c.Eps, UseQuickGELU: c.Act == "quick_gelu", WeightDType: dtype.F32}
	te, err := NewCLIPText(cfg, true, dev)
	if err != nil {
		return nil, err
	}
	w, err := loadCheckpointTensors(dir)
	if err != nil {
		return nil, err
	}
	if err := loadCLIPTextWeights(te, w); err != nil {
		return nil, fmt.Errorf("LoadCLIPTextFromDir: %w", err)
	}
	return te, nil
}

// PooledCLIP is CLIPTextModel's pooler_output for the legacy EOS id (2):
// the final-normed hidden state (B, T, D) at each row's largest token id
// (the EOS, the highest id in CLIP's vocabulary; its first occurrence).
func PooledCLIP(hidden *tensor.Tensor, ids [][]int32) (*tensor.Tensor, error) {
	B, T, D := hidden.Dim(0), hidden.Dim(1), hidden.Dim(2)
	if len(ids) != B {
		return nil, fmt.Errorf("PooledCLIP: %d id rows for batch %d", len(ids), B)
	}
	rows := make([]*tensor.Tensor, B)
	for b, row := range ids {
		at := 0
		for i, id := range row {
			if id > row[at] {
				at = i
			}
		}
		r, err := hidden.Narrow(0, b, 1)
		if err != nil {
			return nil, err
		}
		if r, err = r.Narrow(1, at, 1); err != nil {
			return nil, err
		}
		if r, err = r.Contiguous(); err != nil {
			return nil, err
		}
		if rows[b], err = r.Reshape(1, D); err != nil {
			return nil, err
		}
	}
	_ = T
	if B == 1 {
		return rows[0], nil
	}
	return ops.Concat(rows, 0)
}
