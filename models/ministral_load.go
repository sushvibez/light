package models

import (
	"fmt"
	"strings"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// LoadMinistralFromHF loads the text model of a Ministral 3 checkpoint
// (`mistral3` multimodal wrapper, e.g. mistralai/Ministral-3-3B-Instruct-2512,
// or a bare `ministral3` text model) as a *Llama: a Llama decoder with
// YaRN RoPE and Llama 4's position-dependent query scaling
// (`llama_4_scaling_beta`). The Pixtral vision tower and the multimodal
// projector are skipped.
//
// FP8 checkpoints (per-tensor F8E4M3 weights with a scalar
// `weight_scale_inv`, static `activation_scale`) are dequantized at load
// exactly as transformers does without an FP8 GPU: f32(w) · scale, rounded
// once to BF16 (the scale's dtype, even for an F32 model); the static
// activation scales only matter to FP8 kernels.
func LoadMinistralFromHF(repo string, dev device.Device, weightDT dtype.DType) (*Llama, error) {
	cfg, err := fetchLlamaConfig(repo)
	if err != nil {
		return nil, err
	}
	cfg.WeightDType = weightDT
	cfg.SkipInit = true
	raw, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadMinistralFromHF: %w", err)
	}
	hf, err := ministralTextTensors(raw, weightDT)
	if err != nil {
		return nil, err
	}
	m, err := NewLlama(cfg, dev)
	if err != nil {
		return nil, err
	}
	if err := mapHFToLlama(m, hf); err != nil {
		return nil, fmt.Errorf("LoadMinistralFromHF: %w", err)
	}
	return m, nil
}

// ministralTextTensors renames the language model's tensors to the plain
// Llama layout (`language_model.model.X` / `model.language_model.X` →
// `model.X`, `language_model.lm_head` → `lm_head`), dequantizes FP8 linears
// and drops the vision tower, projector and FP8 scale tensors.
func ministralTextTensors(raw map[string]*tensor.Tensor, weightDT dtype.DType) (map[string]*tensor.Tensor, error) {
	// transformers dequantizes into the scale's dtype (BF16) whatever the
	// model dtype; an F32 load then widens those BF16 weights exactly.
	deqDT := dtype.BF16
	_ = weightDT
	out := make(map[string]*tensor.Tensor, len(raw))
	rename := func(k string) (string, bool) {
		for _, p := range []string{"language_model.model.", "model.language_model."} {
			if strings.HasPrefix(k, p) {
				return "model." + strings.TrimPrefix(k, p), true
			}
		}
		switch k {
		case "language_model.lm_head.weight", "lm_head.weight":
			return "lm_head.weight", true
		}
		if strings.HasPrefix(k, "model.") && !strings.HasPrefix(k, "model.vision_tower.") && !strings.HasPrefix(k, "model.multi_modal_projector.") {
			return k, true // a bare ministral3 text checkpoint
		}
		return "", false // vision tower, projector
	}
	for k, t := range raw {
		if strings.HasSuffix(k, ".weight_scale_inv") || strings.HasSuffix(k, ".activation_scale") {
			continue
		}
		name, ok := rename(k)
		if !ok {
			continue
		}
		if t.DType() == dtype.F8E4M3 {
			base := strings.TrimSuffix(k, ".weight")
			s, ok := raw[base+".weight_scale_inv"]
			if !ok {
				return nil, fmt.Errorf("ministral: FP8 weight %s has no weight_scale_inv", k)
			}
			if s.NumElements() != 1 {
				return nil, fmt.Errorf("ministral: %s.weight_scale_inv has %d elements; only per-tensor FP8 scales are supported", base, s.NumElements())
			}
			s32, err := ops.Cast(s, dtype.F32)
			if err != nil {
				return nil, err
			}
			if s32, err = s32.Reshape(1, 1); err != nil {
				return nil, err
			}
			if t, err = ops.DequantizeF8Block(t, s32, t.Dim(0), t.Dim(1), deqDT); err != nil {
				return nil, fmt.Errorf("ministral: %s: %w", k, err)
			}
		}
		out[name] = t
	}
	return out, nil
}
