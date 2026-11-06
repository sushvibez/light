package models

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"light/device"
	"light/hub"
	"light/safetensors"
	"light/tensor"
)

// mobilenetHFConfig is the subset of HF's MobileNetV2 config.json we need.
type mobilenetHFConfig struct {
	NumChannels      int     `json:"num_channels"`
	ImageSize        int     `json:"image_size"`
	DepthMultiplier  float64 `json:"depth_multiplier"`
	DepthDivisibleBy int     `json:"depth_divisible_by"`
	MinDepth         int     `json:"min_depth"`
	ExpandRatio      float64 `json:"expand_ratio"`
	NumLabels        int     `json:"num_labels"`
}

// LoadMobileNetV2FromHF downloads and loads a pretrained MobileNetV2 from
// HuggingFace (e.g. `google/mobilenet_v2_1.0_224`) into our `*MobileNetV2`.
//
// Architecture mapping notes:
//
//	HF folds the first inverted-residual block (t=1, 32→16) into the stem
//	as `conv_stem.{first_conv, conv_3x3, reduce_1x1}`; we keep them as a
//	separate block. So HF's `layer.{0..15}` map onto our blocks[1..16],
//	and HF's stem covers our Stem + blocks[0].
//
//	Each `MobileNetV2ConvLayer` carries `convolution.weight` (no bias)
//	plus `normalization.{weight, bias, running_mean, running_var}` —
//	identical to our `nn.Conv2d` + `nn.BatchNorm2d` field layout.
//
//	The classification head is `classifier.weight` + `classifier.bias`;
//	no transpose is needed (HF stores nn.Linear weight as (out, in)).
//
// HF includes a pre-classifier dropout that doesn't affect inference; we
// ignore it.
func LoadMobileNetV2FromHF(repo string, dev device.Device) (*MobileNetV2, error) {
	cfg, err := fetchMobileNetConfig(repo)
	if err != nil {
		return nil, err
	}
	stPath, err := hub.Download(repo, "main", "model.safetensors")
	if err != nil {
		return nil, fmt.Errorf("LoadMobileNetV2FromHF: download: %w", err)
	}
	f, err := os.Open(stPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	tensors, _, err := safetensors.Load(f, dev)
	if err != nil {
		return nil, err
	}

	model, err := NewMobileNetV2WithMult(cfg.NumLabels, cfg.DepthMultiplier, dev)
	if err != nil {
		return nil, err
	}
	if err := mapHFToMobileNetV2(model, tensors); err != nil {
		return nil, err
	}
	return model, nil
}

func fetchMobileNetConfig(repo string) (mobilenetHFConfig, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return mobilenetHFConfig{}, fmt.Errorf("fetchMobileNetConfig: %w", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return mobilenetHFConfig{}, err
	}
	var hf mobilenetHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return mobilenetHFConfig{}, fmt.Errorf("fetchMobileNetConfig: parse: %w", err)
	}
	if hf.DepthMultiplier == 0 {
		hf.DepthMultiplier = 1.0
	}
	if hf.NumLabels == 0 {
		hf.NumLabels = 1001 // HF includes a "background" class
	}
	return hf, nil
}

func mapHFToMobileNetV2(m *MobileNetV2, hf map[string]*tensor.Tensor) error {
	// 1) Stem first conv (3 → stemCh).
	if err := copyExact(m.Stem.Weight, hf, "mobilenet_v2.conv_stem.first_conv.convolution.weight"); err != nil {
		return err
	}
	if err := copyBN(m.StemBN, "mobilenet_v2.conv_stem.first_conv.normalization", hf); err != nil {
		return err
	}

	// 2) Block 0 (t=1, no expand) — HF folds this into the stem.
	blk0 := m.Blocks[0]
	if blk0.Expand != nil {
		return fmt.Errorf("LoadMobileNetV2FromHF: blocks[0] unexpectedly has expand layer (config drift?)")
	}
	if err := copyExact(blk0.Depthwise.Weight, hf, "mobilenet_v2.conv_stem.conv_3x3.convolution.weight"); err != nil {
		return err
	}
	if err := copyBN(blk0.DwBN, "mobilenet_v2.conv_stem.conv_3x3.normalization", hf); err != nil {
		return err
	}
	if err := copyExact(blk0.Project.Weight, hf, "mobilenet_v2.conv_stem.reduce_1x1.convolution.weight"); err != nil {
		return err
	}
	if err := copyBN(blk0.ProjectBN, "mobilenet_v2.conv_stem.reduce_1x1.normalization", hf); err != nil {
		return err
	}

	// 3) Remaining blocks — HF `layer.{i}` ↔ ours `Blocks[i+1]`.
	for i := 1; i < len(m.Blocks); i++ {
		hfIdx := i - 1
		p := "mobilenet_v2.layer." + strconv.Itoa(hfIdx) + "."
		blk := m.Blocks[i]
		if blk.Expand == nil {
			return fmt.Errorf("LoadMobileNetV2FromHF: blocks[%d] missing expand (HF model expects t>1 here)", i)
		}
		if err := copyExact(blk.Expand.Weight, hf, p+"expand_1x1.convolution.weight"); err != nil {
			return err
		}
		if err := copyBN(blk.ExpandBN, p+"expand_1x1.normalization", hf); err != nil {
			return err
		}
		if err := copyExact(blk.Depthwise.Weight, hf, p+"conv_3x3.convolution.weight"); err != nil {
			return err
		}
		if err := copyBN(blk.DwBN, p+"conv_3x3.normalization", hf); err != nil {
			return err
		}
		if err := copyExact(blk.Project.Weight, hf, p+"reduce_1x1.convolution.weight"); err != nil {
			return err
		}
		if err := copyBN(blk.ProjectBN, p+"reduce_1x1.normalization", hf); err != nil {
			return err
		}
	}

	// 4) Final 1×1 expansion to 1280.
	if err := copyExact(m.HeadConv.Weight, hf, "mobilenet_v2.conv_1x1.convolution.weight"); err != nil {
		return err
	}
	if err := copyBN(m.HeadBN, "mobilenet_v2.conv_1x1.normalization", hf); err != nil {
		return err
	}

	// 5) Classifier — HF uses a flat `classifier.{weight,bias}` for the linear head.
	if err := copyExact(m.FC.W, hf, "classifier.weight"); err != nil {
		return err
	}
	if err := copyExact(m.FC.B, hf, "classifier.bias"); err != nil {
		return err
	}
	return nil
}
