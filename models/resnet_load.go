package models

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"light/device"
	"light/hub"
	"light/nn"
	"light/safetensors"
	"light/tensor"
)

// bnLike is just an alias to nn.BatchNorm2d so the helper signature reads
// cleanly. Public for users who want to reuse copyBN.
type bnLike = nn.BatchNorm2d

// resnetHFConfig is the subset of HF's ResNet `config.json` we read.
type resnetHFConfig struct {
	NumChannels   int     `json:"num_channels"`
	EmbeddingSize int     `json:"embedding_size"`
	HiddenSizes   []int   `json:"hidden_sizes"`
	Depths        []int   `json:"depths"`
	LayerType     string  `json:"layer_type"`
	HiddenAct     string  `json:"hidden_act"`
	NumLabels     int     `json:"num_labels"`
	LayerNormEps  float64 `json:"layer_norm_eps"`
}

// LoadResNetFromHF downloads and loads a pretrained HuggingFace ResNet
// (e.g. `microsoft/resnet-50`) into our `*ResNet`. Use this to compare
// outputs against `transformers.ResNetForImageClassification` on real
// images.
//
// Mapping notes:
//   - HF stores Conv2d weights as (out, in, kH, kW) — same as ours.
//   - BatchNorm running mean/var are loaded into our `RunningMean`/
//     `RunningVar` tensors. `num_batches_tracked` is ignored.
//   - HF's stem path is `resnet.embedder.embedder.{convolution,normalization}`;
//     each bottleneck has three `layer.{0,1,2}.{convolution,normalization}`
//     pairs and an optional `shortcut.{convolution,normalization}` on the
//     first block of every stage where the shape changes.
//   - The classifier is `classifier.1.{weight,bias}` (`classifier.0` is a
//     `nn.Flatten` with no params).
func LoadResNetFromHF(repo string, dev device.Device) (*ResNet, error) {
	cfg, err := fetchResNetConfig(repo)
	if err != nil {
		return nil, err
	}
	stPath, err := hub.Download(repo, "main", "model.safetensors")
	if err != nil {
		return nil, fmt.Errorf("LoadResNetFromHF: download: %w", err)
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

	// Map HF depths → our resnetSpec.
	if len(cfg.Depths) != 4 {
		return nil, fmt.Errorf("LoadResNetFromHF: expected 4 stages, got %d", len(cfg.Depths))
	}
	spec := resnetSpec{
		blocks:    [4]int{cfg.Depths[0], cfg.Depths[1], cfg.Depths[2], cfg.Depths[3]},
		useBottle: cfg.LayerType == "bottleneck",
	}
	model, err := newResNet(spec, cfg.NumLabels, dev)
	if err != nil {
		return nil, err
	}
	if err := mapHFToResNet(model, spec, tensors); err != nil {
		return nil, err
	}
	return model, nil
}

func fetchResNetConfig(repo string) (resnetHFConfig, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return resnetHFConfig{}, fmt.Errorf("fetchResNetConfig: %w", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return resnetHFConfig{}, err
	}
	var hf resnetHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return resnetHFConfig{}, fmt.Errorf("fetchResNetConfig: parse: %w", err)
	}
	if hf.NumLabels == 0 {
		hf.NumLabels = 1000
	}
	if hf.LayerType == "" {
		hf.LayerType = "bottleneck"
	}
	return hf, nil
}

func mapHFToResNet(r *ResNet, spec resnetSpec, hf map[string]*tensor.Tensor) error {
	// 1) Stem.
	if err := copyExact(r.Stem.Weight, hf, "resnet.embedder.embedder.convolution.weight"); err != nil {
		return err
	}
	if err := copyBN(r.StemBN, "resnet.embedder.embedder.normalization", hf); err != nil {
		return err
	}

	// 2) Stages.
	for s, stage := range r.Stages {
		for i, b := range stage {
			p := "resnet.encoder.stages." + strconv.Itoa(s) + ".layers." + strconv.Itoa(i) + "."
			if spec.useBottle {
				bb := b.(*bottleneck)
				if err := copyExact(bb.Conv1.Weight, hf, p+"layer.0.convolution.weight"); err != nil {
					return err
				}
				if err := copyBN(bb.BN1, p+"layer.0.normalization", hf); err != nil {
					return err
				}
				if err := copyExact(bb.Conv2.Weight, hf, p+"layer.1.convolution.weight"); err != nil {
					return err
				}
				if err := copyBN(bb.BN2, p+"layer.1.normalization", hf); err != nil {
					return err
				}
				if err := copyExact(bb.Conv3.Weight, hf, p+"layer.2.convolution.weight"); err != nil {
					return err
				}
				if err := copyBN(bb.BN3, p+"layer.2.normalization", hf); err != nil {
					return err
				}
				if bb.Down != nil {
					if err := copyExact(bb.Down.Weight, hf, p+"shortcut.convolution.weight"); err != nil {
						return err
					}
					if err := copyBN(bb.DownBN, p+"shortcut.normalization", hf); err != nil {
						return err
					}
				}
			} else {
				bb := b.(*basicBlock)
				if err := copyExact(bb.Conv1.Weight, hf, p+"layer.0.convolution.weight"); err != nil {
					return err
				}
				if err := copyBN(bb.BN1, p+"layer.0.normalization", hf); err != nil {
					return err
				}
				if err := copyExact(bb.Conv2.Weight, hf, p+"layer.1.convolution.weight"); err != nil {
					return err
				}
				if err := copyBN(bb.BN2, p+"layer.1.normalization", hf); err != nil {
					return err
				}
				if bb.Down != nil {
					if err := copyExact(bb.Down.Weight, hf, p+"shortcut.convolution.weight"); err != nil {
						return err
					}
					if err := copyBN(bb.DownBN, p+"shortcut.normalization", hf); err != nil {
						return err
					}
				}
			}
		}
	}

	// 3) Classifier head — "classifier.1" (classifier.0 is the Flatten).
	if err := copyExact(r.FC.W, hf, "classifier.1.weight"); err != nil {
		return err
	}
	if err := copyExact(r.FC.B, hf, "classifier.1.bias"); err != nil {
		return err
	}
	return nil
}

// copyBN copies the four BatchNorm tensors (gamma, beta, running_mean,
// running_var) under the given HF prefix into our nn.BatchNorm2d module.
// Skips `num_batches_tracked` since we don't track it.
func copyBN(bn *bnLike, prefix string, hf map[string]*tensor.Tensor) error {
	if err := copyExact(bn.Weight, hf, prefix+".weight"); err != nil {
		return err
	}
	if err := copyExact(bn.Bias, hf, prefix+".bias"); err != nil {
		return err
	}
	if err := copyExact(bn.RunningMean, hf, prefix+".running_mean"); err != nil {
		return err
	}
	if err := copyExact(bn.RunningVar, hf, prefix+".running_var"); err != nil {
		return err
	}
	return nil
}
