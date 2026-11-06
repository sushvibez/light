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

// vitHFConfig is the subset of HuggingFace's ViT config.json we read.
type vitHFConfig struct {
	HiddenSize        int     `json:"hidden_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	IntermediateSize  int     `json:"intermediate_size"`
	HiddenAct         string  `json:"hidden_act"`
	LayerNormEps      float64 `json:"layer_norm_eps"`
	ImageSize         int     `json:"image_size"`
	PatchSize         int     `json:"patch_size"`
	NumChannels       int     `json:"num_channels"`
	NumLabels         int     `json:"num_labels"`
}

// LoadViTFromHF downloads `config.json` + `model.safetensors` for an HF
// ViT image-classification repo (e.g. `google/vit-base-patch16-224`) and
// returns a `*ViT` with weights copied in. Greedy classification logits
// match PyTorch's `transformers.ViTForImageClassification`.
//
// Mapping notes:
//   - HF stores `nn.Linear` weight as (out, in) — same as ours, so Q/K/V/O
//     and MLP linears copy directly without transpose.
//   - `vit.embeddings.cls_token` is (1, 1, 768); `position_embeddings` is
//     (1, num_patches+1, 768). Both copy as-is.
//   - The patch-embedding `projection.weight` is (768, 3, 16, 16) — same
//     layout as our `nn.Conv2d`.
func LoadViTFromHF(repo string, dev device.Device) (*ViT, error) {
	cfg, numLabels, err := fetchViTConfig(repo)
	if err != nil {
		return nil, err
	}
	stPath, err := hub.Download(repo, "main", "model.safetensors")
	if err != nil {
		return nil, fmt.Errorf("LoadViTFromHF: download: %w", err)
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

	cfg.NumClasses = numLabels
	vit, err := NewViT(cfg, dev)
	if err != nil {
		return nil, err
	}
	if err := mapHFToViT(vit, tensors); err != nil {
		return nil, err
	}
	return vit, nil
}

func fetchViTConfig(repo string) (ViTConfig, int, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return ViTConfig{}, 0, fmt.Errorf("fetchViTConfig: %w", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return ViTConfig{}, 0, err
	}
	var hf vitHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return ViTConfig{}, 0, fmt.Errorf("fetchViTConfig: parse: %w", err)
	}
	if hf.LayerNormEps == 0 {
		hf.LayerNormEps = 1e-12
	}
	if hf.NumLabels == 0 {
		hf.NumLabels = 1000
	}
	mlpRatio := float64(hf.IntermediateSize) / float64(hf.HiddenSize)
	cfg := ViTConfig{
		ImgSize: hf.ImageSize, PatchSize: hf.PatchSize,
		InChannels: hf.NumChannels, NumClasses: hf.NumLabels,
		NEmbd: hf.HiddenSize, NLayer: hf.NumHiddenLayers, NHead: hf.NumAttentionHeads,
		MlpRatio: mlpRatio, LayerNorm: hf.LayerNormEps,
	}
	return cfg, hf.NumLabels, nil
}

func mapHFToViT(vit *ViT, hf map[string]*tensor.Tensor) error {
	// 1) Embeddings.
	if err := copyExact(vit.PatchConv.Weight, hf, "vit.embeddings.patch_embeddings.projection.weight"); err != nil {
		return err
	}
	if err := copyExact(vit.PatchConv.Bias, hf, "vit.embeddings.patch_embeddings.projection.bias"); err != nil {
		return err
	}
	if err := copyExact(vit.ClsToken, hf, "vit.embeddings.cls_token"); err != nil {
		return err
	}
	if err := copyExact(vit.PosEmb, hf, "vit.embeddings.position_embeddings"); err != nil {
		return err
	}
	// 2) Encoder blocks.
	for i, blk := range vit.Blocks {
		p := "vit.encoder.layer." + strconv.Itoa(i) + "."
		if err := copyExact(blk.Norm1.Gamma, hf, p+"layernorm_before.weight"); err != nil {
			return err
		}
		if err := copyExact(blk.Norm1.Beta, hf, p+"layernorm_before.bias"); err != nil {
			return err
		}
		if err := copyExact(blk.Attn.QProj.W, hf, p+"attention.attention.query.weight"); err != nil {
			return err
		}
		if err := copyExact(blk.Attn.QProj.B, hf, p+"attention.attention.query.bias"); err != nil {
			return err
		}
		if err := copyExact(blk.Attn.KProj.W, hf, p+"attention.attention.key.weight"); err != nil {
			return err
		}
		if err := copyExact(blk.Attn.KProj.B, hf, p+"attention.attention.key.bias"); err != nil {
			return err
		}
		if err := copyExact(blk.Attn.VProj.W, hf, p+"attention.attention.value.weight"); err != nil {
			return err
		}
		if err := copyExact(blk.Attn.VProj.B, hf, p+"attention.attention.value.bias"); err != nil {
			return err
		}
		if err := copyExact(blk.Attn.OProj.W, hf, p+"attention.output.dense.weight"); err != nil {
			return err
		}
		if err := copyExact(blk.Attn.OProj.B, hf, p+"attention.output.dense.bias"); err != nil {
			return err
		}
		if err := copyExact(blk.Norm2.Gamma, hf, p+"layernorm_after.weight"); err != nil {
			return err
		}
		if err := copyExact(blk.Norm2.Beta, hf, p+"layernorm_after.bias"); err != nil {
			return err
		}
		if err := copyExact(blk.FC1.W, hf, p+"intermediate.dense.weight"); err != nil {
			return err
		}
		if err := copyExact(blk.FC1.B, hf, p+"intermediate.dense.bias"); err != nil {
			return err
		}
		if err := copyExact(blk.FC2.W, hf, p+"output.dense.weight"); err != nil {
			return err
		}
		if err := copyExact(blk.FC2.B, hf, p+"output.dense.bias"); err != nil {
			return err
		}
	}
	// 3) Final LayerNorm.
	if err := copyExact(vit.NormFinal.Gamma, hf, "vit.layernorm.weight"); err != nil {
		return err
	}
	if err := copyExact(vit.NormFinal.Beta, hf, "vit.layernorm.bias"); err != nil {
		return err
	}
	// 4) Classification head.
	if err := copyExact(vit.Head.W, hf, "classifier.weight"); err != nil {
		return err
	}
	if err := copyExact(vit.Head.B, hf, "classifier.bias"); err != nil {
		return err
	}
	return nil
}
