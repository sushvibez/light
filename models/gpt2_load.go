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

// GPT2Variant captures the public GPT-2 variants we know how to map.
type GPT2Variant string

const (
	GPT2VariantSmall  GPT2Variant = "gpt2"
	GPT2VariantMedium GPT2Variant = "gpt2-medium"
	GPT2VariantLarge  GPT2Variant = "gpt2-large"
	GPT2VariantXL     GPT2Variant = "gpt2-xl"
)

// gpt2HFConfig is the subset of HuggingFace's `config.json` we care about.
type gpt2HFConfig struct {
	VocabSize    int     `json:"vocab_size"`
	NPositions   int     `json:"n_positions"`
	NEmbd        int     `json:"n_embd"`
	NLayer       int     `json:"n_layer"`
	NHead        int     `json:"n_head"`
	LayerNormEps float64 `json:"layer_norm_epsilon"`
}

// LoadGPT2FromHF downloads and loads a pretrained GPT-2 from HuggingFace
// (e.g. `gpt2`, `gpt2-medium`). Returns a fully-initialised GPT2 ready for
// inference on `dev`. The model weights are fetched once and cached under
// `~/.cache/huggingface/hub/`.
//
// The mapping handles GPT-2's three storage idiosyncrasies:
//
//  1. HF stores Conv1D weights as (in, out) — opposite of `nn.Linear`'s
//     (out, in) — so each weight is transposed before loading.
//  2. The Q/K/V projections are fused into a single `c_attn.weight` of shape
//     (D, 3D); we slice it back into three (D, D) blocks.
//  3. The LM-head shares storage with the token embedding (HF doesn't ship
//     a separate `lm_head.weight`).
func LoadGPT2FromHF(repo string, dev device.Device) (*GPT2, error) {
	cfg, err := fetchGPT2Config(repo)
	if err != nil {
		return nil, err
	}
	stPath, err := hub.Download(repo, "main", "model.safetensors")
	if err != nil {
		return nil, fmt.Errorf("LoadGPT2FromHF: download: %w", err)
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

	gpt, err := NewGPT2(cfg, dev)
	if err != nil {
		return nil, err
	}
	if err := mapHFToGPT2(gpt, tensors); err != nil {
		return nil, err
	}
	return gpt, nil
}

// fetchGPT2Config downloads the model's `config.json` and converts it to our
// GPT2Config. Defaults match the public GPT-2 family.
func fetchGPT2Config(repo string) (GPT2Config, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return GPT2Config{}, fmt.Errorf("fetchGPT2Config: %w", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return GPT2Config{}, err
	}
	var hf gpt2HFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return GPT2Config{}, fmt.Errorf("fetchGPT2Config: parse: %w", err)
	}
	if hf.LayerNormEps == 0 {
		hf.LayerNormEps = 1e-5
	}
	return GPT2Config{
		VocabSize: hf.VocabSize, NPositions: hf.NPositions,
		NEmbd: hf.NEmbd, NLayer: hf.NLayer, NHead: hf.NHead,
		LayerNorm: hf.LayerNormEps, Causal: true, Tied: true,
	}, nil
}

// mapHFToGPT2 walks the HF tensor map and copies values into `gpt`'s
// parameters in place. Errors out on the first shape mismatch so the user
// sees what went wrong.
func mapHFToGPT2(gpt *GPT2, hf map[string]*tensor.Tensor) error {
	D := gpt.Cfg.NEmbd

	// 1) Token embedding: wte.weight (V, D)
	if err := copyExact(gpt.TokenEmb.Weight, hf, "wte.weight"); err != nil {
		return err
	}
	// 2) Position embedding: wpe.weight (P, D)
	if err := copyExact(gpt.PosEmb.Weight, hf, "wpe.weight"); err != nil {
		return err
	}
	// 3) Each transformer block.
	for i, blk := range gpt.Blocks {
		p := "h." + strconv.Itoa(i) + "."
		// LayerNorm 1
		if err := copyExact(blk.Norm1.Gamma, hf, p+"ln_1.weight"); err != nil {
			return err
		}
		if err := copyExact(blk.Norm1.Beta, hf, p+"ln_1.bias"); err != nil {
			return err
		}
		// Fused QKV: c_attn.weight (D, 3D), c_attn.bias (3D)
		cAttnW, err := mustGet(hf, p+"attn.c_attn.weight")
		if err != nil {
			return err
		}
		cAttnB, err := mustGet(hf, p+"attn.c_attn.bias")
		if err != nil {
			return err
		}
		if cAttnW.Dim(0) != D || cAttnW.Dim(1) != 3*D {
			return fmt.Errorf("c_attn.weight: shape %v, want (%d, %d)", cAttnW.Shape(), D, 3*D)
		}
		// Slice along axis-1 into 3 chunks of width D, each (D, D).
		// Then transpose to (D, D) (HF Conv1D stores in→out; our Linear is out→in).
		for j, proj := range []*struct {
			W, B *tensor.Tensor
			off  int
		}{
			{blk.Attn.QProj.W, blk.Attn.QProj.B, 0 * D},
			{blk.Attn.KProj.W, blk.Attn.KProj.B, 1 * D},
			{blk.Attn.VProj.W, blk.Attn.VProj.B, 2 * D},
		} {
			wSlice, err := cAttnW.Narrow(1, proj.off, D)
			if err != nil {
				return err
			}
			wT, err := wSlice.Transpose(0, 1)
			if err != nil {
				return err
			}
			wTC, err := wT.Contiguous()
			if err != nil {
				return err
			}
			if err := proj.W.CopyFrom(wTC); err != nil {
				return fmt.Errorf("c_attn proj %d weight: %w", j, err)
			}
			bSlice, err := cAttnB.Narrow(0, proj.off, D)
			if err != nil {
				return err
			}
			bC, err := bSlice.Contiguous()
			if err != nil {
				return err
			}
			if err := proj.B.CopyFrom(bC); err != nil {
				return fmt.Errorf("c_attn proj %d bias: %w", j, err)
			}
		}
		// Output projection: c_proj.weight (D, D), c_proj.bias (D,)
		if err := copyTransposed(blk.Attn.OProj.W, hf, p+"attn.c_proj.weight"); err != nil {
			return err
		}
		if err := copyExact(blk.Attn.OProj.B, hf, p+"attn.c_proj.bias"); err != nil {
			return err
		}
		// LayerNorm 2
		if err := copyExact(blk.Norm2.Gamma, hf, p+"ln_2.weight"); err != nil {
			return err
		}
		if err := copyExact(blk.Norm2.Beta, hf, p+"ln_2.bias"); err != nil {
			return err
		}
		// FFN: c_fc.weight (D, 4D) → FC1.W (4D, D); c_proj.weight (4D, D) → FC2.W (D, 4D)
		if err := copyTransposed(blk.FC1.W, hf, p+"mlp.c_fc.weight"); err != nil {
			return err
		}
		if err := copyExact(blk.FC1.B, hf, p+"mlp.c_fc.bias"); err != nil {
			return err
		}
		if err := copyTransposed(blk.FC2.W, hf, p+"mlp.c_proj.weight"); err != nil {
			return err
		}
		if err := copyExact(blk.FC2.B, hf, p+"mlp.c_proj.bias"); err != nil {
			return err
		}
	}
	// 4) Final LayerNorm.
	if err := copyExact(gpt.FinalNorm.Gamma, hf, "ln_f.weight"); err != nil {
		return err
	}
	if err := copyExact(gpt.FinalNorm.Beta, hf, "ln_f.bias"); err != nil {
		return err
	}
	// 5) LM head: tied — already shares storage with TokenEmb.Weight.
	if !gpt.Cfg.Tied {
		if err := copyExact(gpt.LMHead.W, hf, "lm_head.weight"); err != nil {
			return err
		}
	}
	return nil
}

// mustGet returns the named tensor or a descriptive error.
func mustGet(hf map[string]*tensor.Tensor, name string) (*tensor.Tensor, error) {
	t, ok := hf[name]
	if !ok {
		return nil, fmt.Errorf("HF GPT-2: missing tensor %q", name)
	}
	return t, nil
}

// copyExact copies hf[name] into dst with no transformation, validating shape.
func copyExact(dst *tensor.Tensor, hf map[string]*tensor.Tensor, name string) error {
	src, err := mustGet(hf, name)
	if err != nil {
		return err
	}
	if !dst.Shape().Equal(src.Shape()) {
		return fmt.Errorf("%s: shape mismatch (model %v, HF %v)", name, dst.Shape(), src.Shape())
	}
	if err := dst.CopyFrom(src); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	return nil
}

// copyTransposed copies hf[name].T into dst (HF Conv1D weight → nn.Linear).
func copyTransposed(dst *tensor.Tensor, hf map[string]*tensor.Tensor, name string) error {
	src, err := mustGet(hf, name)
	if err != nil {
		return err
	}
	srcT, err := src.Transpose(0, 1)
	if err != nil {
		return err
	}
	srcTC, err := srcT.Contiguous()
	if err != nil {
		return err
	}
	if !dst.Shape().Equal(srcTC.Shape()) {
		return fmt.Errorf("%s: shape mismatch after transpose (model %v, HF.T %v)", name, dst.Shape(), srcTC.Shape())
	}
	if err := dst.CopyFrom(srcTC); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	return nil
}
