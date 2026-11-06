package models

// LoadMercyFromSafetensors loads a MercyLM from a safetensors file produced
// by the reference `tools/export_to_hf.py` script.
//
// Expected tensor keys (PyTorch nn.Module state_dict layout):
//
//	token_emb.weight                    (V, D)
//	pos_emb.weight                      (C, D)   C = context_len
//	blocks.{i}.norm1.weight             (D,)
//	blocks.{i}.norm1.bias               (D,)
//	blocks.{i}.attn.in_proj_weight      (3D, D)  — Q, K, V stacked row-major
//	blocks.{i}.attn.out_proj.weight     (D, D)
//	blocks.{i}.norm2.weight             (D,)
//	blocks.{i}.norm2.bias               (D,)
//	blocks.{i}.ff.0.weight              (FFN, D)
//	blocks.{i}.ff.0.bias                (FFN,)
//	blocks.{i}.ff.3.weight              (D, FFN)
//	blocks.{i}.ff.3.bias                (D,)
//	norm.weight                         (D,)
//	norm.bias                           (D,)
//	lm_head.weight                      (V, D)   — absent when tied
//
// The attention bias is False in the reference model, so no
// `in_proj_bias` / `out_proj.bias` tensors are expected.

import (
	"fmt"
	"os"
	"strconv"

	"light/device"
	"light/safetensors"
	"light/tensor"
)

// LoadMercyFromSafetensors reads `path` and loads weights into a newly
// constructed MercyLM on `dev`.
func LoadMercyFromSafetensors(path string, cfg MercyConfig, dev device.Device) (*MercyLM, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("LoadMercyFromSafetensors: %w", err)
	}
	defer f.Close()

	hf, _, err := safetensors.Load(f, dev)
	if err != nil {
		return nil, fmt.Errorf("LoadMercyFromSafetensors: load: %w", err)
	}

	m, err := NewMercyLMUninit(cfg, dev)
	if err != nil {
		return nil, err
	}
	if err := mapSafetensorsToMercy(m, hf); err != nil {
		return nil, err
	}
	return m, nil
}

func mapSafetensorsToMercy(m *MercyLM, hf map[string]*tensor.Tensor) error {
	D := m.Cfg.EmbedDim

	if err := mercyCopy(m.TokenEmb.Weight, hf, "token_emb.weight"); err != nil {
		return err
	}
	if err := mercyCopy(m.PosEmb.Weight, hf, "pos_emb.weight"); err != nil {
		return err
	}

	for i, blk := range m.Blocks {
		p := "blocks." + strconv.Itoa(i) + "."

		if err := mercyCopy(blk.Norm1.Gamma, hf, p+"norm1.weight"); err != nil {
			return err
		}
		if err := mercyCopy(blk.Norm1.Beta, hf, p+"norm1.bias"); err != nil {
			return err
		}

		// in_proj_weight: (3D, D) — rows [0:D] = Q, [D:2D] = K, [2D:3D] = V.
		// PyTorch nn.Linear stores (out, in) same as our nn.Linear, so no
		// transpose is needed; just slice along axis-0.
		inProjW, err := mercyGet(hf, p+"attn.in_proj_weight")
		if err != nil {
			return err
		}
		if inProjW.Rank() != 2 || inProjW.Dim(0) != 3*D || inProjW.Dim(1) != D {
			return fmt.Errorf("%v: in_proj_weight shape %v, want (%d, %d)", p, inProjW.Shape(), 3*D, D)
		}
		for j, qkv := range []*tensor.Tensor{
			blk.Attn.QProj.W, blk.Attn.KProj.W, blk.Attn.VProj.W,
		} {
			row, err := inProjW.Narrow(0, j*D, D)
			if err != nil {
				return fmt.Errorf("%vattn in_proj slice %d: %w", p, j, err)
			}
			c, err := row.Contiguous()
			if err != nil {
				return err
			}
			if err := qkv.CopyFrom(c); err != nil {
				return fmt.Errorf("%vattn QKV[%d]: %w", p, j, err)
			}
		}

		if err := mercyCopy(blk.Attn.OProj.W, hf, p+"attn.out_proj.weight"); err != nil {
			return err
		}

		if err := mercyCopy(blk.Norm2.Gamma, hf, p+"norm2.weight"); err != nil {
			return err
		}
		if err := mercyCopy(blk.Norm2.Beta, hf, p+"norm2.bias"); err != nil {
			return err
		}

		if err := mercyCopy(blk.FC1.W, hf, p+"ff.0.weight"); err != nil {
			return err
		}
		if err := mercyCopy(blk.FC1.B, hf, p+"ff.0.bias"); err != nil {
			return err
		}
		if err := mercyCopy(blk.FC2.W, hf, p+"ff.3.weight"); err != nil {
			return err
		}
		if err := mercyCopy(blk.FC2.B, hf, p+"ff.3.bias"); err != nil {
			return err
		}
	}

	if err := mercyCopy(m.FinalNorm.Gamma, hf, "norm.weight"); err != nil {
		return err
	}
	if err := mercyCopy(m.FinalNorm.Beta, hf, "norm.bias"); err != nil {
		return err
	}

	// lm_head.weight is optional — if absent, the tied weight was already
	// loaded via token_emb.weight.
	if lmW, ok := hf["lm_head.weight"]; ok {
		if err := m.LMHead.W.CopyFrom(lmW); err != nil {
			return fmt.Errorf("lm_head.weight: %w", err)
		}
	}
	return nil
}

func mercyGet(hf map[string]*tensor.Tensor, name string) (*tensor.Tensor, error) {
	t, ok := hf[name]
	if !ok {
		return nil, fmt.Errorf("MercyLM: missing tensor %q", name)
	}
	return t, nil
}

func mercyCopy(dst *tensor.Tensor, hf map[string]*tensor.Tensor, name string) error {
	src, err := mercyGet(hf, name)
	if err != nil {
		return err
	}
	if !dst.Shape().Equal(src.Shape()) {
		return fmt.Errorf("%s: shape mismatch (model %v, file %v)", name, dst.Shape(), src.Shape())
	}
	if err := dst.CopyFrom(src); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	return nil
}
