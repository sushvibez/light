package models

import (
	"fmt"
	"os"
	"strconv"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/nn"
	"light/safetensors"
	"light/tensor"
	"light/tokenizer"
)

// LoadSD15FromHF downloads `runwayml/stable-diffusion-v1-5` (or a SD 1.x
// repo with the same layout) at F32. See LoadSD15FromHFWithDType for the
// half-memory BF16 variant.
func LoadSD15FromHF(repo string, dev device.Device) (*StableDiffusion, error) {
	return LoadSD15FromHFWithDType(repo, dtype.F32, dev)
}

// LoadSD15FromHFWithDType loads SD 1.x with UNet + VAE weights stored at
// `wdt`. CLIP stays F32 regardless (it's small — ~492 MB at F32 — and the
// CLIP context is cast to `wdt` before the UNet forward). For Stable
// Diffusion 1.5 on a 16 GB GPU, BF16 drops UNet from 3.4 GB → 1.7 GB and
// VAE from 200 MB → 100 MB, giving plenty of headroom for the per-step
// activations.
func LoadSD15FromHFWithDType(repo string, wdt dtype.DType, dev device.Device) (*StableDiffusion, error) {
	// 1. Tokenizer.
	vocabPath, err := hub.Download(repo, "main", "tokenizer/vocab.json")
	if err != nil {
		return nil, fmt.Errorf("tokenizer/vocab.json: %w", err)
	}
	mergesPath, err := hub.Download(repo, "main", "tokenizer/merges.txt")
	if err != nil {
		return nil, fmt.Errorf("tokenizer/merges.txt: %w", err)
	}
	vocabF, err := os.Open(vocabPath)
	if err != nil {
		return nil, err
	}
	mergesF, err := os.Open(mergesPath)
	if err != nil {
		vocabF.Close()
		return nil, err
	}
	tok, err := tokenizer.LoadCLIPTokenizer(vocabF, mergesF)
	vocabF.Close()
	mergesF.Close()
	if err != nil {
		return nil, fmt.Errorf("CLIP tokenizer: %w", err)
	}
	// 2. Text encoder.
	te, err := loadCLIPTextSD15(repo, dev)
	if err != nil {
		return nil, fmt.Errorf("text encoder: %w", err)
	}
	// 3. UNet (at the requested dtype).
	unet, err := loadSDUNetSD15(repo, wdt, dev)
	if err != nil {
		return nil, fmt.Errorf("unet: %w", err)
	}
	// 4. VAE decoder (at the requested dtype).
	vae, err := loadSDVAESD15(repo, wdt, dev)
	if err != nil {
		return nil, fmt.Errorf("vae: %w", err)
	}
	return &StableDiffusion{
		TextEncoder: te,
		UNet:        unet,
		VAE:         vae,
		Tokenizer:   tok,
		Scheduler:   NewDDIMScheduler(50),
	}, nil
}

func loadComponentSafetensors(repo, subdir string, dev device.Device) (map[string]*tensor.Tensor, error) {
	// Try the subdir's `model.safetensors` (CLIP) or
	// `diffusion_pytorch_model.safetensors` (UNet/VAE) — whichever exists.
	candidates := []string{
		subdir + "/diffusion_pytorch_model.safetensors",
		subdir + "/model.safetensors",
	}
	var path string
	for _, c := range candidates {
		p, err := hub.Download(repo, "main", c)
		if err == nil {
			path = p
			break
		}
	}
	if path == "" {
		return nil, fmt.Errorf("loadComponentSafetensors %s: no checkpoint found", subdir)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	tensors, _, err := safetensors.Load(f, device.CPU())
	return tensors, err
}

// loadCLIPTextSD15 loads the SD 1.5 text encoder (HF CLIPTextModel layout).
func loadCLIPTextSD15(repo string, dev device.Device) (*CLIPText, error) {
	weights, err := loadComponentSafetensors(repo, "text_encoder", dev)
	if err != nil {
		return nil, err
	}
	cfg := CLIPTextSD15()
	te, err := NewCLIPText(cfg, true, dev)
	if err != nil {
		return nil, err
	}
	prefix := "text_model."
	if err := copyExactCast(te.TokenEmb.Weight, weights, prefix+"embeddings.token_embedding.weight"); err != nil {
		return nil, err
	}
	if err := copyExactCast(te.PosEmb.Weight, weights, prefix+"embeddings.position_embedding.weight"); err != nil {
		return nil, err
	}
	for i, blk := range te.Blocks {
		p := fmt.Sprintf("%sencoder.layers.%d.", prefix, i)
		if err := copyExactCast(blk.N1.Gamma, weights, p+"layer_norm1.weight"); err != nil {
			return nil, err
		}
		if err := copyExactCast(blk.N1.Beta, weights, p+"layer_norm1.bias"); err != nil {
			return nil, err
		}
		if err := copyExactCast(blk.Attn.QProj.W, weights, p+"self_attn.q_proj.weight"); err != nil {
			return nil, err
		}
		if err := copyExactCast(blk.Attn.QProj.B, weights, p+"self_attn.q_proj.bias"); err != nil {
			return nil, err
		}
		if err := copyExactCast(blk.Attn.KProj.W, weights, p+"self_attn.k_proj.weight"); err != nil {
			return nil, err
		}
		if err := copyExactCast(blk.Attn.KProj.B, weights, p+"self_attn.k_proj.bias"); err != nil {
			return nil, err
		}
		if err := copyExactCast(blk.Attn.VProj.W, weights, p+"self_attn.v_proj.weight"); err != nil {
			return nil, err
		}
		if err := copyExactCast(blk.Attn.VProj.B, weights, p+"self_attn.v_proj.bias"); err != nil {
			return nil, err
		}
		if err := copyExactCast(blk.Attn.OProj.W, weights, p+"self_attn.out_proj.weight"); err != nil {
			return nil, err
		}
		if err := copyExactCast(blk.Attn.OProj.B, weights, p+"self_attn.out_proj.bias"); err != nil {
			return nil, err
		}
		if err := copyExactCast(blk.N2.Gamma, weights, p+"layer_norm2.weight"); err != nil {
			return nil, err
		}
		if err := copyExactCast(blk.N2.Beta, weights, p+"layer_norm2.bias"); err != nil {
			return nil, err
		}
		if err := copyExactCast(blk.FC1.W, weights, p+"mlp.fc1.weight"); err != nil {
			return nil, err
		}
		if err := copyExactCast(blk.FC1.B, weights, p+"mlp.fc1.bias"); err != nil {
			return nil, err
		}
		if err := copyExactCast(blk.FC2.W, weights, p+"mlp.fc2.weight"); err != nil {
			return nil, err
		}
		if err := copyExactCast(blk.FC2.B, weights, p+"mlp.fc2.bias"); err != nil {
			return nil, err
		}
	}
	if err := copyExactCast(te.FinalNorm.Gamma, weights, prefix+"final_layer_norm.weight"); err != nil {
		return nil, err
	}
	if err := copyExactCast(te.FinalNorm.Beta, weights, prefix+"final_layer_norm.bias"); err != nil {
		return nil, err
	}
	return te, nil
}

// resnetKeys assigns HF ResnetBlock2D keys from `prefix` into our `r`.
func loadResnetBlock2D(r *resnetBlock2D, w map[string]*tensor.Tensor, prefix string) error {
	if err := copyExactCast(r.N1.Gamma, w, prefix+"norm1.weight"); err != nil {
		return err
	}
	if err := copyExactCast(r.N1.Beta, w, prefix+"norm1.bias"); err != nil {
		return err
	}
	if err := copyExactCast(r.Conv1.Weight, w, prefix+"conv1.weight"); err != nil {
		return err
	}
	if r.Conv1.Bias != nil {
		if err := copyExactCast(r.Conv1.Bias, w, prefix+"conv1.bias"); err != nil {
			return err
		}
	}
	if r.TimeEmbProj != nil {
		if err := copyExactCast(r.TimeEmbProj.W, w, prefix+"time_emb_proj.weight"); err != nil {
			return err
		}
		if err := copyExactCast(r.TimeEmbProj.B, w, prefix+"time_emb_proj.bias"); err != nil {
			return err
		}
	}
	if err := copyExactCast(r.N2.Gamma, w, prefix+"norm2.weight"); err != nil {
		return err
	}
	if err := copyExactCast(r.N2.Beta, w, prefix+"norm2.bias"); err != nil {
		return err
	}
	if err := copyExactCast(r.Conv2.Weight, w, prefix+"conv2.weight"); err != nil {
		return err
	}
	if r.Conv2.Bias != nil {
		if err := copyExactCast(r.Conv2.Bias, w, prefix+"conv2.bias"); err != nil {
			return err
		}
	}
	if r.ConvShortcut != nil {
		if err := copyExactCast(r.ConvShortcut.Weight, w, prefix+"conv_shortcut.weight"); err != nil {
			return err
		}
		if err := copyExactCast(r.ConvShortcut.Bias, w, prefix+"conv_shortcut.bias"); err != nil {
			return err
		}
	}
	return nil
}

func loadTransformer2D(t *transformer2DModel, w map[string]*tensor.Tensor, prefix string) error {
	if err := copyExactCast(t.GN.Gamma, w, prefix+"norm.weight"); err != nil {
		return err
	}
	if err := copyExactCast(t.GN.Beta, w, prefix+"norm.bias"); err != nil {
		return err
	}
	if err := copyExactCast(t.ProjIn.W, w, prefix+"proj_in.weight"); err != nil {
		return err
	}
	if err := copyExactCast(t.ProjIn.B, w, prefix+"proj_in.bias"); err != nil {
		return err
	}
	bp := prefix + "transformer_blocks.0."
	if err := loadBasicTransformerBlock(t.Block, w, bp); err != nil {
		return err
	}
	if err := copyExactCast(t.ProjOut.W, w, prefix+"proj_out.weight"); err != nil {
		return err
	}
	if err := copyExactCast(t.ProjOut.B, w, prefix+"proj_out.bias"); err != nil {
		return err
	}
	return nil
}

func loadBasicTransformerBlock(b *basicTransformerBlock, w map[string]*tensor.Tensor, prefix string) error {
	// norm1, attn1 (self), norm2, attn2 (cross), norm3, ff
	if err := copyExactCast(b.N1.Gamma, w, prefix+"norm1.weight"); err != nil {
		return err
	}
	if err := copyExactCast(b.N1.Beta, w, prefix+"norm1.bias"); err != nil {
		return err
	}
	if err := loadCrossAttention(b.Attn1, w, prefix+"attn1."); err != nil {
		return err
	}
	if err := copyExactCast(b.N2.Gamma, w, prefix+"norm2.weight"); err != nil {
		return err
	}
	if err := copyExactCast(b.N2.Beta, w, prefix+"norm2.bias"); err != nil {
		return err
	}
	if err := loadCrossAttention(b.Attn2, w, prefix+"attn2."); err != nil {
		return err
	}
	if err := copyExactCast(b.N3.Gamma, w, prefix+"norm3.weight"); err != nil {
		return err
	}
	if err := copyExactCast(b.N3.Beta, w, prefix+"norm3.bias"); err != nil {
		return err
	}
	// ff.net.0.proj is the GeGLU's Linear (in, 2*inner). ff.net.2 is the
	// final Linear (inner, in). HF's net is [GEGLU, Dropout, Linear] so
	// indices skip Dropout.
	if err := copyExactCast(b.FF.GEGLU.Proj.W, w, prefix+"ff.net.0.proj.weight"); err != nil {
		return err
	}
	if err := copyExactCast(b.FF.GEGLU.Proj.B, w, prefix+"ff.net.0.proj.bias"); err != nil {
		return err
	}
	if err := copyExactCast(b.FF.Out.W, w, prefix+"ff.net.2.weight"); err != nil {
		return err
	}
	if err := copyExactCast(b.FF.Out.B, w, prefix+"ff.net.2.bias"); err != nil {
		return err
	}
	return nil
}

// loadCrossAttention maps HF's to_q/to_k/to_v/to_out keys. Q/K/V have no
// bias; `to_out.0` carries a bias in SD 1.5 — without copying it the
// generated image quality is visibly poor.
func loadCrossAttention(a *nn.CrossAttention, w map[string]*tensor.Tensor, prefix string) error {
	if err := copyExactCast(a.QProj.W, w, prefix+"to_q.weight"); err != nil {
		return err
	}
	if err := copyExactCast(a.KProj.W, w, prefix+"to_k.weight"); err != nil {
		return err
	}
	if err := copyExactCast(a.VProj.W, w, prefix+"to_v.weight"); err != nil {
		return err
	}
	if err := copyExactCast(a.OProj.W, w, prefix+"to_out.0.weight"); err != nil {
		return err
	}
	if a.OProj.B != nil {
		if err := copyExactCast(a.OProj.B, w, prefix+"to_out.0.bias"); err != nil {
			return err
		}
	}
	return nil
}

// loadSDUNetSD15 loads the SD 1.5 UNet at the requested dtype.
func loadSDUNetSD15(repo string, wdt dtype.DType, dev device.Device) (*SDUNet, error) {
	weights, err := loadComponentSafetensors(repo, "unet", dev)
	if err != nil {
		return nil, err
	}
	cfg := SDUNetSD15()
	cfg.WeightDType = wdt
	m, err := NewSDUNet(cfg, dev)
	if err != nil {
		return nil, err
	}
	// conv_in
	if err := copyExactCast(m.ConvIn.Weight, weights, "conv_in.weight"); err != nil {
		return nil, err
	}
	if err := copyExactCast(m.ConvIn.Bias, weights, "conv_in.bias"); err != nil {
		return nil, err
	}
	// time_embedding
	if err := copyExactCast(m.TimeEmbed1.W, weights, "time_embedding.linear_1.weight"); err != nil {
		return nil, err
	}
	if err := copyExactCast(m.TimeEmbed1.B, weights, "time_embedding.linear_1.bias"); err != nil {
		return nil, err
	}
	if err := copyExactCast(m.TimeEmbed2.W, weights, "time_embedding.linear_2.weight"); err != nil {
		return nil, err
	}
	if err := copyExactCast(m.TimeEmbed2.B, weights, "time_embedding.linear_2.bias"); err != nil {
		return nil, err
	}
	// Down blocks
	for i, db := range m.DownBlocks {
		base := "down_blocks." + strconv.Itoa(i) + "."
		if db.CA != nil {
			for j, r := range db.CA.Resnets {
				if err := loadResnetBlock2D(r, weights, base+"resnets."+strconv.Itoa(j)+"."); err != nil {
					return nil, err
				}
			}
			for j, a := range db.CA.Attns {
				if err := loadTransformer2D(a, weights, base+"attentions."+strconv.Itoa(j)+"."); err != nil {
					return nil, err
				}
			}
			if db.CA.Downsample != nil {
				if err := copyExactCast(db.CA.Downsample.Conv.Weight, weights, base+"downsamplers.0.conv.weight"); err != nil {
					return nil, err
				}
				if err := copyExactCast(db.CA.Downsample.Conv.Bias, weights, base+"downsamplers.0.conv.bias"); err != nil {
					return nil, err
				}
			}
		} else {
			for j, r := range db.Plain.Resnets {
				if err := loadResnetBlock2D(r, weights, base+"resnets."+strconv.Itoa(j)+"."); err != nil {
					return nil, err
				}
			}
			if db.Plain.Downsample != nil {
				if err := copyExactCast(db.Plain.Downsample.Conv.Weight, weights, base+"downsamplers.0.conv.weight"); err != nil {
					return nil, err
				}
				if err := copyExactCast(db.Plain.Downsample.Conv.Bias, weights, base+"downsamplers.0.conv.bias"); err != nil {
					return nil, err
				}
			}
		}
	}
	// Mid block
	if err := loadResnetBlock2D(m.Mid.R0, weights, "mid_block.resnets.0."); err != nil {
		return nil, err
	}
	if err := loadTransformer2D(m.Mid.Attn, weights, "mid_block.attentions.0."); err != nil {
		return nil, err
	}
	if err := loadResnetBlock2D(m.Mid.R1, weights, "mid_block.resnets.1."); err != nil {
		return nil, err
	}
	// Up blocks
	for i, ub := range m.UpBlocks {
		base := "up_blocks." + strconv.Itoa(i) + "."
		if ub.CA != nil {
			for j, r := range ub.CA.Resnets {
				if err := loadResnetBlock2D(r, weights, base+"resnets."+strconv.Itoa(j)+"."); err != nil {
					return nil, err
				}
			}
			for j, a := range ub.CA.Attns {
				if err := loadTransformer2D(a, weights, base+"attentions."+strconv.Itoa(j)+"."); err != nil {
					return nil, err
				}
			}
			if ub.CA.Upsample != nil {
				if err := copyExactCast(ub.CA.Upsample.Conv.Weight, weights, base+"upsamplers.0.conv.weight"); err != nil {
					return nil, err
				}
				if err := copyExactCast(ub.CA.Upsample.Conv.Bias, weights, base+"upsamplers.0.conv.bias"); err != nil {
					return nil, err
				}
			}
		} else {
			for j, r := range ub.Plain.Resnets {
				if err := loadResnetBlock2D(r, weights, base+"resnets."+strconv.Itoa(j)+"."); err != nil {
					return nil, err
				}
			}
			if ub.Plain.Upsample != nil {
				if err := copyExactCast(ub.Plain.Upsample.Conv.Weight, weights, base+"upsamplers.0.conv.weight"); err != nil {
					return nil, err
				}
				if err := copyExactCast(ub.Plain.Upsample.Conv.Bias, weights, base+"upsamplers.0.conv.bias"); err != nil {
					return nil, err
				}
			}
		}
	}
	// conv_norm_out + conv_out
	if err := copyExactCast(m.NormOut.Gamma, weights, "conv_norm_out.weight"); err != nil {
		return nil, err
	}
	if err := copyExactCast(m.NormOut.Beta, weights, "conv_norm_out.bias"); err != nil {
		return nil, err
	}
	if err := copyExactCast(m.ConvOut.Weight, weights, "conv_out.weight"); err != nil {
		return nil, err
	}
	if err := copyExactCast(m.ConvOut.Bias, weights, "conv_out.bias"); err != nil {
		return nil, err
	}
	return m, nil
}

// loadSDVAESD15 loads the SD 1.5 VAE decoder (encoder is skipped — we only
// need decode for txt2img).
func loadSDVAESD15(repo string, wdt dtype.DType, dev device.Device) (*VAEDecoder, error) {
	weights, err := loadComponentSafetensors(repo, "vae", dev)
	if err != nil {
		return nil, err
	}
	v, err := NewVAEDecoder(true, wdt, dev)
	if err != nil {
		return nil, err
	}
	if err := copyExactCast(v.PostQuantConv.Weight, weights, "post_quant_conv.weight"); err != nil {
		return nil, err
	}
	if err := copyExactCast(v.PostQuantConv.Bias, weights, "post_quant_conv.bias"); err != nil {
		return nil, err
	}
	if err := copyExactCast(v.ConvIn.Weight, weights, "decoder.conv_in.weight"); err != nil {
		return nil, err
	}
	if err := copyExactCast(v.ConvIn.Bias, weights, "decoder.conv_in.bias"); err != nil {
		return nil, err
	}
	if err := loadResnetBlock2D(v.Mid.R0, weights, "decoder.mid_block.resnets.0."); err != nil {
		return nil, err
	}
	if err := loadVAEAttention(v.Mid.Attn, weights, "decoder.mid_block.attentions.0."); err != nil {
		return nil, err
	}
	if err := loadResnetBlock2D(v.Mid.R1, weights, "decoder.mid_block.resnets.1."); err != nil {
		return nil, err
	}
	for i, ub := range v.Up {
		base := "decoder.up_blocks." + strconv.Itoa(i) + "."
		for j, r := range ub.Resnets {
			if err := loadResnetBlock2D(r, weights, base+"resnets."+strconv.Itoa(j)+"."); err != nil {
				return nil, err
			}
		}
		if ub.Up != nil {
			if err := copyExactCast(ub.Up.Conv.Weight, weights, base+"upsamplers.0.conv.weight"); err != nil {
				return nil, err
			}
			if err := copyExactCast(ub.Up.Conv.Bias, weights, base+"upsamplers.0.conv.bias"); err != nil {
				return nil, err
			}
		}
	}
	if err := copyExactCast(v.NormOut.Gamma, weights, "decoder.conv_norm_out.weight"); err != nil {
		return nil, err
	}
	if err := copyExactCast(v.NormOut.Beta, weights, "decoder.conv_norm_out.bias"); err != nil {
		return nil, err
	}
	if err := copyExactCast(v.ConvOut.Weight, weights, "decoder.conv_out.weight"); err != nil {
		return nil, err
	}
	if err := copyExactCast(v.ConvOut.Bias, weights, "decoder.conv_out.bias"); err != nil {
		return nil, err
	}
	return v, nil
}

// loadVAEAttention maps HF's older AutoencoderKL attention naming
// (`group_norm`, `to_q`/`to_k`/`to_v`, `to_out.0`).
func loadVAEAttention(a *vaeAttention, w map[string]*tensor.Tensor, prefix string) error {
	if err := copyExactCast(a.GN.Gamma, w, prefix+"group_norm.weight"); err != nil {
		return err
	}
	if err := copyExactCast(a.GN.Beta, w, prefix+"group_norm.bias"); err != nil {
		return err
	}
	// Newer Diffusers ships to_q/k/v + to_out.0; older repos shipped
	// `query`/`key`/`value`/`proj_attn`. Try the new names first; if a
	// tensor is missing, fall back to legacy keys.
	qWKey := prefix + "to_q.weight"
	if _, ok := w[qWKey]; !ok {
		// Legacy → translate.
		legacy := map[string]string{
			prefix + "to_q.weight":     prefix + "query.weight",
			prefix + "to_q.bias":       prefix + "query.bias",
			prefix + "to_k.weight":     prefix + "key.weight",
			prefix + "to_k.bias":       prefix + "key.bias",
			prefix + "to_v.weight":     prefix + "value.weight",
			prefix + "to_v.bias":       prefix + "value.bias",
			prefix + "to_out.0.weight": prefix + "proj_attn.weight",
			prefix + "to_out.0.bias":   prefix + "proj_attn.bias",
		}
		// Lift legacy entries under modern names so the rest of this fn
		// stays simple. Done inline since the `w` map is only used
		// transiently for this load.
		for newKey, oldKey := range legacy {
			if t, ok := w[oldKey]; ok {
				w[newKey] = t
			}
		}
	}
	if err := copyExactCast(a.ToQ.W, w, prefix+"to_q.weight"); err != nil {
		return err
	}
	if err := copyExactCast(a.ToQ.B, w, prefix+"to_q.bias"); err != nil {
		return err
	}
	if err := copyExactCast(a.ToK.W, w, prefix+"to_k.weight"); err != nil {
		return err
	}
	if err := copyExactCast(a.ToK.B, w, prefix+"to_k.bias"); err != nil {
		return err
	}
	if err := copyExactCast(a.ToV.W, w, prefix+"to_v.weight"); err != nil {
		return err
	}
	if err := copyExactCast(a.ToV.B, w, prefix+"to_v.bias"); err != nil {
		return err
	}
	if err := copyExactCast(a.ToOut.W, w, prefix+"to_out.0.weight"); err != nil {
		return err
	}
	if err := copyExactCast(a.ToOut.B, w, prefix+"to_out.0.bias"); err != nil {
		return err
	}
	return nil
}

// silence unused-imports placeholder.
var _ = dtype.F32
