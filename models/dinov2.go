package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"light/device"
	"light/nn"
	"light/ops"
	"light/safetensors"
	"light/tensor"
)

// DINOv2Config is transformers' Dinov2WithRegistersConfig (the fields
// inference uses); plain DINOv2 is the same with no registers.
type DINOv2Config struct {
	HiddenSize        int     `json:"hidden_size"`
	NumLayers         int     `json:"num_hidden_layers"`
	NumHeads          int     `json:"num_attention_heads"`
	MLPRatio          float64 `json:"mlp_ratio"`
	HiddenAct         string  `json:"hidden_act"`
	LayerNormEps      float64 `json:"layer_norm_eps"`
	ImageSize         int     `json:"image_size"`
	PatchSize         int     `json:"patch_size"`
	NumRegisterTokens int     `json:"num_register_tokens"`
	UseSwiGLU         bool    `json:"use_swiglu_ffn"`
}

// DINOv2 is Dinov2WithRegistersModel (facebook/dinov2-with-registers-*),
// which equals the torch.hub dinov2_vit*14_reg backbones: patch embedding,
// CLS + learned absolute positions, then the register tokens, pre-norm
// blocks with LayerScale and a final LayerNorm.
type DINOv2 struct {
	Cfg    DINOv2Config
	Patch  *nn.Linear     // the stride-P conv as a linear over (c, ky, kx) patches
	CLS    *tensor.Tensor // (1, 1, D)
	Pos    *tensor.Tensor // (1, 1 + grid², D)
	Regs   *tensor.Tensor // (1, registers, D); nil without registers
	Layers []*dinov2Layer
	NormW  *tensor.Tensor
	NormB  *tensor.Tensor
	dev    device.Device
}

type dinov2Layer struct {
	Norm1W, Norm1B, Norm2W, Norm2B *tensor.Tensor
	Q, K, V, O                     *nn.Linear
	LS1, LS2                       *tensor.Tensor
	FC1, FC2                       *nn.Linear
}

// LoadDINOv2 loads a Dinov2(WithRegisters)Model folder (config.json,
// model.safetensors) onto dev at F32 (16-bit linear weights under
// nn.WithWeightStorage16). Both the released key names
// (attention.attention.query, attention.output.dense) and transformers 5's
// (attention.q_proj, attention.o_proj) load.
func LoadDINOv2(dir string, dev device.Device) (*DINOv2, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	cfg := DINOv2Config{LayerNormEps: 1e-6, MLPRatio: 4, PatchSize: 14, HiddenAct: "gelu"}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("dinov2 config: %w", err)
	}
	if cfg.UseSwiGLU {
		return nil, fmt.Errorf("dinov2: the SwiGLU FFN (ViT-g) is not supported")
	}
	paths, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	hf, _, err := safetensors.LoadFiles(paths, device.CPU())
	if err != nil {
		return nil, err
	}
	// Some exports prefix the keys with the model name.
	for k, v := range hf {
		for _, p := range []string{"dinov2_with_registers.", "dinov2."} {
			if len(k) > len(p) && k[:len(p)] == p {
				hf[k[len(p):]] = v
				delete(hf, k)
			}
		}
	}
	l := &idolLoader{hf: hf, dev: dev}
	m := &DINOv2{Cfg: cfg, dev: dev}
	pw := "embeddings.patch_embeddings.projection.weight"
	if w, ok := hf[pw]; ok {
		if hf[pw], err = w.Reshape(w.Dim(0), w.Dim(1)*w.Dim(2)*w.Dim(3)); err != nil {
			return nil, err
		}
	}
	m.Patch = l.linear("embeddings.patch_embeddings.projection")
	m.CLS, m.Pos = l.param("embeddings.cls_token"), l.param("embeddings.position_embeddings")
	if cfg.NumRegisterTokens > 0 {
		m.Regs = l.param("embeddings.register_tokens")
	}
	alt := func(a, b string) string {
		if _, ok := hf[a+".weight"]; ok {
			return a
		}
		return b
	}
	for i := range cfg.NumLayers {
		p := fmt.Sprintf("encoder.layer.%d.", i)
		m.Layers = append(m.Layers, &dinov2Layer{
			Norm1W: l.param(p + "norm1.weight"), Norm1B: l.param(p + "norm1.bias"),
			Norm2W: l.param(p + "norm2.weight"), Norm2B: l.param(p + "norm2.bias"),
			Q:   l.linear(alt(p+"attention.attention.query", p+"attention.q_proj")),
			K:   l.linear(alt(p+"attention.attention.key", p+"attention.k_proj")),
			V:   l.linear(alt(p+"attention.attention.value", p+"attention.v_proj")),
			O:   l.linear(alt(p+"attention.output.dense", p+"attention.o_proj")),
			LS1: l.param(p + "layer_scale1.lambda1"), LS2: l.param(p + "layer_scale2.lambda1"),
			FC1: l.linear(p + "mlp.fc1"), FC2: l.linear(p + "mlp.fc2"),
		})
	}
	m.NormW, m.NormB = l.param("layernorm.weight"), l.param("layernorm.bias")
	if l.err != nil {
		return nil, fmt.Errorf("dinov2: %w", l.err)
	}
	return m, nil
}

// Hidden returns the last block's output before the final LayerNorm (the
// hub model's x_prenorm, transformers' hidden_states[-1]): (B, 1 +
// registers + patches, D) for pixel values (B, 3, H, W), already
// normalised. The image must give the position table's patch grid (518 x
// 518 for the released models): positions are not interpolated.
func (m *DINOv2) Hidden(pixels *tensor.Tensor) (*tensor.Tensor, error) {
	o := &iop{}
	cfg := m.Cfg
	B, C, H, W := pixels.Dim(0), pixels.Dim(1), pixels.Dim(2), pixels.Dim(3)
	P := cfg.PatchSize
	h, w := H/P, W/P
	if H%P != 0 || W%P != 0 || 1+h*w != m.Pos.Dim(1) {
		return nil, fmt.Errorf("dinov2: image %dx%d does not give the %d positions of the model (resize to %d x %d)",
			H, W, m.Pos.Dim(1)-1, cfg.ImageSize, cfg.ImageSize)
	}
	D, NH := cfg.HiddenSize, cfg.NumHeads
	Dh := D / NH
	x := o.reshape(pixels, B, C, h, P, w, P)
	x = o.reshape(o.perm(x, 0, 2, 4, 1, 3, 5), B, h*w, C*P*P)
	x = o.lin(m.Patch, x)
	rep := func(t *tensor.Tensor) *tensor.Tensor {
		if B == 1 {
			return t
		}
		ts := make([]*tensor.Tensor, B)
		for i := range ts {
			ts[i] = t
		}
		return o.cat(0, ts...)
	}
	x = o.add(o.cat(1, rep(m.CLS), x), m.Pos)
	if m.Regs != nil {
		x = o.cat(1, o.narrow(x, 1, 0, 1), rep(m.Regs), o.narrow(x, 1, 1, h*w))
	}
	if o.err != nil {
		return nil, o.err
	}
	L := x.Dim(1)
	heads := func(t *tensor.Tensor) *tensor.Tensor { return o.perm(o.reshape(t, B, L, NH, Dh), 0, 2, 1, 3) }
	for _, ly := range m.Layers {
		n := o.ln(x, ly.Norm1W, ly.Norm1B, cfg.LayerNormEps)
		a := o.do(func() (*tensor.Tensor, error) {
			return nn.SDPA(heads(o.lin(ly.Q, n)), heads(o.lin(ly.K, n)), heads(o.lin(ly.V, n)), nil)
		})
		a = o.lin(ly.O, o.reshape(o.perm(a, 0, 2, 1, 3), B, L, D))
		x = o.add(x, o.mul(a, ly.LS1))
		n = o.ln(x, ly.Norm2W, ly.Norm2B, cfg.LayerNormEps)
		f := o.lin(ly.FC2, m.act(o, o.lin(ly.FC1, n)))
		x = o.add(x, o.mul(f, ly.LS2))
	}
	return x, o.err
}

// Forward returns last_hidden_state: Hidden with the final LayerNorm.
func (m *DINOv2) Forward(pixels *tensor.Tensor) (*tensor.Tensor, error) {
	x, err := m.Hidden(pixels)
	if err != nil {
		return nil, err
	}
	return ops.LayerNorm(x, m.NormW, m.NormB, m.Cfg.LayerNormEps)
}

func (m *DINOv2) act(o *iop, x *tensor.Tensor) *tensor.Tensor {
	switch m.Cfg.HiddenAct {
	case "gelu_pytorch_tanh", "gelu_new":
		return o.un(ops.Gelu, x)
	case "silu", "swish":
		return o.un(ops.SiLU, x)
	default: // "gelu": exact
		return o.un(ops.GeluErf, x)
	}
}
