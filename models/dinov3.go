package models

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"light/device"
	"light/nn"
	"light/ops"
	"light/safetensors"
	"light/tensor"
)

// DINOv3Config is transformers' DINOv3ViTConfig (the fields inference
// uses).
type DINOv3Config struct {
	HiddenSize        int     `json:"hidden_size"`
	NumLayers         int     `json:"num_hidden_layers"`
	NumHeads          int     `json:"num_attention_heads"`
	IntermediateSize  int     `json:"intermediate_size"`
	HiddenAct         string  `json:"hidden_act"`
	LayerNormEps      float64 `json:"layer_norm_eps"`
	PatchSize         int     `json:"patch_size"`
	NumRegisterTokens int     `json:"num_register_tokens"`
	RopeTheta         float64 `json:"rope_theta"`
	UseGatedMLP       bool    `json:"use_gated_mlp"`
}

// DINOv3 is DINOv3ViTModel: patch embedding, a CLS and register tokens,
// pre-norm blocks with 2-D rotary attention on the patch tokens and
// LayerScale, and a final LayerNorm (last_hidden_state).
type DINOv3 struct {
	Cfg       DINOv3Config
	Patch     *nn.Linear     // the stride-P conv as a linear over (c, ky, kx) patches
	Prefix    *tensor.Tensor // (1, 1 + registers, D): CLS then registers
	Layers    []*dinov3Layer
	NormW     *tensor.Tensor
	NormB     *tensor.Tensor
	dev       device.Device
	ropeCache map[[2]int][2]*tensor.Tensor
}

type dinov3Layer struct {
	Norm1W, Norm1B, Norm2W, Norm2B *tensor.Tensor
	Q, K, V, O                     *nn.Linear
	LS1, LS2                       *tensor.Tensor
	Gate, Up, Down                 *nn.Linear // Gate nil without the gated MLP
}

// LoadDINOv3 loads a DINOv3ViTModel folder (config.json, model.safetensors)
// onto dev at F32 (16-bit linear weights under nn.WithWeightStorage16).
func LoadDINOv3(dir string, dev device.Device) (*DINOv3, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	cfg := DINOv3Config{LayerNormEps: 1e-5, RopeTheta: 100, PatchSize: 16, HiddenAct: "gelu"}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("dinov3 config: %w", err)
	}
	paths, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	hf, _, err := safetensors.LoadFiles(paths, device.CPU())
	if err != nil {
		return nil, err
	}
	l := &idolLoader{hf: hf, dev: dev}
	m := &DINOv3{Cfg: cfg, dev: dev, ropeCache: map[[2]int][2]*tensor.Tensor{}}
	// Patch conv (D, 3, P, P) as a linear over flattened patches.
	if w, ok := hf["embeddings.patch_embeddings.weight"]; ok {
		if hf["embeddings.patch_embeddings.weight"], err = w.Reshape(w.Dim(0), w.Dim(1)*w.Dim(2)*w.Dim(3)); err != nil {
			return nil, err
		}
	}
	m.Patch = l.linear("embeddings.patch_embeddings")
	cls, regs := l.param("embeddings.cls_token"), l.param("embeddings.register_tokens")
	if l.err == nil {
		if m.Prefix, err = ops.Concat([]*tensor.Tensor{cls, regs}, 1); err != nil {
			return nil, err
		}
	}
	for i := range cfg.NumLayers {
		p := fmt.Sprintf("layer.%d.", i)
		ly := &dinov3Layer{
			Norm1W: l.param(p + "norm1.weight"), Norm1B: l.param(p + "norm1.bias"),
			Norm2W: l.param(p + "norm2.weight"), Norm2B: l.param(p + "norm2.bias"),
			Q: l.linear(p + "attention.q_proj"), K: l.linear(p + "attention.k_proj"),
			V: l.linear(p + "attention.v_proj"), O: l.linear(p + "attention.o_proj"),
			LS1: l.param(p + "layer_scale1.lambda1"), LS2: l.param(p + "layer_scale2.lambda1"),
			Up: l.linear(p + "mlp.up_proj"), Down: l.linear(p + "mlp.down_proj"),
		}
		if cfg.UseGatedMLP {
			ly.Gate = l.linear(p + "mlp.gate_proj")
		}
		m.Layers = append(m.Layers, ly)
	}
	m.NormW, m.NormB = l.param("norm.weight"), l.param("norm.bias")
	if l.err != nil {
		return nil, fmt.Errorf("dinov3: %w", l.err)
	}
	return m, nil
}

// rope returns cos and sin (1, 1, L, Dh) for 1 + registers prefix tokens
// (identity: cos 1, sin 0) and h × w patches: angles 2π·c·θ^(−4i/Dh) for
// the patch centres c in [−1, 1], height then width, tiled twice.
func (m *DINOv3) rope(h, w int) (*tensor.Tensor, *tensor.Tensor, error) {
	if r, ok := m.ropeCache[[2]int{h, w}]; ok {
		return r[0], r[1], nil
	}
	Dh := m.Cfg.HiddenSize / m.Cfg.NumHeads
	nf := Dh / 4
	inv := make([]float32, nf)
	for i := range inv {
		inv[i] = float32(1 / math.Pow(m.Cfg.RopeTheta, float64(float32(4*i)/float32(Dh))))
	}
	pre := 1 + m.Cfg.NumRegisterTokens
	L := pre + h*w
	cos, sin := make([]float32, L*Dh), make([]float32, L*Dh)
	for t := range pre {
		for d := range Dh {
			cos[t*Dh+d] = 1
		}
	}
	for y := range h {
		for x := range w {
			t := pre + y*w + x
			cy := 2*((float32(y)+0.5)/float32(h)) - 1
			cx := 2*((float32(x)+0.5)/float32(w)) - 1
			for i, f := range inv {
				for k, c := range []float32{cy, cx} {
					a := float64(float32(2*math.Pi) * c * f)
					for _, off := range []int{0, Dh / 2} {
						cos[t*Dh+off+k*nf+i] = float32(math.Cos(a))
						sin[t*Dh+off+k*nf+i] = float32(math.Sin(a))
					}
				}
			}
		}
	}
	ct, err := tensor.New(cos, []int{1, 1, L, Dh}, m.dev)
	if err != nil {
		return nil, nil, err
	}
	st, err := tensor.New(sin, []int{1, 1, L, Dh}, m.dev)
	if err != nil {
		return nil, nil, err
	}
	m.ropeCache[[2]int{h, w}] = [2]*tensor.Tensor{ct, st}
	return ct, st, nil
}

// rotateHalf applies x·cos + rotate_half(x)·sin to (B, H, L, Dh).
func dinov3Rotate(o *iop, x, cos, sin *tensor.Tensor) *tensor.Tensor {
	Dh := x.Dim(3)
	x1, x2 := o.narrow(x, 3, 0, Dh/2), o.narrow(x, 3, Dh/2, Dh/2)
	rh := o.cat(3, o.un(ops.Neg, x2), x1)
	return o.add(o.mul(x, cos), o.mul(rh, sin))
}

// Forward returns last_hidden_state (B, 1 + registers + patches, D) for
// pixel values (B, 3, H, W) (already normalised), H and W multiples of the
// patch size.
func (m *DINOv3) Forward(pixels *tensor.Tensor) (*tensor.Tensor, error) {
	o := &iop{}
	cfg := m.Cfg
	B, C, H, W := pixels.Dim(0), pixels.Dim(1), pixels.Dim(2), pixels.Dim(3)
	P := cfg.PatchSize
	if H%P != 0 || W%P != 0 {
		return nil, fmt.Errorf("dinov3: image %dx%d is not a multiple of the patch size %d", H, W, P)
	}
	h, w := H/P, W/P
	D, NH := cfg.HiddenSize, cfg.NumHeads
	Dh := D / NH
	// Patches in (c, ky, kx) order, as the conv weight flattens.
	x := o.reshape(pixels, B, C, h, P, w, P)
	x = o.reshape(o.perm(x, 0, 2, 4, 1, 3, 5), B, h*w, C*P*P)
	x = o.lin(m.Patch, x)
	prefix := m.Prefix
	if B > 1 {
		ps := make([]*tensor.Tensor, B)
		for i := range ps {
			ps[i] = m.Prefix
		}
		prefix = o.cat(0, ps...)
	}
	x = o.cat(1, prefix, x)
	if o.err != nil {
		return nil, o.err
	}
	cos, sin, err := m.rope(h, w)
	if err != nil {
		return nil, err
	}
	L := x.Dim(1)
	heads := func(t *tensor.Tensor) *tensor.Tensor { return o.perm(o.reshape(t, B, L, NH, Dh), 0, 2, 1, 3) }
	for _, ly := range m.Layers {
		n := o.do(func() (*tensor.Tensor, error) { return ops.LayerNorm(x, ly.Norm1W, ly.Norm1B, cfg.LayerNormEps) })
		q := dinov3Rotate(o, heads(o.lin(ly.Q, n)), cos, sin)
		k := dinov3Rotate(o, heads(o.lin(ly.K, n)), cos, sin)
		v := heads(o.lin(ly.V, n))
		a := o.do(func() (*tensor.Tensor, error) { return nn.SDPA(q, k, v, nil) })
		a = o.lin(ly.O, o.reshape(o.perm(a, 0, 2, 1, 3), B, L, D))
		x = o.add(x, o.mul(a, ly.LS1))
		n = o.do(func() (*tensor.Tensor, error) { return ops.LayerNorm(x, ly.Norm2W, ly.Norm2B, cfg.LayerNormEps) })
		var f *tensor.Tensor
		if ly.Gate != nil {
			f = o.mul(m.act(o, o.lin(ly.Gate, n)), o.lin(ly.Up, n))
		} else {
			f = m.act(o, o.lin(ly.Up, n))
		}
		x = o.add(x, o.mul(o.lin(ly.Down, f), ly.LS2))
	}
	out := o.do(func() (*tensor.Tensor, error) { return ops.LayerNorm(x, m.NormW, m.NormB, cfg.LayerNormEps) })
	return out, o.err
}

func (m *DINOv3) act(o *iop, x *tensor.Tensor) *tensor.Tensor {
	switch m.Cfg.HiddenAct {
	case "silu", "swish":
		return o.un(ops.SiLU, x)
	case "gelu_pytorch_tanh", "gelu_new":
		return o.un(ops.Gelu, x)
	default: // "gelu": exact
		return o.un(ops.GeluErf, x)
	}
}
