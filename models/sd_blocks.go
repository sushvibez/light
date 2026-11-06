package models

import (
	"fmt"
	"math"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// resnetBlock2D is the residual building block shared by Stable Diffusion's
// UNet and VAE. When TimeEmbDim > 0, a (B, TimeEmbDim) time embedding is
// projected to (B, OutChannels) and added per-channel after Conv1.
type resnetBlock2D struct {
	N1           *nn.GroupNorm
	Conv1        *nn.Conv2d
	TimeEmbProj  *nn.Linear // nil when TimeEmbDim == 0 (VAE)
	N2           *nn.GroupNorm
	Conv2        *nn.Conv2d
	ConvShortcut *nn.Conv2d // nil when in == out
	in, out      int
}

// newResnetBlock2D constructs an SD-style residual block. `eps` is the
// GroupNorm epsilon: HF SD UNet uses 1e-5, but the VAE uses 1e-6 — and
// silently using the wrong value shifts colors in the final image.
func newResnetBlock2D(in, out, tembDim int, eps float64, wdt dtype.DType, dev device.Device) (*resnetBlock2D, error) {
	const groups = 32
	n1, err := nn.NewGroupNormWithDType(in, groups, eps, wdt, dev)
	if err != nil {
		return nil, err
	}
	c1, err := nn.NewConv2dUninit(in, out, 3, 3, [2]int{1, 1}, [2]int{1, 1}, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	var teProj *nn.Linear
	if tembDim > 0 {
		teProj, err = nn.NewLinearUninit(tembDim, out, true, wdt, dev)
		if err != nil {
			return nil, err
		}
	}
	n2, err := nn.NewGroupNormWithDType(out, groups, eps, wdt, dev)
	if err != nil {
		return nil, err
	}
	c2, err := nn.NewConv2dUninit(out, out, 3, 3, [2]int{1, 1}, [2]int{1, 1}, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	var sc *nn.Conv2d
	if in != out {
		sc, err = nn.NewConv2dUninit(in, out, 1, 1, [2]int{1, 1}, [2]int{0, 0}, true, wdt, dev)
		if err != nil {
			return nil, err
		}
	}
	return &resnetBlock2D{N1: n1, Conv1: c1, TimeEmbProj: teProj, N2: n2, Conv2: c2, ConvShortcut: sc, in: in, out: out}, nil
}

// SD eps constants — UNet's ResnetBlock2D uses 1e-5 (HF
// `UNet2DConditionModel.norm_eps` default), VAE's uses 1e-6 (HF
// `Encoder/Decoder` hardcodes `resnet_eps=1e-6`).
const (
	sdResnetEpsUNet = 1e-5
	sdResnetEpsVAE  = 1e-6
)

// forward runs the block. `temb` may be nil when the block has no time
// embedding (VAE case).
func (r *resnetBlock2D) forward(x, temb *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := r.N1.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = autograd.SiLU(h)
	if err != nil {
		return nil, err
	}
	h, err = r.Conv1.Forward(h)
	if err != nil {
		return nil, err
	}
	if r.TimeEmbProj != nil {
		// temb: (B, TEDim) → SiLU → Linear → (B, out). Add per-channel
		// (broadcast over H, W) by reshape to (B, out, 1, 1).
		t, err := autograd.SiLU(temb)
		if err != nil {
			return nil, err
		}
		t, err = r.TimeEmbProj.Forward(t)
		if err != nil {
			return nil, err
		}
		t, err = autograd.Reshape(t, t.Dim(0), t.Dim(1), 1, 1)
		if err != nil {
			return nil, err
		}
		h, err = autograd.Add(h, t)
		if err != nil {
			return nil, err
		}
	}
	h, err = r.N2.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = autograd.SiLU(h)
	if err != nil {
		return nil, err
	}
	h, err = r.Conv2.Forward(h)
	if err != nil {
		return nil, err
	}
	skip := x
	if r.ConvShortcut != nil {
		skip, err = r.ConvShortcut.Forward(x)
		if err != nil {
			return nil, err
		}
	}
	return autograd.Add(h, skip)
}

func (r *resnetBlock2D) parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, r.N1.Parameters()...)
	ps = append(ps, r.Conv1.Parameters()...)
	if r.TimeEmbProj != nil {
		ps = append(ps, r.TimeEmbProj.Parameters()...)
	}
	ps = append(ps, r.N2.Parameters()...)
	ps = append(ps, r.Conv2.Parameters()...)
	if r.ConvShortcut != nil {
		ps = append(ps, r.ConvShortcut.Parameters()...)
	}
	return ps
}

// vaeAttention is the single-head self-attention block used by SD's VAE
// mid-block. Operates on (B, C, H, W) by reshaping to (B, H*W, C); norm,
// Q/K/V, scaled-dot-product, output proj, then reshape back + residual.
type vaeAttention struct {
	GN    *nn.GroupNorm
	ToQ   *nn.Linear
	ToK   *nn.Linear
	ToV   *nn.Linear
	ToOut *nn.Linear
	scale *tensor.Tensor
	c     int
}

func newVAEAttention(channels int, wdt dtype.DType, dev device.Device) (*vaeAttention, error) {
	gn, err := nn.NewGroupNormWithDType(channels, 32, 1e-5, wdt, dev)
	if err != nil {
		return nil, err
	}
	q, err := nn.NewLinearUninit(channels, channels, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	k, err := nn.NewLinearUninit(channels, channels, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	v, err := nn.NewLinearUninit(channels, channels, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	o, err := nn.NewLinearUninit(channels, channels, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	scaleVal := float32(1.0 / math.Sqrt(float64(channels)))
	scaleT, err := tensor.New([]float32{scaleVal}, []int{1}, dev)
	if err != nil {
		return nil, err
	}
	if wdt != dtype.F32 {
		scaleT, err = ops.Cast(scaleT, wdt)
		if err != nil {
			return nil, err
		}
	}
	return &vaeAttention{GN: gn, ToQ: q, ToK: k, ToV: v, ToOut: o, scale: scaleT, c: channels}, nil
}

func (a *vaeAttention) forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	residual := x
	B, C, H, W := x.Dim(0), x.Dim(1), x.Dim(2), x.Dim(3)
	if C != a.c {
		return nil, fmt.Errorf("vaeAttention: input channels %d != %d", C, a.c)
	}
	h, err := a.GN.Forward(x)
	if err != nil {
		return nil, err
	}
	// (B, C, H, W) → (B, C, H*W) → (B, H*W, C)
	h, err = autograd.Reshape(h, B, C, H*W)
	if err != nil {
		return nil, err
	}
	h, err = autograd.Transpose(h, 1, 2)
	if err != nil {
		return nil, err
	}
	h, err = autograd.Contiguous(h)
	if err != nil {
		return nil, err
	}
	q, err := a.ToQ.Forward(h)
	if err != nil {
		return nil, err
	}
	k, err := a.ToK.Forward(h)
	if err != nil {
		return nil, err
	}
	v, err := a.ToV.Forward(h)
	if err != nil {
		return nil, err
	}
	kT, err := autograd.Transpose(k, 1, 2)
	if err != nil {
		return nil, err
	}
	kTC, err := autograd.Contiguous(kT)
	if err != nil {
		return nil, err
	}
	scores, err := autograd.MatMul(q, kTC) // (B, H*W, H*W)
	if err != nil {
		return nil, err
	}
	scores, err = autograd.Mul(scores, a.scale)
	if err != nil {
		return nil, err
	}
	attn, err := autograd.Softmax(scores, -1)
	if err != nil {
		return nil, err
	}
	o, err := autograd.MatMul(attn, v)
	if err != nil {
		return nil, err
	}
	o, err = a.ToOut.Forward(o)
	if err != nil {
		return nil, err
	}
	// (B, H*W, C) → (B, C, H*W) → (B, C, H, W)
	o, err = autograd.Transpose(o, 1, 2)
	if err != nil {
		return nil, err
	}
	o, err = autograd.Contiguous(o)
	if err != nil {
		return nil, err
	}
	o, err = autograd.Reshape(o, B, C, H, W)
	if err != nil {
		return nil, err
	}
	return autograd.Add(o, residual)
}

func (a *vaeAttention) parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, a.GN.Parameters()...)
	ps = append(ps, a.ToQ.Parameters()...)
	ps = append(ps, a.ToK.Parameters()...)
	ps = append(ps, a.ToV.Parameters()...)
	ps = append(ps, a.ToOut.Parameters()...)
	return ps
}

// upsample2D does nearest-neighbour 2× upsample then a 3×3 conv (stride 1,
// pad 1). HF's Upsample2D layer.
type upsample2D struct {
	Conv *nn.Conv2d
}

func newUpsample2D(channels int, wdt dtype.DType, dev device.Device) (*upsample2D, error) {
	c, err := nn.NewConv2dUninit(channels, channels, 3, 3, [2]int{1, 1}, [2]int{1, 1}, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	return &upsample2D{Conv: c}, nil
}

func (u *upsample2D) forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := ops.UpsampleNearest(x, [2]int{2, 2})
	if err != nil {
		return nil, err
	}
	return u.Conv.Forward(h)
}

func (u *upsample2D) parameters() []*tensor.Tensor { return u.Conv.Parameters() }

// downsample2D is a strided 3×3 conv (HF's Downsample2D with use_conv=True).
type downsample2D struct {
	Conv *nn.Conv2d
}

func newDownsample2D(channels int, wdt dtype.DType, dev device.Device) (*downsample2D, error) {
	c, err := nn.NewConv2dUninit(channels, channels, 3, 3, [2]int{2, 2}, [2]int{1, 1}, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	return &downsample2D{Conv: c}, nil
}

func (d *downsample2D) forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	return d.Conv.Forward(x)
}

func (d *downsample2D) parameters() []*tensor.Tensor { return d.Conv.Parameters() }
