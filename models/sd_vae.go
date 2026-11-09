package models

import (
	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// vaeMidBlock — UNetMidBlock2D in HF Diffusers when used as the VAE
// bottleneck: ResnetBlock → vaeAttention → ResnetBlock. No time conditioning.
type vaeMidBlock struct {
	R0   *resnetBlock2D
	Attn *vaeAttention
	R1   *resnetBlock2D
}

func newVAEMidBlock(channels int, wdt dtype.DType, dev device.Device) (*vaeMidBlock, error) {
	r0, err := newResnetBlock2D(channels, channels, 0, sdResnetEpsVAE, wdt, dev)
	if err != nil {
		return nil, err
	}
	a, err := newVAEAttention(channels, wdt, dev)
	if err != nil {
		return nil, err
	}
	r1, err := newResnetBlock2D(channels, channels, 0, sdResnetEpsVAE, wdt, dev)
	if err != nil {
		return nil, err
	}
	return &vaeMidBlock{R0: r0, Attn: a, R1: r1}, nil
}

func (m *vaeMidBlock) forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := m.R0.forward(x, nil)
	if err != nil {
		return nil, err
	}
	h, err = m.Attn.forward(h)
	if err != nil {
		return nil, err
	}
	return m.R1.forward(h, nil)
}

func (m *vaeMidBlock) parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, m.R0.parameters()...)
	ps = append(ps, m.Attn.parameters()...)
	ps = append(ps, m.R1.parameters()...)
	return ps
}

// upDecoderBlock2D — N ResnetBlocks (first does in→out, rest do out→out)
// followed by an optional Upsample2D.
type upDecoderBlock2D struct {
	Resnets []*resnetBlock2D
	Up      *upsample2D // nil for the final block
}

func newUpDecoderBlock2D(in, out, numResnets int, useUpsample bool, wdt dtype.DType, dev device.Device) (*upDecoderBlock2D, error) {
	rs := make([]*resnetBlock2D, numResnets)
	for i := 0; i < numResnets; i++ {
		ic := out
		if i == 0 {
			ic = in
		}
		r, err := newResnetBlock2D(ic, out, 0, sdResnetEpsVAE, wdt, dev)
		if err != nil {
			return nil, err
		}
		rs[i] = r
	}
	var up *upsample2D
	if useUpsample {
		var err error
		up, err = newUpsample2D(out, wdt, dev)
		if err != nil {
			return nil, err
		}
	}
	return &upDecoderBlock2D{Resnets: rs, Up: up}, nil
}

func (b *upDecoderBlock2D) forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	h := x
	var err error
	for _, r := range b.Resnets {
		h, err = r.forward(h, nil)
		if err != nil {
			return nil, err
		}
	}
	if b.Up != nil {
		h, err = b.Up.forward(h)
		if err != nil {
			return nil, err
		}
	}
	return h, nil
}

func (b *upDecoderBlock2D) parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, r := range b.Resnets {
		ps = append(ps, r.parameters()...)
	}
	if b.Up != nil {
		ps = append(ps, b.Up.parameters()...)
	}
	return ps
}

// VAEDecoder is the AutoencoderKL decoder for Stable Diffusion 1.x. Takes a
// (B, 4, H, W) latent and produces a (B, 3, H*8, W*8) RGB image in [-1, 1].
//
// Channel schedule (HF Diffusers default):
//
//	conv_in:  4 → 512
//	mid:      512 ↔ 512 (resnet, attn, resnet)
//	up[0]:    512 → 512 + upsample (×2)
//	up[1]:    512 → 512 + upsample (×2)
//	up[2]:    512 → 256 + upsample (×2)  ← total 8× upsample
//	up[3]:    256 → 128                  (no upsample at the last block)
//	conv_out: 128 → 3
type VAEDecoder struct {
	PostQuantConv *nn.Conv2d
	ConvIn        *nn.Conv2d
	Mid           *vaeMidBlock
	Up            []*upDecoderBlock2D
	NormOut       *nn.GroupNorm
	ConvOut       *nn.Conv2d
}

// VAEDecoderConfig is the decoder half of diffusers' AutoencoderKL config.
type VAEDecoderConfig struct {
	LatentChannels int   // 4 (SD 1.x / 2.x / XL), 16 (Flux, SD3)
	BlockOut       []int // block_out_channels in encoder order, e.g. 128, 256, 512, 512
	LayersPerBlock int   // the decoder runs LayersPerBlock+1 resnets per block
	PostQuantConv  bool  // use_post_quant_conv (false for Flux / SD3)
}

// VAEDecoderSD15 is the SD 1.x decoder (also SD 2.x and SDXL's layout).
func VAEDecoderSD15() VAEDecoderConfig {
	return VAEDecoderConfig{LatentChannels: 4, BlockOut: []int{128, 256, 512, 512}, LayersPerBlock: 2, PostQuantConv: true}
}

// NewVAEDecoder constructs the decoder with the SD 1.5 channel schedule.
// `wdt` is the storage dtype for Conv2d weights (F32 default; BF16 for
// half-memory inference). Loaders are expected to overwrite values, so all
// allocations are zero-init. `skipInit` is currently always honoured.
func NewVAEDecoder(skipInit bool, wdt dtype.DType, dev device.Device) (*VAEDecoder, error) {
	return NewVAEDecoderWith(VAEDecoderSD15(), wdt, dev)
}

// NewVAEDecoderWith constructs a zero-initialised decoder for cfg.
func NewVAEDecoderWith(cfg VAEDecoderConfig, wdt dtype.DType, dev device.Device) (*VAEDecoder, error) {
	n := len(cfg.BlockOut)
	top := cfg.BlockOut[n-1]
	var postQ *nn.Conv2d
	var err error
	if cfg.PostQuantConv {
		if postQ, err = nn.NewConv2dUninit(cfg.LatentChannels, cfg.LatentChannels, 1, 1, [2]int{1, 1}, [2]int{0, 0}, true, wdt, dev); err != nil {
			return nil, err
		}
	}
	convIn, err := nn.NewConv2dUninit(cfg.LatentChannels, top, 3, 3, [2]int{1, 1}, [2]int{1, 1}, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	mid, err := newVAEMidBlock(top, wdt, dev)
	if err != nil {
		return nil, err
	}
	ups := make([]*upDecoderBlock2D, n)
	prev := top
	for i := range n {
		c := cfg.BlockOut[n-1-i] // reversed
		b, err := newUpDecoderBlock2D(prev, c, cfg.LayersPerBlock+1, i < n-1, wdt, dev)
		if err != nil {
			return nil, err
		}
		ups[i] = b
		prev = c
	}
	normOut, err := nn.NewGroupNormWithDType(cfg.BlockOut[0], 32, 1e-6, wdt, dev)
	if err != nil {
		return nil, err
	}
	convOut, err := nn.NewConv2dUninit(cfg.BlockOut[0], 3, 3, 3, [2]int{1, 1}, [2]int{1, 1}, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	return &VAEDecoder{
		PostQuantConv: postQ, ConvIn: convIn, Mid: mid,
		Up: ups, NormOut: normOut, ConvOut: convOut,
	}, nil
}

// Decode runs the VAE decoder. SD applies a 0.18215 inverse scaling factor
// at the input boundary (HF Diffusers calls it `scaling_factor`); the
// caller should divide the latent by 0.18215 before passing it in, or use
// `DecodeWithScaling` which folds that division in.
func (v *VAEDecoder) Decode(latent *tensor.Tensor) (*tensor.Tensor, error) {
	h := latent
	var err error
	if v.PostQuantConv != nil {
		if h, err = v.PostQuantConv.Forward(latent); err != nil {
			return nil, err
		}
	}
	h, err = v.ConvIn.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = v.Mid.forward(h)
	if err != nil {
		return nil, err
	}
	for _, b := range v.Up {
		h, err = b.forward(h)
		if err != nil {
			return nil, err
		}
	}
	h, err = v.NormOut.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = autograd.SiLU(h)
	if err != nil {
		return nil, err
	}
	return v.ConvOut.Forward(h)
}

// DecodeWithScaling divides the latent by `scalingFactor` (0.18215 for SD
// 1.x) before running the decoder.
func (v *VAEDecoder) DecodeWithScaling(latent *tensor.Tensor, scalingFactor float32) (*tensor.Tensor, error) {
	if scalingFactor == 0 || scalingFactor == 1 {
		return v.Decode(latent)
	}
	inv, err := tensor.New([]float32{1.0 / scalingFactor}, []int{1}, latent.Device())
	if err != nil {
		return nil, err
	}
	if latent.DType() != dtype.F32 {
		inv, err = ops.Cast(inv, latent.DType())
		if err != nil {
			return nil, err
		}
	}
	scaled, err := autograd.Mul(latent, inv)
	if err != nil {
		return nil, err
	}
	return v.Decode(scaled)
}

// Parameters returns every learnable tensor.
func (v *VAEDecoder) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	if v.PostQuantConv != nil {
		ps = append(ps, v.PostQuantConv.Parameters()...)
	}
	ps = append(ps, v.ConvIn.Parameters()...)
	ps = append(ps, v.Mid.parameters()...)
	for _, b := range v.Up {
		ps = append(ps, b.parameters()...)
	}
	ps = append(ps, v.NormOut.Parameters()...)
	ps = append(ps, v.ConvOut.Parameters()...)
	return ps
}

// ParamCount returns the total trainable element count.
func (v *VAEDecoder) ParamCount() int {
	total := 0
	for _, p := range v.Parameters() {
		total += p.NumElements()
	}
	return total
}

// silence unused-import for device when only used downstream by callers.
var _ = device.CPU
