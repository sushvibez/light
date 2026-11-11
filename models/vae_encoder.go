package models

import (
	"fmt"
	"strconv"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// VAEEncoder is the AutoencoderKL encoder: conv_in, down blocks (resnets,
// then a stride-2 convolution on the right / bottom padded input, except
// the last block), the mid block, GroupNorm + SiLU, conv_out to the
// latent mean and log-variance, and the optional quant_conv. Mode returns
// the mean (diffusers' latent_dist.mode(), what image-to-image pipelines
// encode references with).
type VAEEncoder struct {
	Cfg       VAEDecoderConfig // same geometry; PostQuantConv here means quant_conv
	ConvIn    *nn.Conv2d
	Down      []*vaeDownBlock
	Mid       *vaeMidBlock
	NormOut   *nn.GroupNorm
	ConvOut   *nn.Conv2d
	QuantConv *nn.Conv2d
}

type vaeDownBlock struct {
	Resnets []*resnetBlock2D
	Down    *nn.Conv2d // nil for the last block
}

// NewVAEEncoderWith builds a zero-initialised encoder; quantConv adds
// quant_conv (use_quant_conv).
func NewVAEEncoderWith(cfg VAEDecoderConfig, quantConv bool, wdt dtype.DType, dev device.Device) (*VAEEncoder, error) {
	n := len(cfg.BlockOut)
	e := &VAEEncoder{Cfg: cfg}
	var err error
	if e.ConvIn, err = nn.NewConv2dUninit(3, cfg.BlockOut[0], 3, 3, [2]int{1, 1}, [2]int{1, 1}, true, wdt, dev); err != nil {
		return nil, err
	}
	prev := cfg.BlockOut[0]
	for i, c := range cfg.BlockOut {
		b := &vaeDownBlock{}
		for j := range cfg.LayersPerBlock {
			in := c
			if j == 0 {
				in = prev
			}
			r, err := newResnetBlock2D(in, c, 0, sdResnetEpsVAE, wdt, dev)
			if err != nil {
				return nil, err
			}
			b.Resnets = append(b.Resnets, r)
		}
		if i < n-1 {
			if b.Down, err = nn.NewConv2dUninit(c, c, 3, 3, [2]int{2, 2}, [2]int{0, 0}, true, wdt, dev); err != nil {
				return nil, err
			}
		}
		e.Down = append(e.Down, b)
		prev = c
	}
	top := cfg.BlockOut[n-1]
	if e.Mid, err = newVAEMidBlock(top, wdt, dev); err != nil {
		return nil, err
	}
	if e.NormOut, err = nn.NewGroupNormWithDType(top, 32, 1e-6, wdt, dev); err != nil {
		return nil, err
	}
	if e.ConvOut, err = nn.NewConv2dUninit(top, 2*cfg.LatentChannels, 3, 3, [2]int{1, 1}, [2]int{1, 1}, true, wdt, dev); err != nil {
		return nil, err
	}
	if quantConv {
		if e.QuantConv, err = nn.NewConv2dUninit(2*cfg.LatentChannels, 2*cfg.LatentChannels, 1, 1, [2]int{1, 1}, [2]int{0, 0}, true, wdt, dev); err != nil {
			return nil, err
		}
	}
	return e, nil
}

// Mode encodes images (B, 3, H, W) in [-1, 1] to the latent mean
// (B, latent, H / 2^(blocks−1), W / 2^(blocks−1)).
func (e *VAEEncoder) Mode(x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := e.ConvIn.Forward(x)
	if err != nil {
		return nil, err
	}
	for _, b := range e.Down {
		for _, r := range b.Resnets {
			if h, err = r.forward(h, nil); err != nil {
				return nil, err
			}
		}
		if b.Down != nil {
			// Downsample2D: pad (0, 1, 0, 1) then a stride-2 3 x 3 convolution.
			if h, err = ops.Pad(h, 3, 0, 1); err != nil {
				return nil, err
			}
			if h, err = ops.Pad(h, 2, 0, 1); err != nil {
				return nil, err
			}
			if h, err = b.Down.Forward(h); err != nil {
				return nil, err
			}
		}
	}
	if h, err = e.Mid.forward(h); err != nil {
		return nil, err
	}
	if h, err = e.NormOut.Forward(h); err != nil {
		return nil, err
	}
	if h, err = autograd.SiLU(h); err != nil {
		return nil, err
	}
	if h, err = e.ConvOut.Forward(h); err != nil {
		return nil, err
	}
	if e.QuantConv != nil {
		if h, err = e.QuantConv.Forward(h); err != nil {
			return nil, err
		}
	}
	mean, err := h.Narrow(1, 0, e.Cfg.LatentChannels)
	if err != nil {
		return nil, err
	}
	return mean.Contiguous()
}

// loadVAEEncoder builds an encoder for cfg and fills it from an
// AutoencoderKL state dict.
func loadVAEEncoder(w map[string]*tensor.Tensor, cfg VAEDecoderConfig, quantConv bool, wdt dtype.DType, dev device.Device) (*VAEEncoder, error) {
	e, err := NewVAEEncoderWith(cfg, quantConv, wdt, dev)
	if err != nil {
		return nil, err
	}
	var lerr error
	cp := func(dst *tensor.Tensor, name string) {
		if lerr == nil {
			lerr = copyExactCast(dst, w, name)
		}
	}
	conv := func(c *nn.Conv2d, name string) {
		cp(c.Weight, name+".weight")
		cp(c.Bias, name+".bias")
	}
	conv(e.ConvIn, "encoder.conv_in")
	for i, b := range e.Down {
		base := "encoder.down_blocks." + strconv.Itoa(i) + "."
		for j, r := range b.Resnets {
			if lerr == nil {
				lerr = loadResnetBlock2D(r, w, base+"resnets."+strconv.Itoa(j)+".")
			}
		}
		if b.Down != nil {
			conv(b.Down, base+"downsamplers.0.conv")
		}
	}
	if lerr == nil {
		lerr = loadResnetBlock2D(e.Mid.R0, w, "encoder.mid_block.resnets.0.")
	}
	if lerr == nil {
		lerr = loadVAEAttention(e.Mid.Attn, w, "encoder.mid_block.attentions.0.")
	}
	if lerr == nil {
		lerr = loadResnetBlock2D(e.Mid.R1, w, "encoder.mid_block.resnets.1.")
	}
	cp(e.NormOut.Gamma, "encoder.conv_norm_out.weight")
	cp(e.NormOut.Beta, "encoder.conv_norm_out.bias")
	conv(e.ConvOut, "encoder.conv_out")
	if e.QuantConv != nil {
		conv(e.QuantConv, "quant_conv")
	}
	if lerr != nil {
		return nil, fmt.Errorf("vae encoder: %w", lerr)
	}
	return e, nil
}
