package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// QwenImageVAEConfig is diffusers' AutoencoderKLQwenImage21 config (the
// Wan-2.2-style VAE of Qwen-Image 2.1).
type QwenImageVAEConfig struct {
	BaseDim            int       `json:"base_dim"`
	DecoderBaseDim     *int      `json:"decoder_base_dim"`
	ZDim               int       `json:"z_dim"`
	DimMult            []int     `json:"dim_mult"`
	NumResBlocks       int       `json:"num_res_blocks"`
	TemperalDownsample []bool    `json:"temperal_downsample"`
	IsResidual         *bool     `json:"is_residual"`
	InChannels         int       `json:"in_channels"`
	OutChannels        int       `json:"out_channels"`
	PatchSize          *int      `json:"patch_size"`
	LatentsMean        []float32 `json:"latents_mean"`
	LatentsStd         []float32 `json:"latents_std"`
}

// qiResBlock: x' = conv2(silu(rms(conv1(silu(rms(x)))))) + shortcut(x).
type qiResBlock struct {
	Norm1, Norm2 *tensor.Tensor // (C,) gammas
	Conv1, Conv2 *nn.Conv2d
	Shortcut     *nn.Conv2d // nil when in == out
}

type qiAttn struct {
	Norm      *tensor.Tensor
	QKV, Proj *nn.Conv2d
}

// qiUpBlock is QwenImage21ResidualUpBlock: residual blocks, an optional
// nearest x2 upsample + conv, and the duplicate-up shortcut of its input.
type qiUpBlock struct {
	Res     []*qiResBlock
	UpConv  *nn.Conv2d // nil without upsampling
	in, out int
	factorT int
}

// QwenImageVAE is the decoder of Qwen-Image 2.1's VAE, specialised to one
// frame (the "causal 3D" convolutions are 2-D convolutions there): the
// first chunk of diffusers' frame-by-frame decode.
type QwenImageVAE struct {
	Cfg           QwenImageVAEConfig
	PostQuantConv *nn.Conv2d
	ConvIn        *nn.Conv2d
	MidRes        [2]*qiResBlock
	MidAttn       *qiAttn
	Up            []*qiUpBlock
	NormOut       *tensor.Tensor
	ConvOut       *nn.Conv2d
}

// Scale is the spatial factor from latents to pixels (patching included).
func (v *QwenImageVAE) Scale() int {
	s := 1 << (len(v.Cfg.DimMult) - 1)
	if v.Cfg.PatchSize != nil {
		s *= *v.Cfg.PatchSize
	}
	return s
}

// chanRMS is QwenImage21RMS_norm over channels: F.normalize(x, dim=1) ·
// √C · γ, i.e. x / rms(x) · γ per pixel.
func chanRMS(x, gamma *tensor.Tensor) (*tensor.Tensor, error) {
	p, err := x.Permute(0, 2, 3, 1)
	if err != nil {
		return nil, err
	}
	if p, err = p.Contiguous(); err != nil {
		return nil, err
	}
	if p, err = ops.RMSNorm(p, gamma, 1e-24); err != nil {
		return nil, err
	}
	if p, err = p.Permute(0, 3, 1, 2); err != nil {
		return nil, err
	}
	return p.Contiguous()
}

func (b *qiResBlock) forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	h := x
	var err error
	if b.Shortcut != nil {
		if h, err = b.Shortcut.Forward(x); err != nil {
			return nil, err
		}
	}
	y, err := chanRMS(x, b.Norm1)
	if err != nil {
		return nil, err
	}
	if y, err = ops.SiLU(y); err != nil {
		return nil, err
	}
	if y, err = b.Conv1.Forward(y); err != nil {
		return nil, err
	}
	if y, err = chanRMS(y, b.Norm2); err != nil {
		return nil, err
	}
	if y, err = ops.SiLU(y); err != nil {
		return nil, err
	}
	if y, err = b.Conv2.Forward(y); err != nil {
		return nil, err
	}
	return ops.Add(y, h)
}

// forward is single-head attention over the pixels with 1x1 projections.
func (a *qiAttn) forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	C, H, W := x.Dim(1), x.Dim(2), x.Dim(3)
	n, err := chanRMS(x, a.Norm)
	if err != nil {
		return nil, err
	}
	qkv, err := a.QKV.Forward(n) // (1, 3C, H, W)
	if err != nil {
		return nil, err
	}
	if qkv, err = qkv.Reshape(1, 3*C, H*W); err != nil {
		return nil, err
	}
	if qkv, err = qkv.Permute(0, 2, 1); err != nil { // (1, HW, 3C)
		return nil, err
	}
	if qkv, err = qkv.Contiguous(); err != nil {
		return nil, err
	}
	part := func(i int) (*tensor.Tensor, error) {
		t, err := narrowContig(qkv, 2, i*C, C)
		if err != nil {
			return nil, err
		}
		return t.Reshape(1, 1, H*W, C)
	}
	q, err := part(0)
	if err != nil {
		return nil, err
	}
	k, err := part(1)
	if err != nil {
		return nil, err
	}
	v, err := part(2)
	if err != nil {
		return nil, err
	}
	o, err := nn.SDPA(q, k, v, nil) // (1, 1, HW, C)
	if err != nil {
		return nil, err
	}
	if o, err = o.Reshape(1, H*W, C); err != nil {
		return nil, err
	}
	if o, err = o.Permute(0, 2, 1); err != nil {
		return nil, err
	}
	if o, err = o.Contiguous(); err != nil {
		return nil, err
	}
	if o, err = o.Reshape(1, C, H, W); err != nil {
		return nil, err
	}
	if o, err = a.Proj.Forward(o); err != nil {
		return nil, err
	}
	return ops.Add(o, x)
}

// dupUp is QwenImage21DupUp3D for the first chunk of one frame: channels
// repeated, folded into a 2 x 2 spatial upscale, the last temporal slot kept.
func dupUp(x *tensor.Tensor, in, out, factorT int) (*tensor.Tensor, error) {
	H, W := x.Dim(2), x.Dim(3)
	r := out * factorT * 4 / in
	y, err := x.Reshape(in, 1, H, W)
	if err != nil {
		return nil, err
	}
	if y, err = y.Expand(in, r, H, W); err != nil {
		return nil, err
	}
	if y, err = y.Contiguous(); err != nil {
		return nil, err
	}
	if y, err = y.Reshape(out, factorT, 2, 2, H, W); err != nil {
		return nil, err
	}
	if y, err = y.Narrow(1, factorT-1, 1); err != nil {
		return nil, err
	}
	if y, err = y.Contiguous(); err != nil {
		return nil, err
	}
	if y, err = y.Reshape(out, 2, 2, H, W); err != nil {
		return nil, err
	}
	if y, err = y.Permute(0, 3, 1, 4, 2); err != nil { // (out, H, 2, W, 2)
		return nil, err
	}
	if y, err = y.Contiguous(); err != nil {
		return nil, err
	}
	return y.Reshape(1, out, 2*H, 2*W)
}

func (u *qiUpBlock) forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	y := x
	var err error
	for _, r := range u.Res {
		if y, err = r.forward(y); err != nil {
			return nil, err
		}
	}
	if u.UpConv == nil {
		return y, nil
	}
	if y, err = ops.UpsampleNearest(y, [2]int{2, 2}); err != nil {
		return nil, err
	}
	if y, err = u.UpConv.Forward(y); err != nil {
		return nil, err
	}
	s, err := dupUp(x, u.in, u.out, u.factorT)
	if err != nil {
		return nil, err
	}
	return ops.Add(y, s)
}

// Decode maps latents (1, z_dim, h, w) — already de-normalised with the
// latents mean / std — to an image (1, 3, h·s, w·s) in [-1, 1].
func (v *QwenImageVAE) Decode(z *tensor.Tensor) (*tensor.Tensor, error) {
	x, err := v.PostQuantConv.Forward(z)
	if err != nil {
		return nil, err
	}
	if x, err = v.ConvIn.Forward(x); err != nil {
		return nil, err
	}
	if x, err = v.MidRes[0].forward(x); err != nil {
		return nil, err
	}
	if x, err = v.MidAttn.forward(x); err != nil {
		return nil, err
	}
	if x, err = v.MidRes[1].forward(x); err != nil {
		return nil, err
	}
	for i, u := range v.Up {
		if x, err = u.forward(x); err != nil {
			return nil, fmt.Errorf("qwen-image vae up block %d: %w", i, err)
		}
	}
	if x, err = chanRMS(x, v.NormOut); err != nil {
		return nil, err
	}
	if x, err = ops.SiLU(x); err != nil {
		return nil, err
	}
	if x, err = v.ConvOut.Forward(x); err != nil {
		return nil, err
	}
	if v.Cfg.PatchSize != nil && *v.Cfg.PatchSize > 1 {
		p := *v.Cfg.PatchSize
		C, H, W := x.Dim(1)/(p*p), x.Dim(2), x.Dim(3)
		// _unpatchify: channel c·p² + a·p + b goes to row h·p + b, column w·p + a.
		if x, err = x.Reshape(C, p, p, H, W); err != nil {
			return nil, err
		}
		if x, err = x.Permute(0, 3, 2, 4, 1); err != nil {
			return nil, err
		}
		if x, err = x.Contiguous(); err != nil {
			return nil, err
		}
		if x, err = x.Reshape(1, C, H*p, W*p); err != nil {
			return nil, err
		}
	}
	return ops.Clamp(x, -1, 1)
}

// LoadQwenImageVAE loads the decoder of an AutoencoderKLQwenImage21 folder.
func LoadQwenImageVAE(dir string, dev device.Device) (*QwenImageVAE, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	cfg := QwenImageVAEConfig{BaseDim: 96, ZDim: 64, DimMult: []int{1, 2, 4, 8, 8}, NumResBlocks: 2,
		TemperalDownsample: []bool{false, true, true, true}, InChannels: 4, OutChannels: 4}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("qwen-image vae config: %w", err)
	}
	if cfg.IsResidual != nil && !*cfg.IsResidual {
		return nil, fmt.Errorf("qwen-image vae: is_residual=false not supported")
	}
	base := cfg.BaseDim
	if cfg.DecoderBaseDim != nil {
		base = *cfg.DecoderBaseDim
	}
	w, err := loadCheckpointTensors(dir)
	if err != nil {
		return nil, err
	}
	var lerr error
	conv := func(name string, in, out, k int) *nn.Conv2d {
		if lerr != nil {
			return nil
		}
		var c *nn.Conv2d
		if c, lerr = nn.NewConv2d(in, out, k, k, [2]int{1, 1}, [2]int{k / 2, k / 2}, true, dev); lerr != nil {
			return nil
		}
		if lerr = copyExactCast(c.Weight, w, name+".weight"); lerr == nil {
			lerr = copyExactCast(c.Bias, w, name+".bias")
		}
		return c
	}
	gamma := func(name string, n int) *tensor.Tensor {
		if lerr != nil {
			return nil
		}
		src, ok := w[name]
		if !ok {
			lerr = fmt.Errorf("missing %s", name)
			return nil
		}
		var g *tensor.Tensor
		if g, lerr = ops.Cast(src, dtype.F32); lerr != nil {
			return nil
		}
		if g, lerr = g.Reshape(n); lerr != nil {
			return nil
		}
		g, lerr = g.To(dev)
		return g
	}
	res := func(p string, in, out int) *qiResBlock {
		b := &qiResBlock{Norm1: gamma(p+"norm1.gamma", in), Conv1: conv(p+"conv1", in, out, 3),
			Norm2: gamma(p+"norm2.gamma", out), Conv2: conv(p+"conv2", out, out, 3)}
		if in != out {
			b.Shortcut = conv(p+"conv_shortcut", in, out, 1)
		}
		return b
	}
	mult := cfg.DimMult
	dims := []int{base * mult[len(mult)-1]}
	for i := len(mult) - 1; i >= 0; i-- {
		dims = append(dims, base*mult[i])
	}
	tUp := make([]bool, len(cfg.TemperalDownsample))
	for i, t := range cfg.TemperalDownsample {
		tUp[len(tUp)-1-i] = t
	}
	v := &QwenImageVAE{Cfg: cfg,
		PostQuantConv: conv("post_quant_conv", cfg.ZDim, cfg.ZDim, 1),
		ConvIn:        conv("decoder.conv_in", cfg.ZDim, dims[0], 3)}
	v.MidRes[0] = res("decoder.mid_block.resnets.0.", dims[0], dims[0])
	v.MidAttn = &qiAttn{Norm: gamma("decoder.mid_block.attentions.0.norm.gamma", dims[0]),
		QKV:  conv("decoder.mid_block.attentions.0.to_qkv", dims[0], 3*dims[0], 1),
		Proj: conv("decoder.mid_block.attentions.0.proj", dims[0], dims[0], 1)}
	v.MidRes[1] = res("decoder.mid_block.resnets.1.", dims[0], dims[0])
	for i := range len(dims) - 1 {
		in, out := dims[i], dims[i+1]
		up := i != len(mult)-1
		u := &qiUpBlock{in: in, out: out, factorT: 1}
		if up && tUp[i] {
			u.factorT = 2
		}
		cur := in
		for j := range cfg.NumResBlocks + 1 {
			u.Res = append(u.Res, res(fmt.Sprintf("decoder.up_blocks.%d.resnets.%d.", i, j), cur, out))
			cur = out
		}
		if up {
			u.UpConv = conv(fmt.Sprintf("decoder.up_blocks.%d.upsampler.resample.1", i), out, out, 3)
		}
		v.Up = append(v.Up, u)
	}
	v.NormOut = gamma("decoder.norm_out.gamma", dims[len(dims)-1])
	v.ConvOut = conv("decoder.conv_out", dims[len(dims)-1], cfg.OutChannels, 3)
	if lerr != nil {
		return nil, fmt.Errorf("LoadQwenImageVAE: %w", lerr)
	}
	return v, nil
}
