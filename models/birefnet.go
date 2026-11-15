package models

import (
	"fmt"
	"math"

	"light/device"
	"light/ops"
	"light/tensor"
)

// BiRefNet (ZhengPeng7/BiRefNet as idol-sdk ships it in
// idol/models/birefnet.py, speridlabs/idol's birefnet.safetensors): a
// dichotomous segmentation net that predicts a foreground matte. The Swin
// backbone runs on the image and on a half-resolution copy (mul_scl_ipt
// 'cat': each stage's features concatenated with the upsampled
// half-resolution ones); the three shallower stages are upsampled onto the
// deepest and concatenated (cxt) into the squeeze block; the decoder climbs
// back up through ASPP-deformable blocks gated by gdt attention, adding
// lateral 1 x 1 projections of the encoder features and, at every level,
// convolutions of the input image cut into patches of that level's size.
// All interpolation is bilinear with align_corners; BatchNorms (inference)
// are folded into the preceding convolutions at load time.
type BiRefNet struct {
	BB      *Swin
	Squeeze *birefDecBlock
	Dec     birefDecoder
	// InputSize is the (H, W) RemoveBackground resizes to: 1024 x 1024.
	InputSize [2]int
	dev       device.Device
}

// birefConv is a convolution with any BatchNorm after it folded in.
type birefConv struct {
	W, B        *tensor.Tensor
	Stride, Pad int
}

func (c *birefConv) forward(o *iop, x *tensor.Tensor) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) {
		return ops.Conv2d(x, c.W, c.B, [2]int{c.Stride, c.Stride}, [2]int{c.Pad, c.Pad})
	})
}

// birefDeform is _DeformableConv2d (+ its BatchNorm, folded): offsets and a
// 2·sigmoid modulator from plain convolutions, then torchvision's
// deform_conv2d with the regular convolution's weight.
type birefDeform struct {
	Offset, Mod *birefConv
	Regular     *birefConv
}

func (d *birefDeform) forward(o *iop, x *tensor.Tensor) *tensor.Tensor {
	off := d.Offset.forward(o, x)
	m := o.affine(o.un(ops.Sigmoid, d.Mod.forward(o, x)), 2, 0)
	return o.do(func() (*tensor.Tensor, error) {
		p := d.Regular.Pad
		return ops.DeformConv2d(x, off, m, d.Regular.W, d.Regular.B, ops.DeformConv2dOptions{Padding: [2]int{p, p}})
	})
}

// birefASPP is _ASPPDeformable: a 1 x 1 and 1 / 3 / 7 deformable branches,
// a global-average branch broadcast back, concatenated and projected.
type birefASPP struct {
	ASPP1   *birefDeform
	Deforms []*birefDeform
	Pool    *birefConv
	Conv1   *birefConv
}

func (a *birefASPP) forward(o *iop, x *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	H, W := x.Dim(2), x.Dim(3)
	parts := []*tensor.Tensor{o.un(ops.Relu, a.ASPP1.forward(o, x))}
	for _, d := range a.Deforms {
		parts = append(parts, o.un(ops.Relu, d.forward(o, x)))
	}
	g := o.do(func() (*tensor.Tensor, error) { return ops.AdaptiveAvgPool2d(x, 1, 1) })
	g = o.un(ops.Relu, a.Pool.forward(o, g))
	parts = append(parts, birefUp(o, g, H, W))
	return o.un(ops.Relu, a.Conv1.forward(o, o.cat(1, parts...)))
}

// birefDecBlock is _BasicDecBlk: conv + BN + ReLU, ASPP, conv + BN.
type birefDecBlock struct {
	In   *birefConv
	ASPP *birefASPP
	Out  *birefConv
}

func (b *birefDecBlock) forward(o *iop, x *tensor.Tensor) *tensor.Tensor {
	x = o.un(ops.Relu, b.In.forward(o, x))
	return b.Out.forward(o, b.ASPP.forward(o, x))
}

// birefSimple is _SimpleConvs: two 3 x 3 convolutions.
type birefSimple struct{ C1, C2 *birefConv }

func (s *birefSimple) forward(o *iop, x *tensor.Tensor) *tensor.Tensor {
	return s.C2.forward(o, s.C1.forward(o, x))
}

// birefGate is a gdt branch: p · sigmoid(conv1x1(ReLU(BN(conv3x3(p))))).
type birefGate struct{ Conv, Attn *birefConv }

func (g *birefGate) forward(o *iop, p *tensor.Tensor) *tensor.Tensor {
	a := g.Attn.forward(o, o.un(ops.Relu, g.Conv.forward(o, p)))
	return o.mul(p, o.un(ops.Sigmoid, a))
}

type birefDecoder struct {
	Ipt     [5]*birefSimple   // ipt_blk1 .. ipt_blk5
	Blocks  [4]*birefDecBlock // decoder_block1 .. 4
	Lateral [3]*birefConv     // lateral_block2 .. 4
	Gates   [3]*birefGate     // gdt 2 .. 4
	Out     *birefConv
}

// birefUp is F.interpolate(mode="bilinear", align_corners=True) to (h, w).
func birefUp(o *iop, x *tensor.Tensor, h, w int) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	if x.Dim(2) == h && x.Dim(3) == w {
		return x
	}
	return o.do(func() (*tensor.Tensor, error) { return ops.UpsampleBilinear2d(x, h, w, true) })
}

// birefPatches is _image2patches: the image (B, c, H, W) cut into the
// (hg x wg) grid of (h, w) patches stacked on the channels, (B, c·hg·wg, h,
// w).
func birefPatches(o *iop, img *tensor.Tensor, h, w int) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	B, c, H, W := img.Dim(0), img.Dim(1), img.Dim(2), img.Dim(3)
	hg, wg := H/h, W/w
	return o.reshape(o.perm(o.reshape(img, B, c, hg, h, wg, w), 0, 1, 2, 4, 3, 5), B, c*hg*wg, h, w)
}

func (d *birefDecoder) forward(o *iop, x, x1, x2, x3, x4 *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	withPatches := func(t *tensor.Tensor, ipt *birefSimple) *tensor.Tensor {
		return o.cat(1, t, ipt.forward(o, birefPatches(o, x, t.Dim(2), t.Dim(3))))
	}
	x4 = withPatches(x4, d.Ipt[4])
	p4 := d.Gates[2].forward(o, d.Blocks[3].forward(o, x4))
	p3 := o.add(birefUp(o, p4, x3.Dim(2), x3.Dim(3)), d.Lateral[2].forward(o, x3))
	p3 = d.Gates[1].forward(o, d.Blocks[2].forward(o, withPatches(p3, d.Ipt[3])))
	p2 := o.add(birefUp(o, p3, x2.Dim(2), x2.Dim(3)), d.Lateral[1].forward(o, x2))
	p2 = d.Gates[0].forward(o, d.Blocks[1].forward(o, withPatches(p2, d.Ipt[2])))
	p1 := o.add(birefUp(o, p2, x1.Dim(2), x1.Dim(3)), d.Lateral[0].forward(o, x1))
	p1 = d.Blocks[0].forward(o, withPatches(p1, d.Ipt[1]))
	p1 = birefUp(o, p1, x.Dim(2), x.Dim(3))
	return d.Out.forward(o, withPatches(p1, d.Ipt[0]))
}

// Forward returns the foreground probability (B, 1, H, W) for
// ImageNet-normalised images x (B, 3, H, W); H and W must be multiples of
// 64 (the backbone at half resolution down to 1/32).
func (m *BiRefNet) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	o := &iop{}
	H, W := x.Dim(2), x.Dim(3)
	if H%64 != 0 || W%64 != 0 {
		return nil, fmt.Errorf("birefnet: input %d x %d is not a multiple of 64", H, W)
	}
	full, err := m.BB.Forward(x)
	if err != nil {
		return nil, err
	}
	half, err := m.BB.Forward(birefUp(o, x, H/2, W/2))
	if err != nil {
		return nil, err
	}
	xs := make([]*tensor.Tensor, len(full))
	for i, f := range full {
		xs[i] = o.cat(1, f, birefUp(o, half[i], f.Dim(2), f.Dim(3)))
	}
	if o.err != nil {
		return nil, o.err
	}
	x1, x2, x3, x4 := xs[0], xs[1], xs[2], xs[3]
	h4, w4 := x4.Dim(2), x4.Dim(3)
	x4 = o.cat(1, birefUp(o, x1, h4, w4), birefUp(o, x2, h4, w4), birefUp(o, x3, h4, w4), x4)
	x4 = m.Squeeze.forward(o, x4)
	return o.un(ops.Sigmoid, m.Dec.forward(o, x, x1, x2, x3, x4)), o.err
}

// RemoveBackground is BiRefNet.remove_background's matte: rgb (3, h, w)
// planes in [0, 1] are resized (bilinear, align_corners) to InputSize,
// ImageNet-normalised and segmented, and the map resized back to (h, w)
// and quantised as uint8(clamp(a, 0, 1)·255) (truncating), row-major.
func (m *BiRefNet) RemoveBackground(rgb []float32, w, h int) ([]uint8, error) {
	if len(rgb) != 3*w*h {
		return nil, fmt.Errorf("birefnet: %d values for a %d x %d RGB image", len(rgb), w, h)
	}
	o := &iop{}
	x := o.do(func() (*tensor.Tensor, error) { return tensor.New(rgb, []int{1, 3, h, w}, m.dev) })
	x = o.do(func() (*tensor.Tensor, error) { return ops.UpsampleBilinear2d(x, m.InputSize[0], m.InputSize[1], true) })
	mean := o.do(func() (*tensor.Tensor, error) { return tensor.New(imagenetMeanStd[0][:], []int{1, 3, 1, 1}, m.dev) })
	std := o.do(func() (*tensor.Tensor, error) { return tensor.New(imagenetMeanStd[1][:], []int{1, 3, 1, 1}, m.dev) })
	x = o.sub(x, mean)
	x = o.do(func() (*tensor.Tensor, error) { return ops.Div(x, std) })
	if o.err != nil {
		return nil, o.err
	}
	a, err := m.Forward(x)
	if err != nil {
		return nil, err
	}
	a = o.do(func() (*tensor.Tensor, error) { return ops.UpsampleBilinear2d(a, h, w, true) })
	a = o.do(func() (*tensor.Tensor, error) { return a.To(device.CPU()) })
	if o.err != nil {
		return nil, o.err
	}
	v := tensor.MustSlice[float32](a)
	out := make([]uint8, len(v))
	for i, f := range v {
		out[i] = uint8(math.Min(math.Max(float64(f), 0), 1) * 255)
	}
	return out, nil
}

var imagenetMeanStd = [2][3]float32{{0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}}

// --- loading -----------------------------------------------------------------

// conv loads a Conv2d (stride 1, padding kernel/2; patch: stride = kernel,
// no padding) at F32, folding the BatchNorm named bn (eps 1e-5) into it
// when bn is set.
func (l *idolLoader) conv(name, bn string, patch bool) *birefConv {
	if l.err != nil {
		return nil
	}
	wt, ok := l.hf[name+".weight"]
	if !ok {
		l.err = fmt.Errorf("birefnet: missing %s.weight", name)
		return nil
	}
	dims := wt.Shape().Dims()
	O, per := dims[0], wt.NumElements()/dims[0]
	w := l.host(name + ".weight")
	var b []float32
	if _, ok := l.hf[name+".bias"]; ok {
		b = l.host(name + ".bias")
	}
	if bn != "" {
		g, beta := l.host(bn+".weight"), l.host(bn+".bias")
		mean, vr := l.host(bn+".running_mean"), l.host(bn+".running_var")
		if l.err != nil {
			return nil
		}
		if b == nil {
			b = make([]float32, O)
		}
		for i := range O {
			s := float32(float64(g[i]) / math.Sqrt(float64(vr[i])+1e-5))
			for j := range per {
				w[i*per+j] *= s
			}
			b[i] = (b[i]-mean[i])*s + beta[i]
		}
	}
	if l.err != nil {
		return nil
	}
	c := &birefConv{Stride: 1, Pad: dims[2] / 2}
	if patch {
		c.Stride, c.Pad = dims[2], 0
	}
	if c.W, l.err = tensor.New(w, dims, l.dev); l.err != nil {
		return nil
	}
	if b != nil {
		c.B, l.err = tensor.New(b, []int{O}, l.dev)
	}
	return c
}

func (l *idolLoader) deform(p string) *birefDeform {
	return &birefDeform{Offset: l.conv(p+".atrous_conv.offset_conv", "", false), Mod: l.conv(p+".atrous_conv.modulator_conv", "", false),
		Regular: l.conv(p+".atrous_conv.regular_conv", p+".bn", false)}
}

func (l *idolLoader) decBlock(p string) *birefDecBlock {
	a := &birefASPP{ASPP1: l.deform(p + ".dec_att.aspp1"),
		Pool:  l.conv(p+".dec_att.global_avg_pool.1", p+".dec_att.global_avg_pool.2", false),
		Conv1: l.conv(p+".dec_att.conv1", p+".dec_att.bn1", false)}
	for i := 0; ; i++ {
		q := fmt.Sprintf("%s.dec_att.aspp_deforms.%d", p, i)
		if _, ok := l.hf[q+".atrous_conv.regular_conv.weight"]; !ok {
			break
		}
		a.Deforms = append(a.Deforms, l.deform(q))
	}
	return &birefDecBlock{In: l.conv(p+".conv_in", p+".bn_in", false), ASPP: a, Out: l.conv(p+".conv_out", p+".bn_out", false)}
}

// LoadBiRefNet loads idol-sdk's BiRefNet (birefnet.safetensors) onto dev.
// The layer sizes follow from the tensors; under nn.WithWeightStorage16 the
// backbone's linears keep 16-bit weights (the convolutions stay F32).
func LoadBiRefNet(path string, dev device.Device) (*BiRefNet, error) {
	l, err := newIdolLoader(path)
	if err != nil {
		return nil, err
	}
	l.dev = dev
	m := &BiRefNet{BB: l.swin("bb."), Squeeze: l.decBlock("squeeze_module.0"), InputSize: [2]int{1024, 1024}, dev: dev}
	d := &m.Dec
	for i := range 5 {
		p := fmt.Sprintf("decoder.ipt_blk%d", i+1)
		d.Ipt[i] = &birefSimple{C1: l.conv(p+".conv1", "", false), C2: l.conv(p+".conv_out", "", false)}
	}
	for i := range 4 {
		d.Blocks[i] = l.decBlock(fmt.Sprintf("decoder.decoder_block%d", i+1))
	}
	for i := range 3 {
		d.Lateral[i] = l.conv(fmt.Sprintf("decoder.lateral_block%d.conv", i+2), "", false)
		g := fmt.Sprintf("decoder.gdt_convs_%d", i+2)
		d.Gates[i] = &birefGate{Conv: l.conv(g+".0", g+".1", false), Attn: l.conv(fmt.Sprintf("decoder.gdt_convs_attn_%d.0", i+2), "", false)}
	}
	d.Out = l.conv("decoder.conv_out1.0", "", false)
	if l.err != nil {
		return nil, fmt.Errorf("birefnet: %w", l.err)
	}
	if len(m.BB.Stages) != 4 {
		return nil, fmt.Errorf("birefnet: backbone has %d stages, want 4", len(m.BB.Stages))
	}
	return m, nil
}
