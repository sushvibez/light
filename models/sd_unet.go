package models

import (
	"fmt"
	"runtime"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// ─── FeedForward (GeGLU + Linear) ────────────────────────────────────────────

// gegluLinear is the first half of a GeGLU FeedForward: Linear(d, 2*inner)
// followed by a split into (hidden, gate) and the GELU-modulated product
// `hidden * GELU(gate)`. HF SD's `BasicTransformerBlock.ff.net.0`.
type gegluLinear struct {
	Proj *nn.Linear // (d → 2*inner)
}

func newGEGLU(dim, inner int, wdt dtype.DType, dev device.Device) (*gegluLinear, error) {
	p, err := nn.NewLinearUninit(dim, 2*inner, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	return &gegluLinear{Proj: p}, nil
}

func (g *gegluLinear) forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	y, err := g.Proj.Forward(x)
	if err != nil {
		return nil, err
	}
	// One fused kernel — `hidden * GELU_tanh(gate)`. Replaces an old
	// 4×Narrow/Contiguous + 10-op composed GELU + Mul chain (16 launches +
	// two transient `(B, T, inner)` scratch tensors) with a single device
	// kernel.
	return ops.GeGLU(y)
}

// feedForward — HF's FeedForward(net=[GEGLU, Dropout, Linear]) with mult=4
// (so inner = 4*dim). Dropout is skipped at inference.
type feedForward struct {
	GEGLU *gegluLinear
	Out   *nn.Linear
}

func newFeedForward(dim int, wdt dtype.DType, dev device.Device) (*feedForward, error) {
	inner := 4 * dim
	g, err := newGEGLU(dim, inner, wdt, dev)
	if err != nil {
		return nil, err
	}
	o, err := nn.NewLinearUninit(inner, dim, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	return &feedForward{GEGLU: g, Out: o}, nil
}

func (f *feedForward) forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := f.GEGLU.forward(x)
	if err != nil {
		return nil, err
	}
	return f.Out.Forward(h)
}

func (f *feedForward) parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, f.GEGLU.Proj.Parameters()...)
	ps = append(ps, f.Out.Parameters()...)
	return ps
}

// ─── BasicTransformerBlock ────────────────────────────────────────────────────

// basicTransformerBlock is the per-Transformer2DModel layer: pre-norm
// self-attn → pre-norm cross-attn → pre-norm FF, each with a residual.
type basicTransformerBlock struct {
	N1    *nn.LayerNorm
	Attn1 *nn.CrossAttention // self-attn (ctx==x); uses qDim=kvDim=dim
	N2    *nn.LayerNorm
	Attn2 *nn.CrossAttention // cross-attn (kvDim=ctxDim)
	N3    *nn.LayerNorm
	FF    *feedForward
}

func newBasicTransformerBlock(dim, numHeads, ctxDim int, wdt dtype.DType, dev device.Device) (*basicTransformerBlock, error) {
	n1, err := nn.NewLayerNormWithDType(dim, 1e-5, wdt, dev)
	if err != nil {
		return nil, err
	}
	a1, err := nn.NewCrossAttentionWithDType(dim, dim, numHeads, wdt, dev)
	if err != nil {
		return nil, err
	}
	n2, err := nn.NewLayerNormWithDType(dim, 1e-5, wdt, dev)
	if err != nil {
		return nil, err
	}
	a2, err := nn.NewCrossAttentionWithDType(dim, ctxDim, numHeads, wdt, dev)
	if err != nil {
		return nil, err
	}
	n3, err := nn.NewLayerNormWithDType(dim, 1e-5, wdt, dev)
	if err != nil {
		return nil, err
	}
	ff, err := newFeedForward(dim, wdt, dev)
	if err != nil {
		return nil, err
	}
	return &basicTransformerBlock{N1: n1, Attn1: a1, N2: n2, Attn2: a2, N3: n3, FF: ff}, nil
}

func (b *basicTransformerBlock) forward(x, ctx *tensor.Tensor) (*tensor.Tensor, error) {
	// Self-attn
	h, err := b.N1.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = b.Attn1.Forward(h, h)
	if err != nil {
		return nil, err
	}
	x, err = autograd.Add(x, h)
	if err != nil {
		return nil, err
	}
	// Cross-attn
	h, err = b.N2.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = b.Attn2.Forward(h, ctx)
	if err != nil {
		return nil, err
	}
	x, err = autograd.Add(x, h)
	if err != nil {
		return nil, err
	}
	// FF
	h, err = b.N3.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = b.FF.forward(h)
	if err != nil {
		return nil, err
	}
	return autograd.Add(x, h)
}

func (b *basicTransformerBlock) parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, b.N1.Parameters()...)
	ps = append(ps, b.Attn1.Parameters()...)
	ps = append(ps, b.N2.Parameters()...)
	ps = append(ps, b.Attn2.Parameters()...)
	ps = append(ps, b.N3.Parameters()...)
	ps = append(ps, b.FF.parameters()...)
	return ps
}

// ─── Transformer2DModel ───────────────────────────────────────────────────────

// transformer2DModel: GroupNorm → reshape (B,C,H,W)→(B,HW,C) → proj_in
// (Linear) → BasicTransformerBlock(s) → proj_out (Linear) → reshape back +
// residual. SD 1.5 uses exactly one BasicTransformerBlock per Transformer2D.
type transformer2DModel struct {
	GN      *nn.GroupNorm
	ProjIn  *nn.Linear
	Block   *basicTransformerBlock
	ProjOut *nn.Linear
	c       int
}

func newTransformer2DModel(channels, numHeads, ctxDim int, wdt dtype.DType, dev device.Device) (*transformer2DModel, error) {
	gn, err := nn.NewGroupNormWithDType(channels, 32, 1e-6, wdt, dev)
	if err != nil {
		return nil, err
	}
	pi, err := nn.NewLinearUninit(channels, channels, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	b, err := newBasicTransformerBlock(channels, numHeads, ctxDim, wdt, dev)
	if err != nil {
		return nil, err
	}
	po, err := nn.NewLinearUninit(channels, channels, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	return &transformer2DModel{GN: gn, ProjIn: pi, Block: b, ProjOut: po, c: channels}, nil
}

func (t *transformer2DModel) forward(x, ctx *tensor.Tensor) (*tensor.Tensor, error) {
	residual := x
	B, C, H, W := x.Dim(0), x.Dim(1), x.Dim(2), x.Dim(3)
	if C != t.c {
		return nil, fmt.Errorf("transformer2D: channels %d != %d", C, t.c)
	}
	h, err := t.GN.Forward(x)
	if err != nil {
		return nil, err
	}
	// (B, C, H, W) → (B, H*W, C)
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
	h, err = t.ProjIn.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = t.Block.forward(h, ctx)
	if err != nil {
		return nil, err
	}
	h, err = t.ProjOut.Forward(h)
	if err != nil {
		return nil, err
	}
	// (B, H*W, C) → (B, C, H, W)
	h, err = autograd.Transpose(h, 1, 2)
	if err != nil {
		return nil, err
	}
	h, err = autograd.Contiguous(h)
	if err != nil {
		return nil, err
	}
	h, err = autograd.Reshape(h, B, C, H, W)
	if err != nil {
		return nil, err
	}
	out, err := autograd.Add(h, residual)
	if err != nil {
		return nil, err
	}
	// At 64×64 self-attention generates ~768 MB of (4096, 4096) BF16
	// scratch (scores → scaled → softmax) per layer. Without an explicit
	// GC between layers, three back-to-back transformer2D calls in the
	// last up_block of SD's UNet pile up >2 GB of pending finalizer
	// memory and OOM the device. Forcing a GC here keeps the per-layer
	// peak at one layer's worth.
	runtime.GC()
	return out, nil
}

func (t *transformer2DModel) parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, t.GN.Parameters()...)
	ps = append(ps, t.ProjIn.Parameters()...)
	ps = append(ps, t.Block.parameters()...)
	ps = append(ps, t.ProjOut.Parameters()...)
	return ps
}

// ─── Down blocks ──────────────────────────────────────────────────────────────

type crossAttnDownBlock struct {
	Resnets    []*resnetBlock2D
	Attns      []*transformer2DModel
	Downsample *downsample2D // nil for the last down block
}

func newCrossAttnDownBlock(in, out, layers, numHeads, ctxDim, tembDim int, addDown bool, wdt dtype.DType, dev device.Device) (*crossAttnDownBlock, error) {
	rs := make([]*resnetBlock2D, layers)
	as := make([]*transformer2DModel, layers)
	for i := 0; i < layers; i++ {
		ic := out
		if i == 0 {
			ic = in
		}
		r, err := newResnetBlock2D(ic, out, tembDim, sdResnetEpsUNet, wdt, dev)
		if err != nil {
			return nil, err
		}
		a, err := newTransformer2DModel(out, numHeads, ctxDim, wdt, dev)
		if err != nil {
			return nil, err
		}
		rs[i] = r
		as[i] = a
	}
	var ds *downsample2D
	if addDown {
		var err error
		ds, err = newDownsample2D(out, wdt, dev)
		if err != nil {
			return nil, err
		}
	}
	return &crossAttnDownBlock{Resnets: rs, Attns: as, Downsample: ds}, nil
}

func (b *crossAttnDownBlock) forward(x, temb, ctx *tensor.Tensor) (*tensor.Tensor, []*tensor.Tensor, error) {
	skips := make([]*tensor.Tensor, 0, len(b.Resnets)+1)
	h := x
	var err error
	for i := range b.Resnets {
		h, err = b.Resnets[i].forward(h, temb)
		if err != nil {
			return nil, nil, err
		}
		h, err = b.Attns[i].forward(h, ctx)
		if err != nil {
			return nil, nil, err
		}
		skips = append(skips, h)
	}
	if b.Downsample != nil {
		h, err = b.Downsample.forward(h)
		if err != nil {
			return nil, nil, err
		}
		skips = append(skips, h)
	}
	return h, skips, nil
}

func (b *crossAttnDownBlock) parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for i := range b.Resnets {
		ps = append(ps, b.Resnets[i].parameters()...)
		ps = append(ps, b.Attns[i].parameters()...)
	}
	if b.Downsample != nil {
		ps = append(ps, b.Downsample.parameters()...)
	}
	return ps
}

type downBlock struct {
	Resnets    []*resnetBlock2D
	Downsample *downsample2D
}

func newDownBlock(in, out, layers, tembDim int, addDown bool, wdt dtype.DType, dev device.Device) (*downBlock, error) {
	rs := make([]*resnetBlock2D, layers)
	for i := 0; i < layers; i++ {
		ic := out
		if i == 0 {
			ic = in
		}
		r, err := newResnetBlock2D(ic, out, tembDim, sdResnetEpsUNet, wdt, dev)
		if err != nil {
			return nil, err
		}
		rs[i] = r
	}
	var ds *downsample2D
	if addDown {
		var err error
		ds, err = newDownsample2D(out, wdt, dev)
		if err != nil {
			return nil, err
		}
	}
	return &downBlock{Resnets: rs, Downsample: ds}, nil
}

func (b *downBlock) forward(x, temb *tensor.Tensor) (*tensor.Tensor, []*tensor.Tensor, error) {
	skips := make([]*tensor.Tensor, 0, len(b.Resnets)+1)
	h := x
	var err error
	for _, r := range b.Resnets {
		h, err = r.forward(h, temb)
		if err != nil {
			return nil, nil, err
		}
		skips = append(skips, h)
	}
	if b.Downsample != nil {
		h, err = b.Downsample.forward(h)
		if err != nil {
			return nil, nil, err
		}
		skips = append(skips, h)
	}
	return h, skips, nil
}

func (b *downBlock) parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, r := range b.Resnets {
		ps = append(ps, r.parameters()...)
	}
	if b.Downsample != nil {
		ps = append(ps, b.Downsample.parameters()...)
	}
	return ps
}

// ─── Mid block ────────────────────────────────────────────────────────────────

type unetMidBlockCrossAttn struct {
	R0   *resnetBlock2D
	Attn *transformer2DModel
	R1   *resnetBlock2D
}

func newUNetMidBlockCrossAttn(channels, numHeads, ctxDim, tembDim int, wdt dtype.DType, dev device.Device) (*unetMidBlockCrossAttn, error) {
	r0, err := newResnetBlock2D(channels, channels, tembDim, sdResnetEpsUNet, wdt, dev)
	if err != nil {
		return nil, err
	}
	a, err := newTransformer2DModel(channels, numHeads, ctxDim, wdt, dev)
	if err != nil {
		return nil, err
	}
	r1, err := newResnetBlock2D(channels, channels, tembDim, sdResnetEpsUNet, wdt, dev)
	if err != nil {
		return nil, err
	}
	return &unetMidBlockCrossAttn{R0: r0, Attn: a, R1: r1}, nil
}

func (m *unetMidBlockCrossAttn) forward(x, temb, ctx *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := m.R0.forward(x, temb)
	if err != nil {
		return nil, err
	}
	h, err = m.Attn.forward(h, ctx)
	if err != nil {
		return nil, err
	}
	return m.R1.forward(h, temb)
}

func (m *unetMidBlockCrossAttn) parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, m.R0.parameters()...)
	ps = append(ps, m.Attn.parameters()...)
	ps = append(ps, m.R1.parameters()...)
	return ps
}

// ─── Up blocks ────────────────────────────────────────────────────────────────

type crossAttnUpBlock struct {
	Resnets  []*resnetBlock2D
	Attns    []*transformer2DModel
	Upsample *upsample2D // nil for the last up block
}

// resInChannels returns the resnet input channel count for layer i in an up
// block, mirroring HF's:
//
//	prev_output_channel + (in_channels if i == num_layers-1 else out_channels)  if i==0
//	out_channels + ...                                                          otherwise
func resInChannels(i, numLayers, prevOut, out, in int) int {
	skipCh := out
	if i == numLayers-1 {
		skipCh = in
	}
	resnetIn := out
	if i == 0 {
		resnetIn = prevOut
	}
	return resnetIn + skipCh
}

func newCrossAttnUpBlock(prevOut, out, in, layers, numHeads, ctxDim, tembDim int, addUp bool, wdt dtype.DType, dev device.Device) (*crossAttnUpBlock, error) {
	rs := make([]*resnetBlock2D, layers)
	as := make([]*transformer2DModel, layers)
	for i := 0; i < layers; i++ {
		ic := resInChannels(i, layers, prevOut, out, in)
		r, err := newResnetBlock2D(ic, out, tembDim, sdResnetEpsUNet, wdt, dev)
		if err != nil {
			return nil, err
		}
		a, err := newTransformer2DModel(out, numHeads, ctxDim, wdt, dev)
		if err != nil {
			return nil, err
		}
		rs[i] = r
		as[i] = a
	}
	var up *upsample2D
	if addUp {
		var err error
		up, err = newUpsample2D(out, wdt, dev)
		if err != nil {
			return nil, err
		}
	}
	return &crossAttnUpBlock{Resnets: rs, Attns: as, Upsample: up}, nil
}

func (b *crossAttnUpBlock) forward(x, temb, ctx *tensor.Tensor, skips *[]*tensor.Tensor) (*tensor.Tensor, error) {
	h := x
	var err error
	for i := range b.Resnets {
		// Pop the most recent skip and concat along channels.
		s := (*skips)[len(*skips)-1]
		*skips = (*skips)[:len(*skips)-1]
		h, err = ops.Concat([]*tensor.Tensor{h, s}, 1)
		if err != nil {
			return nil, err
		}
		h, err = b.Resnets[i].forward(h, temb)
		if err != nil {
			return nil, err
		}
		h, err = b.Attns[i].forward(h, ctx)
		if err != nil {
			return nil, err
		}
	}
	if b.Upsample != nil {
		h, err = b.Upsample.forward(h)
		if err != nil {
			return nil, err
		}
	}
	return h, nil
}

func (b *crossAttnUpBlock) parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for i := range b.Resnets {
		ps = append(ps, b.Resnets[i].parameters()...)
		ps = append(ps, b.Attns[i].parameters()...)
	}
	if b.Upsample != nil {
		ps = append(ps, b.Upsample.parameters()...)
	}
	return ps
}

type upBlock struct {
	Resnets  []*resnetBlock2D
	Upsample *upsample2D
}

func newUpBlock(prevOut, out, in, layers, tembDim int, addUp bool, wdt dtype.DType, dev device.Device) (*upBlock, error) {
	rs := make([]*resnetBlock2D, layers)
	for i := 0; i < layers; i++ {
		ic := resInChannels(i, layers, prevOut, out, in)
		r, err := newResnetBlock2D(ic, out, tembDim, sdResnetEpsUNet, wdt, dev)
		if err != nil {
			return nil, err
		}
		rs[i] = r
	}
	var up *upsample2D
	if addUp {
		var err error
		up, err = newUpsample2D(out, wdt, dev)
		if err != nil {
			return nil, err
		}
	}
	return &upBlock{Resnets: rs, Upsample: up}, nil
}

func (b *upBlock) forward(x, temb *tensor.Tensor, skips *[]*tensor.Tensor) (*tensor.Tensor, error) {
	h := x
	var err error
	for _, r := range b.Resnets {
		s := (*skips)[len(*skips)-1]
		*skips = (*skips)[:len(*skips)-1]
		h, err = ops.Concat([]*tensor.Tensor{h, s}, 1)
		if err != nil {
			return nil, err
		}
		h, err = r.forward(h, temb)
		if err != nil {
			return nil, err
		}
	}
	if b.Upsample != nil {
		h, err = b.Upsample.forward(h)
		if err != nil {
			return nil, err
		}
	}
	return h, nil
}

func (b *upBlock) parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, r := range b.Resnets {
		ps = append(ps, r.parameters()...)
	}
	if b.Upsample != nil {
		ps = append(ps, b.Upsample.parameters()...)
	}
	return ps
}

// ─── UNet2DConditionModel ─────────────────────────────────────────────────────

// SDUNetConfig describes UNet2DConditionModel. Defaults match SD 1.5
// (`runwayml/stable-diffusion-v1-5/unet/config.json`).
type SDUNetConfig struct {
	InChannels        int
	OutChannels       int
	BlockOutChannels  []int       // {320, 640, 1280, 1280} for SD 1.5
	LayersPerBlock    int         // 2
	NumHeads          int         // 8 for SD 1.5
	CrossAttentionDim int         // 768 (CLIP-L hidden)
	TimeEmbDim        int         // 1280 = 4 * BlockOutChannels[0]
	NormGroups        int         // 32
	WeightDType       dtype.DType // BF16 for half-memory inference; defaults to F32
}

// SDUNetSD15 returns the SD 1.5 default UNet config.
func SDUNetSD15() SDUNetConfig {
	return SDUNetConfig{
		InChannels: 4, OutChannels: 4,
		BlockOutChannels:  []int{320, 640, 1280, 1280},
		LayersPerBlock:    2,
		NumHeads:          8,
		CrossAttentionDim: 768,
		TimeEmbDim:        1280,
		NormGroups:        32,
	}
}

// SDUNet is the UNet2DConditionModel for SD 1.5. Forward consumes a
// (B, 4, H, W) latent, a (B, 1) F32 timestep tensor, and a (B, T, 768)
// context (CLIP hidden states), and produces the predicted noise ε with
// the same shape as the latent.
type SDUNet struct {
	Cfg SDUNetConfig

	ConvIn     *nn.Conv2d
	TimeEmbed1 *nn.Linear
	TimeEmbed2 *nn.Linear
	DownBlocks []sdDownBlock // mixed types: crossAttnDownBlock or downBlock
	Mid        *unetMidBlockCrossAttn
	UpBlocks   []sdUpBlock // mixed: crossAttnUpBlock or upBlock
	NormOut    *nn.GroupNorm
	ConvOut    *nn.Conv2d
}

// sdDownBlock is a sum-type wrapper so DownBlocks can mix the CrossAttn and
// plain variants. Exactly one of `CA` or `Plain` is non-nil.
type sdDownBlock struct {
	CA    *crossAttnDownBlock
	Plain *downBlock
}

func (b sdDownBlock) forward(x, temb, ctx *tensor.Tensor) (*tensor.Tensor, []*tensor.Tensor, error) {
	if b.CA != nil {
		return b.CA.forward(x, temb, ctx)
	}
	return b.Plain.forward(x, temb)
}

func (b sdDownBlock) parameters() []*tensor.Tensor {
	if b.CA != nil {
		return b.CA.parameters()
	}
	return b.Plain.parameters()
}

type sdUpBlock struct {
	CA    *crossAttnUpBlock
	Plain *upBlock
}

func (b sdUpBlock) forward(x, temb, ctx *tensor.Tensor, skips *[]*tensor.Tensor) (*tensor.Tensor, error) {
	if b.CA != nil {
		return b.CA.forward(x, temb, ctx, skips)
	}
	return b.Plain.forward(x, temb, skips)
}

func (b sdUpBlock) parameters() []*tensor.Tensor {
	if b.CA != nil {
		return b.CA.parameters()
	}
	return b.Plain.parameters()
}

// NewSDUNet constructs the SD 1.5 UNet with zero-init weights at the dtype
// specified by `cfg.WeightDType` (F32 default; BF16 supported for half-mem
// inference). Loaders are expected to overwrite values immediately.
func NewSDUNet(cfg SDUNetConfig, dev device.Device) (*SDUNet, error) {
	if len(cfg.BlockOutChannels) != 4 {
		return nil, fmt.Errorf("SDUNet: expect 4 block_out_channels, got %d", len(cfg.BlockOutChannels))
	}
	wdt := cfg.WeightDType
	if wdt == dtype.F16 { // F16 is the dtype enum's zero-value; treat as "default to F32"
		wdt = dtype.F32
	}
	bOC := cfg.BlockOutChannels
	convIn, err := nn.NewConv2dUninit(cfg.InChannels, bOC[0], 3, 3, [2]int{1, 1}, [2]int{1, 1}, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	t1, err := nn.NewLinearUninit(bOC[0], cfg.TimeEmbDim, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	t2, err := nn.NewLinearUninit(cfg.TimeEmbDim, cfg.TimeEmbDim, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	// Down blocks: 3 cross-attn, 1 plain.
	downBlocks := make([]sdDownBlock, 4)
	prev := bOC[0]
	for i := 0; i < 4; i++ {
		out := bOC[i]
		isLast := i == 3
		addDown := !isLast
		if i < 3 {
			b, err := newCrossAttnDownBlock(prev, out, cfg.LayersPerBlock, cfg.NumHeads, cfg.CrossAttentionDim, cfg.TimeEmbDim, addDown, wdt, dev)
			if err != nil {
				return nil, fmt.Errorf("down block %d: %w", i, err)
			}
			downBlocks[i] = sdDownBlock{CA: b}
		} else {
			b, err := newDownBlock(prev, out, cfg.LayersPerBlock, cfg.TimeEmbDim, addDown, wdt, dev)
			if err != nil {
				return nil, fmt.Errorf("down block %d: %w", i, err)
			}
			downBlocks[i] = sdDownBlock{Plain: b}
		}
		prev = out
	}
	mid, err := newUNetMidBlockCrossAttn(bOC[3], cfg.NumHeads, cfg.CrossAttentionDim, cfg.TimeEmbDim, wdt, dev)
	if err != nil {
		return nil, err
	}
	// Up blocks: reversed channel layout, layers_per_block + 1 = 3 resnets.
	upBlocks := make([]sdUpBlock, 4)
	revOC := []int{bOC[3], bOC[2], bOC[1], bOC[0]}
	prev = bOC[3] // mid output
	for i := 0; i < 4; i++ {
		out := revOC[i]
		var in int
		if i+1 >= 4 {
			in = revOC[3]
		} else {
			in = revOC[i+1]
		}
		isLast := i == 3
		addUp := !isLast
		layers := cfg.LayersPerBlock + 1
		if i == 0 {
			b, err := newUpBlock(prev, out, in, layers, cfg.TimeEmbDim, addUp, wdt, dev)
			if err != nil {
				return nil, fmt.Errorf("up block %d: %w", i, err)
			}
			upBlocks[i] = sdUpBlock{Plain: b}
		} else {
			b, err := newCrossAttnUpBlock(prev, out, in, layers, cfg.NumHeads, cfg.CrossAttentionDim, cfg.TimeEmbDim, addUp, wdt, dev)
			if err != nil {
				return nil, fmt.Errorf("up block %d: %w", i, err)
			}
			upBlocks[i] = sdUpBlock{CA: b}
		}
		prev = out
	}
	normOut, err := nn.NewGroupNormWithDType(bOC[0], cfg.NormGroups, 1e-5, wdt, dev)
	if err != nil {
		return nil, err
	}
	convOut, err := nn.NewConv2dUninit(bOC[0], cfg.OutChannels, 3, 3, [2]int{1, 1}, [2]int{1, 1}, true, wdt, dev)
	if err != nil {
		return nil, err
	}
	return &SDUNet{
		Cfg: cfg, ConvIn: convIn, TimeEmbed1: t1, TimeEmbed2: t2,
		DownBlocks: downBlocks, Mid: mid, UpBlocks: upBlocks,
		NormOut: normOut, ConvOut: convOut,
	}, nil
}

// Forward runs the UNet noise-prediction. `timesteps` is a (B,) F32 tensor
// of current diffusion step indices; the time-embedding MLP is applied
// internally. `ctx` is the (B, T, ctxDim) cross-attention context.
func (m *SDUNet) Forward(latent, timestepEmb, ctx *tensor.Tensor) (*tensor.Tensor, error) {
	// Time-embed MLP: (B, BlockOut[0]) → (B, TimeEmbDim) → SiLU → (B, TimeEmbDim)
	t, err := m.TimeEmbed1.Forward(timestepEmb)
	if err != nil {
		return nil, err
	}
	t, err = autograd.SiLU(t)
	if err != nil {
		return nil, err
	}
	temb, err := m.TimeEmbed2.Forward(t)
	if err != nil {
		return nil, err
	}
	// conv_in
	h, err := m.ConvIn.Forward(latent)
	if err != nil {
		return nil, err
	}
	skips := []*tensor.Tensor{h}
	// Down blocks
	for i := range m.DownBlocks {
		var bs []*tensor.Tensor
		h, bs, err = m.DownBlocks[i].forward(h, temb, ctx)
		if err != nil {
			return nil, fmt.Errorf("down block %d: %w", i, err)
		}
		skips = append(skips, bs...)
	}
	// Mid
	h, err = m.Mid.forward(h, temb, ctx)
	if err != nil {
		return nil, err
	}
	// Up blocks (consume skips in LIFO order)
	for i := range m.UpBlocks {
		h, err = m.UpBlocks[i].forward(h, temb, ctx, &skips)
		if err != nil {
			return nil, fmt.Errorf("up block %d: %w", i, err)
		}
	}
	// conv_norm_out → SiLU → conv_out
	h, err = m.NormOut.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = autograd.SiLU(h)
	if err != nil {
		return nil, err
	}
	return m.ConvOut.Forward(h)
}

// Parameters returns every learnable tensor.
func (m *SDUNet) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, m.ConvIn.Parameters()...)
	ps = append(ps, m.TimeEmbed1.Parameters()...)
	ps = append(ps, m.TimeEmbed2.Parameters()...)
	for _, b := range m.DownBlocks {
		ps = append(ps, b.parameters()...)
	}
	ps = append(ps, m.Mid.parameters()...)
	for _, b := range m.UpBlocks {
		ps = append(ps, b.parameters()...)
	}
	ps = append(ps, m.NormOut.Parameters()...)
	ps = append(ps, m.ConvOut.Parameters()...)
	return ps
}

// ParamCount returns the total trainable element count.
func (m *SDUNet) ParamCount() int {
	total := 0
	for _, p := range m.Parameters() {
		total += p.NumElements()
	}
	return total
}
