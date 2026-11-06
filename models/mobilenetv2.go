package models

import (
	"fmt"
	"math"

	"light/autograd"
	"light/device"
	"light/nn"
	"light/tensor"
)

// MobileNetV2 implements the canonical 224×224 ImageNet network from
// Sandler et al. 2018, "MobileNetV2: Inverted Residuals and Linear
// Bottlenecks". Uses depthwise-separable convolutions everywhere. The
// inverted-residual block expands channels by factor t, runs a 3×3
// depthwise conv at stride s, then projects back to a smaller width with
// a 1×1 conv (no activation on the project — the "linear bottleneck").
//
// Standard configuration (t, c, n, s):
//
//	(1, 16, 1, 1) (6, 24, 2, 2) (6, 32, 3, 2) (6, 64, 4, 2)
//	(6, 96, 3, 1) (6, 160, 3, 2) (6, 320, 1, 1)
//
// Width multiplier α = 1.0 in this preset; pass `widthMult` to scale.
type MobileNetV2 struct {
	Stem     *nn.Conv2d
	StemBN   *nn.BatchNorm2d
	Blocks   []*invertedResidual
	HeadConv *nn.Conv2d
	HeadBN   *nn.BatchNorm2d
	Pool     *nn.AdaptiveAvgPool2d
	FC       *nn.Linear
	training bool
}

type invertedResidual struct {
	Expand      *nn.Conv2d // 1×1 expand (nil when t=1)
	ExpandBN    *nn.BatchNorm2d
	Depthwise   *nn.GroupedConv2d // 3×3 depthwise stride=s
	DwBN        *nn.BatchNorm2d
	Project     *nn.Conv2d // 1×1 project, linear (no activation)
	ProjectBN   *nn.BatchNorm2d
	useResidual bool
}

// NewMobileNetV2 builds the standard width-multiplier-1.0 network.
func NewMobileNetV2(numClasses int, dev device.Device) (*MobileNetV2, error) {
	return NewMobileNetV2WithMult(numClasses, 1.0, dev)
}

// NewMobileNetV2WithMult builds MobileNetV2 with a custom width multiplier.
// Channel counts are rounded to the nearest multiple of 8 (matching torchvision).
func NewMobileNetV2WithMult(numClasses int, widthMult float64, dev device.Device) (*MobileNetV2, error) {
	if numClasses <= 0 {
		return nil, fmt.Errorf("models.MobileNetV2: numClasses must be positive")
	}
	round := func(c int) int {
		ch := int(float64(c) * widthMult)
		// Round to the nearest multiple of 8, never below 0.9·c.
		newC := ((ch + 4) / 8) * 8
		if float64(newC) < 0.9*float64(c)*widthMult {
			newC += 8
		}
		if newC < 8 {
			newC = 8
		}
		return newC
	}
	stemCh := round(32)
	lastCh := round(1280)
	if widthMult <= 1.0 {
		// torchvision keeps the head at 1280 unless mult > 1.
		lastCh = 1280
	}

	stem, err := nn.NewConv2d(3, stemCh, 3, 3, [2]int{2, 2}, [2]int{1, 1}, false, dev)
	if err != nil {
		return nil, err
	}
	stemBN, err := nn.NewBatchNorm2d(stemCh, 1e-5, 0.1, dev)
	if err != nil {
		return nil, err
	}

	type stage struct{ t, c, n, s int }
	cfg := []stage{
		{1, 16, 1, 1}, {6, 24, 2, 2}, {6, 32, 3, 2}, {6, 64, 4, 2},
		{6, 96, 3, 1}, {6, 160, 3, 2}, {6, 320, 1, 1},
	}

	blocks := make([]*invertedResidual, 0, 17)
	cin := stemCh
	for _, s := range cfg {
		out := round(s.c)
		for i := range s.n {
			stride := 1
			if i == 0 {
				stride = s.s
			}
			blk, err := newInvertedResidual(cin, out, s.t, stride, dev)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, blk)
			cin = out
		}
	}

	headConv, err := nn.NewConv2d(cin, lastCh, 1, 1, [2]int{1, 1}, [2]int{0, 0}, false, dev)
	if err != nil {
		return nil, err
	}
	headBN, err := nn.NewBatchNorm2d(lastCh, 1e-5, 0.1, dev)
	if err != nil {
		return nil, err
	}
	fc, err := nn.NewLinear(lastCh, numClasses, true, dev)
	if err != nil {
		return nil, err
	}
	return &MobileNetV2{
		Stem: stem, StemBN: stemBN,
		Blocks:   blocks,
		HeadConv: headConv, HeadBN: headBN,
		Pool:     nn.NewAdaptiveAvgPool2d(1, 1),
		FC:       fc,
		training: true,
	}, nil
}

func newInvertedResidual(cin, cout, expand, stride int, dev device.Device) (*invertedResidual, error) {
	hidden := cin * expand
	b := &invertedResidual{useResidual: stride == 1 && cin == cout}
	if expand != 1 {
		ex, err := nn.NewConv2d(cin, hidden, 1, 1, [2]int{1, 1}, [2]int{0, 0}, false, dev)
		if err != nil {
			return nil, err
		}
		exBN, err := nn.NewBatchNorm2d(hidden, 1e-5, 0.1, dev)
		if err != nil {
			return nil, err
		}
		b.Expand = ex
		b.ExpandBN = exBN
	}
	dw, err := nn.NewDepthwiseConv2d(hidden, 3, 3, [2]int{stride, stride}, [2]int{1, 1}, false, dev)
	if err != nil {
		return nil, err
	}
	dwBN, err := nn.NewBatchNorm2d(hidden, 1e-5, 0.1, dev)
	if err != nil {
		return nil, err
	}
	pr, err := nn.NewConv2d(hidden, cout, 1, 1, [2]int{1, 1}, [2]int{0, 0}, false, dev)
	if err != nil {
		return nil, err
	}
	prBN, err := nn.NewBatchNorm2d(cout, 1e-5, 0.1, dev)
	if err != nil {
		return nil, err
	}
	b.Depthwise = dw
	b.DwBN = dwBN
	b.Project = pr
	b.ProjectBN = prBN
	return b, nil
}

func (b *invertedResidual) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	h := x
	var err error
	if b.Expand != nil {
		h, err = b.Expand.Forward(h)
		if err != nil {
			return nil, err
		}
		h, err = b.ExpandBN.Forward(h)
		if err != nil {
			return nil, err
		}
		h, err = autograd.ReLU6(h)
		if err != nil {
			return nil, err
		}
	}
	h, err = b.Depthwise.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = b.DwBN.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = autograd.ReLU6(h)
	if err != nil {
		return nil, err
	}
	h, err = b.Project.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = b.ProjectBN.Forward(h)
	if err != nil {
		return nil, err
	}
	if b.useResidual {
		return autograd.Add(h, x)
	}
	return h, nil
}

func (b *invertedResidual) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	if b.Expand != nil {
		ps = append(ps, b.Expand.Parameters()...)
		ps = append(ps, b.ExpandBN.Parameters()...)
	}
	ps = append(ps, b.Depthwise.Parameters()...)
	ps = append(ps, b.DwBN.Parameters()...)
	ps = append(ps, b.Project.Parameters()...)
	ps = append(ps, b.ProjectBN.Parameters()...)
	return ps
}

func (b *invertedResidual) SetTraining(v bool) {
	if b.ExpandBN != nil {
		b.ExpandBN.SetTraining(v)
	}
	b.DwBN.SetTraining(v)
	b.ProjectBN.SetTraining(v)
}

// Forward runs the full network.
func (m *MobileNetV2) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := m.Stem.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = m.StemBN.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = autograd.ReLU6(h)
	if err != nil {
		return nil, err
	}
	for _, b := range m.Blocks {
		h, err = b.Forward(h)
		if err != nil {
			return nil, err
		}
	}
	h, err = m.HeadConv.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = m.HeadBN.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = autograd.ReLU6(h)
	if err != nil {
		return nil, err
	}
	h, err = m.Pool.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = (nn.Flatten{}).Forward(h)
	if err != nil {
		return nil, err
	}
	return m.FC.Forward(h)
}

func (m *MobileNetV2) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, m.Stem.Parameters()...)
	ps = append(ps, m.StemBN.Parameters()...)
	for _, b := range m.Blocks {
		ps = append(ps, b.Parameters()...)
	}
	ps = append(ps, m.HeadConv.Parameters()...)
	ps = append(ps, m.HeadBN.Parameters()...)
	ps = append(ps, m.FC.Parameters()...)
	return ps
}

func (m *MobileNetV2) SetTraining(v bool) {
	m.training = v
	m.StemBN.SetTraining(v)
	for _, b := range m.Blocks {
		b.SetTraining(v)
	}
	m.HeadBN.SetTraining(v)
}

func (m *MobileNetV2) ParamCount() int {
	total := 0
	for _, p := range m.Parameters() {
		total += p.NumElements()
	}
	return total
}

// guard against the math import being unused if presets are pruned.
var _ = math.Pi
