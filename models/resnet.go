// Package models holds reference implementations of canonical
// architectures. Each model is a `nn.Module`-compatible struct so users get
// a one-call constructor and a predictable Parameters() iteration order —
// important for loading PyTorch-trained checkpoints into Light.
package models

import (
	"fmt"

	"light/autograd"
	"light/device"
	"light/nn"
	"light/tensor"
)

// ResNetCIFAR implements ResNet-20/32/44/56 for CIFAR-10 / CIFAR-100
// following He et al. 2015 (Deep Residual Learning for Image Recognition,
// section 4.2). All variants start with a 3×3 "stem" conv to 16 channels,
// stack three groups of residual blocks at widths (16, 32, 64) with 2n
// blocks per group (so the total depth is 6n+2), then global-average-pool
// to a 10-wide classification head.
//
//	depth = 6n + 2   ⇒   n blocks per group
//	ResNet-20  → n = 3
//	ResNet-32  → n = 5
//	ResNet-56  → n = 9
type ResNetCIFAR struct {
	Stem     *nn.Conv2d
	StemBN   *nn.BatchNorm2d
	Group1   []*resBlock
	Group2   []*resBlock
	Group3   []*resBlock
	Pool     *nn.AdaptiveAvgPool2d
	FC       *nn.Linear
	training bool
}

type resBlock struct {
	Conv1, Conv2 *nn.Conv2d
	BN1, BN2     *nn.BatchNorm2d
	// Shortcut is only populated when stride != 1 or cin != cout.
	ShortConv *nn.Conv2d
	ShortBN   *nn.BatchNorm2d
	stride    int
}

// NewResNetCIFAR builds a ResNet with 2n residual blocks per group (i.e.,
// depth 6n+2). Use n=3 for ResNet-20.
func NewResNetCIFAR(n, numClasses int, dev device.Device) (*ResNetCIFAR, error) {
	if n <= 0 {
		return nil, fmt.Errorf("models.ResNetCIFAR: n must be positive")
	}
	stem, err := nn.NewConv2d(3, 16, 3, 3, [2]int{1, 1}, [2]int{1, 1}, false, dev)
	if err != nil {
		return nil, err
	}
	stemBN, err := nn.NewBatchNorm2d(16, 1e-5, 0.1, dev)
	if err != nil {
		return nil, err
	}
	g1, err := buildGroup(n, 16, 16, 1, dev)
	if err != nil {
		return nil, err
	}
	g2, err := buildGroup(n, 16, 32, 2, dev)
	if err != nil {
		return nil, err
	}
	g3, err := buildGroup(n, 32, 64, 2, dev)
	if err != nil {
		return nil, err
	}
	fc, err := nn.NewLinear(64, numClasses, true, dev)
	if err != nil {
		return nil, err
	}
	return &ResNetCIFAR{
		Stem: stem, StemBN: stemBN,
		Group1: g1, Group2: g2, Group3: g3,
		Pool:     nn.NewAdaptiveAvgPool2d(1, 1),
		FC:       fc,
		training: true,
	}, nil
}

// buildGroup returns a slice of n residual blocks: the first one can change
// width / downsample (via `firstStride`), the rest keep it at `cout`.
func buildGroup(n, cin, cout, firstStride int, dev device.Device) ([]*resBlock, error) {
	blocks := make([]*resBlock, n)
	for i := range n {
		stride := 1
		inC := cout
		if i == 0 {
			stride = firstStride
			inC = cin
		}
		b, err := newResBlock(inC, cout, stride, dev)
		if err != nil {
			return nil, err
		}
		blocks[i] = b
	}
	return blocks, nil
}

func newResBlock(cin, cout, stride int, dev device.Device) (*resBlock, error) {
	c1, err := nn.NewConv2d(cin, cout, 3, 3, [2]int{stride, stride}, [2]int{1, 1}, false, dev)
	if err != nil {
		return nil, err
	}
	bn1, err := nn.NewBatchNorm2d(cout, 1e-5, 0.1, dev)
	if err != nil {
		return nil, err
	}
	c2, err := nn.NewConv2d(cout, cout, 3, 3, [2]int{1, 1}, [2]int{1, 1}, false, dev)
	if err != nil {
		return nil, err
	}
	bn2, err := nn.NewBatchNorm2d(cout, 1e-5, 0.1, dev)
	if err != nil {
		return nil, err
	}
	rb := &resBlock{Conv1: c1, BN1: bn1, Conv2: c2, BN2: bn2, stride: stride}
	if stride != 1 || cin != cout {
		sc, err := nn.NewConv2d(cin, cout, 1, 1, [2]int{stride, stride}, [2]int{0, 0}, false, dev)
		if err != nil {
			return nil, err
		}
		sbn, err := nn.NewBatchNorm2d(cout, 1e-5, 0.1, dev)
		if err != nil {
			return nil, err
		}
		rb.ShortConv = sc
		rb.ShortBN = sbn
	}
	return rb, nil
}

func (r *ResNetCIFAR) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, r.Stem.Parameters()...)
	ps = append(ps, r.StemBN.Parameters()...)
	for _, g := range [][]*resBlock{r.Group1, r.Group2, r.Group3} {
		for _, b := range g {
			ps = append(ps, b.Conv1.Parameters()...)
			ps = append(ps, b.BN1.Parameters()...)
			ps = append(ps, b.Conv2.Parameters()...)
			ps = append(ps, b.BN2.Parameters()...)
			if b.ShortConv != nil {
				ps = append(ps, b.ShortConv.Parameters()...)
				ps = append(ps, b.ShortBN.Parameters()...)
			}
		}
	}
	ps = append(ps, r.FC.Parameters()...)
	return ps
}

// SetTraining toggles BatchNorm's train/eval mode across the whole network.
func (r *ResNetCIFAR) SetTraining(v bool) {
	r.training = v
	r.StemBN.SetTraining(v)
	for _, g := range [][]*resBlock{r.Group1, r.Group2, r.Group3} {
		for _, b := range g {
			b.BN1.SetTraining(v)
			b.BN2.SetTraining(v)
			if b.ShortBN != nil {
				b.ShortBN.SetTraining(v)
			}
		}
	}
}

// Forward runs the full stack: stem → 3 groups → pool → FC.
func (r *ResNetCIFAR) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := r.Stem.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = r.StemBN.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = autograd.ReLU(h)
	if err != nil {
		return nil, err
	}
	for _, g := range [][]*resBlock{r.Group1, r.Group2, r.Group3} {
		for _, b := range g {
			h, err = b.Forward(h)
			if err != nil {
				return nil, err
			}
		}
	}
	h, err = r.Pool.Forward(h) // (N, 64, 1, 1)
	if err != nil {
		return nil, err
	}
	h, err = (nn.Flatten{}).Forward(h) // (N, 64)
	if err != nil {
		return nil, err
	}
	return r.FC.Forward(h)
}

func (b *resBlock) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	// Main path: Conv → BN → ReLU → Conv → BN.
	h, err := b.Conv1.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = b.BN1.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = autograd.ReLU(h)
	if err != nil {
		return nil, err
	}
	h, err = b.Conv2.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = b.BN2.Forward(h)
	if err != nil {
		return nil, err
	}
	// Shortcut: identity or 1×1 projection.
	short := x
	if b.ShortConv != nil {
		short, err = b.ShortConv.Forward(x)
		if err != nil {
			return nil, err
		}
		short, err = b.ShortBN.Forward(short)
		if err != nil {
			return nil, err
		}
	}
	sum, err := autograd.Add(h, short)
	if err != nil {
		return nil, err
	}
	return autograd.ReLU(sum)
}
