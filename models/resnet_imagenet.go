package models

import (
	"fmt"

	"light/autograd"
	"light/device"
	"light/nn"
	"light/tensor"
)

// ResNet is the torchvision-style ResNet for ImageNet (224×224 input):
//
//	Stem    : 7×7 / stride 2 / 64ch → BN → ReLU → 3×3 MaxPool / stride 2
//	Stage 1 : N₁ blocks at 64 ch
//	Stage 2 : N₂ blocks at 128 ch (first block downsamples 2×)
//	Stage 3 : N₃ blocks at 256 ch (first block downsamples 2×)
//	Stage 4 : N₄ blocks at 512 ch (first block downsamples 2×)
//	Head    : AdaptiveAvgPool(1) → Flatten → Linear(num_classes)
//
// ResNet-18 / 34 use `BasicBlock`; ResNet-50 / 101 / 152 use `Bottleneck`.
type ResNet struct {
	Stem      *nn.Conv2d
	StemBN    *nn.BatchNorm2d
	StemPool  *nn.MaxPool2d
	Stages    [][]resBlocker
	Pool      *nn.AdaptiveAvgPool2d
	FC        *nn.Linear
	expansion int
}

// resBlocker is the abstract residual-block interface so a ResNet can hold
// either BasicBlock or Bottleneck blocks in the same slice.
type resBlocker interface {
	Forward(*tensor.Tensor) (*tensor.Tensor, error)
	Parameters() []*tensor.Tensor
	SetTraining(bool)
}

// resnetSpec packages a depth's per-stage block count and which block type
// to use, mirroring the torchvision factory functions.
type resnetSpec struct {
	blocks    [4]int
	useBottle bool
}

var (
	specResNet18  = resnetSpec{blocks: [4]int{2, 2, 2, 2}, useBottle: false}
	specResNet34  = resnetSpec{blocks: [4]int{3, 4, 6, 3}, useBottle: false}
	specResNet50  = resnetSpec{blocks: [4]int{3, 4, 6, 3}, useBottle: true}
	specResNet101 = resnetSpec{blocks: [4]int{3, 4, 23, 3}, useBottle: true}
	specResNet152 = resnetSpec{blocks: [4]int{3, 8, 36, 3}, useBottle: true}
)

// NewResNet18 builds the canonical 18-layer torchvision ResNet for `numClasses`.
func NewResNet18(numClasses int, dev device.Device) (*ResNet, error) {
	return newResNet(specResNet18, numClasses, dev)
}

// NewResNet34 builds the 34-layer variant.
func NewResNet34(numClasses int, dev device.Device) (*ResNet, error) {
	return newResNet(specResNet34, numClasses, dev)
}

// NewResNet50 builds the 50-layer variant (bottleneck blocks).
func NewResNet50(numClasses int, dev device.Device) (*ResNet, error) {
	return newResNet(specResNet50, numClasses, dev)
}

// NewResNet101 builds the 101-layer variant.
func NewResNet101(numClasses int, dev device.Device) (*ResNet, error) {
	return newResNet(specResNet101, numClasses, dev)
}

// NewResNet152 builds the 152-layer variant.
func NewResNet152(numClasses int, dev device.Device) (*ResNet, error) {
	return newResNet(specResNet152, numClasses, dev)
}

func newResNet(spec resnetSpec, numClasses int, dev device.Device) (*ResNet, error) {
	if numClasses <= 0 {
		return nil, fmt.Errorf("models.ResNet: numClasses must be positive")
	}
	stem, err := nn.NewConv2d(3, 64, 7, 7, [2]int{2, 2}, [2]int{3, 3}, false, dev)
	if err != nil {
		return nil, err
	}
	stemBN, err := nn.NewBatchNorm2d(64, 1e-5, 0.1, dev)
	if err != nil {
		return nil, err
	}
	stemPool := nn.NewMaxPool2d([2]int{3, 3}, [2]int{2, 2}, [2]int{1, 1})

	expansion := 1
	if spec.useBottle {
		expansion = 4
	}

	stages := make([][]resBlocker, 4)
	prevOut := 64
	planes := [4]int{64, 128, 256, 512}
	for s := range 4 {
		stride := 2
		if s == 0 {
			stride = 1
		}
		layer, lastOut, err := buildResStage(spec, prevOut, planes[s], spec.blocks[s], stride, dev)
		if err != nil {
			return nil, err
		}
		stages[s] = layer
		prevOut = lastOut
	}
	pool := nn.NewAdaptiveAvgPool2d(1, 1)
	fc, err := nn.NewLinear(512*expansion, numClasses, true, dev)
	if err != nil {
		return nil, err
	}
	return &ResNet{
		Stem: stem, StemBN: stemBN, StemPool: stemPool,
		Stages: stages, Pool: pool, FC: fc, expansion: expansion,
	}, nil
}

func buildResStage(spec resnetSpec, cin, planes, n, firstStride int, dev device.Device) ([]resBlocker, int, error) {
	expansion := 1
	if spec.useBottle {
		expansion = 4
	}
	out := planes * expansion
	blocks := make([]resBlocker, 0, n)
	for i := range n {
		stride := 1
		inC := out
		if i == 0 {
			stride = firstStride
			inC = cin
		}
		var b resBlocker
		var err error
		if spec.useBottle {
			b, err = newBottleneck(inC, planes, stride, dev)
		} else {
			b, err = newBasicBlock(inC, planes, stride, dev)
		}
		if err != nil {
			return nil, 0, err
		}
		blocks = append(blocks, b)
	}
	return blocks, out, nil
}

// Parameters concatenates the parameters of every layer in iteration order.
func (r *ResNet) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, r.Stem.Parameters()...)
	ps = append(ps, r.StemBN.Parameters()...)
	for _, stage := range r.Stages {
		for _, b := range stage {
			ps = append(ps, b.Parameters()...)
		}
	}
	ps = append(ps, r.FC.Parameters()...)
	return ps
}

// SetTraining cascades the train/eval flag to every BatchNorm.
func (r *ResNet) SetTraining(v bool) {
	r.StemBN.SetTraining(v)
	for _, stage := range r.Stages {
		for _, b := range stage {
			b.SetTraining(v)
		}
	}
}

// Forward runs the full network: stem → 4 stages → AdaptiveAvgPool → FC.
func (r *ResNet) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
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
	h, err = r.StemPool.Forward(h)
	if err != nil {
		return nil, err
	}
	for _, stage := range r.Stages {
		for _, b := range stage {
			h, err = b.Forward(h)
			if err != nil {
				return nil, err
			}
		}
	}
	h, err = r.Pool.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = (nn.Flatten{}).Forward(h)
	if err != nil {
		return nil, err
	}
	return r.FC.Forward(h)
}

// ParamCount sums the trainable element count across all params.
func (r *ResNet) ParamCount() int {
	total := 0
	for _, p := range r.Parameters() {
		total += p.NumElements()
	}
	return total
}

// ---- BasicBlock (ResNet-18 / 34) ----

// basicBlock implements the 2×3×3 residual block: Conv→BN→ReLU→Conv→BN + skip.
type basicBlock struct {
	Conv1, Conv2 *nn.Conv2d
	BN1, BN2     *nn.BatchNorm2d
	Down         *nn.Conv2d
	DownBN       *nn.BatchNorm2d
}

func newBasicBlock(cin, planes, stride int, dev device.Device) (*basicBlock, error) {
	c1, err := nn.NewConv2d(cin, planes, 3, 3, [2]int{stride, stride}, [2]int{1, 1}, false, dev)
	if err != nil {
		return nil, err
	}
	bn1, err := nn.NewBatchNorm2d(planes, 1e-5, 0.1, dev)
	if err != nil {
		return nil, err
	}
	c2, err := nn.NewConv2d(planes, planes, 3, 3, [2]int{1, 1}, [2]int{1, 1}, false, dev)
	if err != nil {
		return nil, err
	}
	bn2, err := nn.NewBatchNorm2d(planes, 1e-5, 0.1, dev)
	if err != nil {
		return nil, err
	}
	b := &basicBlock{Conv1: c1, BN1: bn1, Conv2: c2, BN2: bn2}
	if stride != 1 || cin != planes {
		d, err := nn.NewConv2d(cin, planes, 1, 1, [2]int{stride, stride}, [2]int{0, 0}, false, dev)
		if err != nil {
			return nil, err
		}
		dbn, err := nn.NewBatchNorm2d(planes, 1e-5, 0.1, dev)
		if err != nil {
			return nil, err
		}
		b.Down = d
		b.DownBN = dbn
	}
	return b, nil
}

func (b *basicBlock) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
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
	short := x
	if b.Down != nil {
		short, err = b.Down.Forward(x)
		if err != nil {
			return nil, err
		}
		short, err = b.DownBN.Forward(short)
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

func (b *basicBlock) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, b.Conv1.Parameters()...)
	ps = append(ps, b.BN1.Parameters()...)
	ps = append(ps, b.Conv2.Parameters()...)
	ps = append(ps, b.BN2.Parameters()...)
	if b.Down != nil {
		ps = append(ps, b.Down.Parameters()...)
		ps = append(ps, b.DownBN.Parameters()...)
	}
	return ps
}

func (b *basicBlock) SetTraining(v bool) {
	b.BN1.SetTraining(v)
	b.BN2.SetTraining(v)
	if b.DownBN != nil {
		b.DownBN.SetTraining(v)
	}
}

// ---- Bottleneck (ResNet-50 / 101 / 152) ----

// bottleneck implements the 3-conv bottleneck block:
//
//	1×1 (cin→planes) → BN → ReLU
//	3×3 (planes→planes, stride=stride) → BN → ReLU
//	1×1 (planes→planes*4) → BN
//	+ shortcut (1×1 stride=stride if shape changes)
type bottleneck struct {
	Conv1, Conv2, Conv3 *nn.Conv2d
	BN1, BN2, BN3       *nn.BatchNorm2d
	Down                *nn.Conv2d
	DownBN              *nn.BatchNorm2d
}

func newBottleneck(cin, planes, stride int, dev device.Device) (*bottleneck, error) {
	out := planes * 4
	c1, err := nn.NewConv2d(cin, planes, 1, 1, [2]int{1, 1}, [2]int{0, 0}, false, dev)
	if err != nil {
		return nil, err
	}
	bn1, err := nn.NewBatchNorm2d(planes, 1e-5, 0.1, dev)
	if err != nil {
		return nil, err
	}
	c2, err := nn.NewConv2d(planes, planes, 3, 3, [2]int{stride, stride}, [2]int{1, 1}, false, dev)
	if err != nil {
		return nil, err
	}
	bn2, err := nn.NewBatchNorm2d(planes, 1e-5, 0.1, dev)
	if err != nil {
		return nil, err
	}
	c3, err := nn.NewConv2d(planes, out, 1, 1, [2]int{1, 1}, [2]int{0, 0}, false, dev)
	if err != nil {
		return nil, err
	}
	bn3, err := nn.NewBatchNorm2d(out, 1e-5, 0.1, dev)
	if err != nil {
		return nil, err
	}
	b := &bottleneck{Conv1: c1, BN1: bn1, Conv2: c2, BN2: bn2, Conv3: c3, BN3: bn3}
	if stride != 1 || cin != out {
		d, err := nn.NewConv2d(cin, out, 1, 1, [2]int{stride, stride}, [2]int{0, 0}, false, dev)
		if err != nil {
			return nil, err
		}
		dbn, err := nn.NewBatchNorm2d(out, 1e-5, 0.1, dev)
		if err != nil {
			return nil, err
		}
		b.Down = d
		b.DownBN = dbn
	}
	return b, nil
}

func (b *bottleneck) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
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
	h, err = autograd.ReLU(h)
	if err != nil {
		return nil, err
	}
	h, err = b.Conv3.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = b.BN3.Forward(h)
	if err != nil {
		return nil, err
	}
	short := x
	if b.Down != nil {
		short, err = b.Down.Forward(x)
		if err != nil {
			return nil, err
		}
		short, err = b.DownBN.Forward(short)
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

func (b *bottleneck) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, b.Conv1.Parameters()...)
	ps = append(ps, b.BN1.Parameters()...)
	ps = append(ps, b.Conv2.Parameters()...)
	ps = append(ps, b.BN2.Parameters()...)
	ps = append(ps, b.Conv3.Parameters()...)
	ps = append(ps, b.BN3.Parameters()...)
	if b.Down != nil {
		ps = append(ps, b.Down.Parameters()...)
		ps = append(ps, b.DownBN.Parameters()...)
	}
	return ps
}

func (b *bottleneck) SetTraining(v bool) {
	b.BN1.SetTraining(v)
	b.BN2.SetTraining(v)
	b.BN3.SetTraining(v)
	if b.DownBN != nil {
		b.DownBN.SetTraining(v)
	}
}
