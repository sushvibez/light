package models

import (
	"fmt"

	"light/autograd"
	"light/device"
	"light/nn"
	"light/tensor"
)

// ResNet9 is the wide shallow ResNet popularised by the fastai
// "DAWNBench" CIFAR-10 super-convergence run and the Kaggle notebook
// "CIFAR10 ResNet 90% Accuracy Less Than 5 Min". Designed to hit 90+%
// test accuracy on CIFAR-10 in under 10 epochs:
//
//	conv 3→64,   k3 p1, BN, ReLU                           # 32×32
//	conv 64→128, k3 p1, BN, ReLU, MaxPool(2)               # 16×16
//	residual: conv 128→128 + BN + ReLU + conv 128→128 + BN + ReLU  (skip)
//	conv 128→256,k3 p1, BN, ReLU, MaxPool(2)               # 8×8
//	conv 256→512,k3 p1, BN, ReLU, MaxPool(2)               # 4×4
//	residual: conv 512→512 + BN + ReLU + conv 512→512 + BN + ReLU  (skip)
//	MaxPool(4)                                             # 1×1
//	Flatten → Linear(512, num_classes)
//
// ~6.5M parameters — about 24× our `ResNetCIFAR(n=3)` (270k) at the cost
// of more FLOPs, but still well within an 8 GB GPU.
type ResNet9 struct {
	Conv1 *resnet9Conv
	Conv2 *resnet9Conv
	Pool2 *nn.MaxPool2d
	Res1A *resnet9Conv
	Res1B *resnet9Conv
	Conv3 *resnet9Conv
	Pool3 *nn.MaxPool2d
	Conv4 *resnet9Conv
	Pool4 *nn.MaxPool2d
	Res2A *resnet9Conv
	Res2B *resnet9Conv
	PoolF *nn.MaxPool2d
	FC    *nn.Linear

	training bool
}

// resnet9Conv bundles Conv2d + BN as a single module since every conv in
// ResNet9 is followed by BN+ReLU.
type resnet9Conv struct {
	Conv *nn.Conv2d
	BN   *nn.BatchNorm2d
}

func newResnet9Conv(cin, cout int, dev device.Device) (*resnet9Conv, error) {
	c, err := nn.NewConv2d(cin, cout, 3, 3, [2]int{1, 1}, [2]int{1, 1}, false, dev)
	if err != nil {
		return nil, err
	}
	bn, err := nn.NewBatchNorm2d(cout, 1e-5, 0.1, dev)
	if err != nil {
		return nil, err
	}
	return &resnet9Conv{Conv: c, BN: bn}, nil
}

// forward runs Conv → BN → ReLU.
func (c *resnet9Conv) forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := c.Conv.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = c.BN.Forward(h)
	if err != nil {
		return nil, err
	}
	return autograd.ReLU(h)
}

func (c *resnet9Conv) parameters() []*tensor.Tensor {
	return append(c.Conv.Parameters(), c.BN.Parameters()...)
}

// NewResNet9 constructs the network with random initialisation.
func NewResNet9(numClasses int, dev device.Device) (*ResNet9, error) {
	if numClasses <= 0 {
		return nil, fmt.Errorf("models.ResNet9: numClasses must be positive")
	}
	c1, err := newResnet9Conv(3, 64, dev)
	if err != nil {
		return nil, err
	}
	c2, err := newResnet9Conv(64, 128, dev)
	if err != nil {
		return nil, err
	}
	r1a, err := newResnet9Conv(128, 128, dev)
	if err != nil {
		return nil, err
	}
	r1b, err := newResnet9Conv(128, 128, dev)
	if err != nil {
		return nil, err
	}
	c3, err := newResnet9Conv(128, 256, dev)
	if err != nil {
		return nil, err
	}
	c4, err := newResnet9Conv(256, 512, dev)
	if err != nil {
		return nil, err
	}
	r2a, err := newResnet9Conv(512, 512, dev)
	if err != nil {
		return nil, err
	}
	r2b, err := newResnet9Conv(512, 512, dev)
	if err != nil {
		return nil, err
	}
	fc, err := nn.NewLinear(512, numClasses, true, dev)
	if err != nil {
		return nil, err
	}
	return &ResNet9{
		Conv1: c1, Conv2: c2, Pool2: nn.NewMaxPool2d([2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0}),
		Res1A: r1a, Res1B: r1b,
		Conv3: c3, Pool3: nn.NewMaxPool2d([2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0}),
		Conv4: c4, Pool4: nn.NewMaxPool2d([2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0}),
		Res2A: r2a, Res2B: r2b,
		PoolF:    nn.NewMaxPool2d([2]int{4, 4}, [2]int{4, 4}, [2]int{0, 0}),
		FC:       fc,
		training: true,
	}, nil
}

// Forward runs the full ResNet9 over a (B, 3, 32, 32) batch.
func (r *ResNet9) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := r.Conv1.forward(x) // 32×32, 64ch
	if err != nil {
		return nil, err
	}
	h, err = r.Conv2.forward(h) // 32×32, 128ch
	if err != nil {
		return nil, err
	}
	h, err = r.Pool2.Forward(h) // 16×16
	if err != nil {
		return nil, err
	}
	// Residual block 1: skip = h, body = Res1A → Res1B
	skip := h
	h, err = r.Res1A.forward(h)
	if err != nil {
		return nil, err
	}
	h, err = r.Res1B.forward(h)
	if err != nil {
		return nil, err
	}
	h, err = autograd.Add(h, skip)
	if err != nil {
		return nil, err
	}
	h, err = r.Conv3.forward(h) // 16×16, 256ch
	if err != nil {
		return nil, err
	}
	h, err = r.Pool3.Forward(h) // 8×8
	if err != nil {
		return nil, err
	}
	h, err = r.Conv4.forward(h) // 8×8, 512ch
	if err != nil {
		return nil, err
	}
	h, err = r.Pool4.Forward(h) // 4×4
	if err != nil {
		return nil, err
	}
	skip = h
	h, err = r.Res2A.forward(h)
	if err != nil {
		return nil, err
	}
	h, err = r.Res2B.forward(h)
	if err != nil {
		return nil, err
	}
	h, err = autograd.Add(h, skip)
	if err != nil {
		return nil, err
	}
	h, err = r.PoolF.Forward(h) // 1×1
	if err != nil {
		return nil, err
	}
	h, err = (nn.Flatten{}).Forward(h)
	if err != nil {
		return nil, err
	}
	return r.FC.Forward(h)
}

// Parameters concatenates every learnable tensor in iteration order.
func (r *ResNet9) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, c := range []*resnet9Conv{r.Conv1, r.Conv2, r.Res1A, r.Res1B, r.Conv3, r.Conv4, r.Res2A, r.Res2B} {
		ps = append(ps, c.parameters()...)
	}
	ps = append(ps, r.FC.Parameters()...)
	return ps
}

// SetTraining cascades the train/eval flag to every BatchNorm.
func (r *ResNet9) SetTraining(v bool) {
	r.training = v
	for _, c := range []*resnet9Conv{r.Conv1, r.Conv2, r.Res1A, r.Res1B, r.Conv3, r.Conv4, r.Res2A, r.Res2B} {
		c.BN.SetTraining(v)
	}
}

// ParamCount sums the trainable element count across all params.
func (r *ResNet9) ParamCount() int {
	total := 0
	for _, p := range r.Parameters() {
		total += p.NumElements()
	}
	return total
}
