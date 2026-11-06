package models

import (
	"fmt"

	"light/autograd"
	"light/device"
	"light/nn"
	"light/tensor"
)

// VGG implements the Simonyan & Zisserman 2014 architecture, torchvision
// preset (all 4 depths). Input is 224×224 RGB; the feature extractor
// repeatedly applies Conv 3×3 / pad 1 + ReLU and MaxPool 2×2 / stride 2,
// followed by an AdaptiveAvgPool(7×7) and 3 fully-connected layers
// 512·7·7 → 4096 → 4096 → num_classes with Dropout(0.5) between FCs.
//
// VGG-16 has ~138M parameters; VGG-19 ~144M. The big FC1 (~102M params,
// 392 MB in F32) dominates memory.
type VGG struct {
	Features           []nn.Module
	Pool               *nn.AdaptiveAvgPool2d
	FC1, FC2, FC3      *nn.Linear
	Dropout1, Dropout2 *nn.Dropout
	training           bool
}

// VGGConfig captures one variant's per-stage Conv channel list. "0" entries
// are interpreted as MaxPool layers.
type VGGConfig []int

var (
	vgg11Cfg = VGGConfig{64, 0, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0}
	vgg13Cfg = VGGConfig{64, 64, 0, 128, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0}
	vgg16Cfg = VGGConfig{64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0}
	vgg19Cfg = VGGConfig{64, 64, 0, 128, 128, 0, 256, 256, 256, 256, 0, 512, 512, 512, 512, 0, 512, 512, 512, 512, 0}
)

// NewVGG11 / NewVGG13 / NewVGG16 / NewVGG19 build the corresponding torchvision variants.
func NewVGG11(numClasses int, dev device.Device) (*VGG, error) {
	return newVGG(vgg11Cfg, numClasses, dev)
}
func NewVGG13(numClasses int, dev device.Device) (*VGG, error) {
	return newVGG(vgg13Cfg, numClasses, dev)
}
func NewVGG16(numClasses int, dev device.Device) (*VGG, error) {
	return newVGG(vgg16Cfg, numClasses, dev)
}
func NewVGG19(numClasses int, dev device.Device) (*VGG, error) {
	return newVGG(vgg19Cfg, numClasses, dev)
}

func newVGG(cfg VGGConfig, numClasses int, dev device.Device) (*VGG, error) {
	if numClasses <= 0 {
		return nil, fmt.Errorf("models.VGG: numClasses must be positive")
	}
	feats := make([]nn.Module, 0, len(cfg)*2)
	cin := 3
	for _, c := range cfg {
		if c == 0 {
			feats = append(feats, nn.NewMaxPool2d([2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0}))
			continue
		}
		conv, err := nn.NewConv2d(cin, c, 3, 3, [2]int{1, 1}, [2]int{1, 1}, true, dev)
		if err != nil {
			return nil, err
		}
		feats = append(feats, conv, vggReLU{})
		cin = c
	}
	pool := nn.NewAdaptiveAvgPool2d(7, 7)
	fc1, err := nn.NewLinear(512*7*7, 4096, true, dev)
	if err != nil {
		return nil, err
	}
	fc2, err := nn.NewLinear(4096, 4096, true, dev)
	if err != nil {
		return nil, err
	}
	fc3, err := nn.NewLinear(4096, numClasses, true, dev)
	if err != nil {
		return nil, err
	}
	d1 := nn.NewDropout(0.5)
	d2 := nn.NewDropout(0.5)
	v := &VGG{Features: feats, Pool: pool, FC1: fc1, FC2: fc2, FC3: fc3, Dropout1: d1, Dropout2: d2, training: true}
	v.SetTraining(false) // default to eval mode
	return v, nil
}

// vggReLU is a tiny adapter that lets us put ReLU in the Features slice.
type vggReLU struct{}

func (vggReLU) Forward(x *tensor.Tensor) (*tensor.Tensor, error) { return autograd.ReLU(x) }
func (vggReLU) Parameters() []*tensor.Tensor                     { return nil }

// Forward runs feature extractor → AdaptiveAvgPool → 3-layer FC head.
func (v *VGG) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	h := x
	var err error
	for _, m := range v.Features {
		h, err = m.Forward(h)
		if err != nil {
			return nil, err
		}
	}
	h, err = v.Pool.Forward(h) // (N, 512, 7, 7)
	if err != nil {
		return nil, err
	}
	h, err = (nn.Flatten{}).Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = v.FC1.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = autograd.ReLU(h)
	if err != nil {
		return nil, err
	}
	h, err = v.Dropout1.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = v.FC2.Forward(h)
	if err != nil {
		return nil, err
	}
	h, err = autograd.ReLU(h)
	if err != nil {
		return nil, err
	}
	h, err = v.Dropout2.Forward(h)
	if err != nil {
		return nil, err
	}
	return v.FC3.Forward(h)
}

func (v *VGG) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, m := range v.Features {
		ps = append(ps, m.Parameters()...)
	}
	ps = append(ps, v.FC1.Parameters()...)
	ps = append(ps, v.FC2.Parameters()...)
	ps = append(ps, v.FC3.Parameters()...)
	return ps
}

func (v *VGG) SetTraining(t bool) {
	v.training = t
	v.Dropout1.SetTraining(t)
	v.Dropout2.SetTraining(t)
}

func (v *VGG) ParamCount() int {
	total := 0
	for _, p := range v.Parameters() {
		total += p.NumElements()
	}
	return total
}
