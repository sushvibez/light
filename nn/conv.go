package nn

import (
	"fmt"
	"math"
	"unsafe"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// Conv2d is a trainable 2-D convolution module.
//
//	Weight: (Cout, Cin, kH, kW)
//	Bias:   (Cout,) — nil if no bias
type Conv2d struct {
	Weight          *tensor.Tensor
	Bias            *tensor.Tensor
	stride, padding [2]int
	cin, cout       int
	kH, kW          int
}

// NewConv2d constructs a Conv2d with Kaiming-uniform init (fan-in) and F32
// weights. See NewConv2dWithDType for BF16 / mixed-precision construction.
func NewConv2d(cin, cout, kH, kW int, stride, padding [2]int, bias bool, dev device.Device) (*Conv2d, error) {
	return newConv2dImpl(cin, cout, kH, kW, stride, padding, bias, dtype.F32, dev, true)
}

// NewConv2dWithDType constructs a Conv2d with weights stored at `wdt`. Bias
// matches the weight dtype — caller is expected to land everything at one
// dtype at load time so the inference path stays single-dtype.
func NewConv2dWithDType(cin, cout, kH, kW int, stride, padding [2]int, bias bool, wdt dtype.DType, dev device.Device) (*Conv2d, error) {
	return newConv2dImpl(cin, cout, kH, kW, stride, padding, bias, wdt, dev, true)
}

// NewConv2dUninit allocates the weight tensor with zeros (no Kaiming init).
// Use when a loader is about to overwrite values; saves the per-tensor
// random draw on CPU.
func NewConv2dUninit(cin, cout, kH, kW int, stride, padding [2]int, bias bool, wdt dtype.DType, dev device.Device) (*Conv2d, error) {
	return newConv2dImpl(cin, cout, kH, kW, stride, padding, bias, wdt, dev, false)
}

func newConv2dImpl(cin, cout, kH, kW int, stride, padding [2]int, bias bool, wdt dtype.DType, dev device.Device, kaiming bool) (*Conv2d, error) {
	if cin <= 0 || cout <= 0 || kH <= 0 || kW <= 0 {
		return nil, fmt.Errorf("nn.Conv2d: positive dims required")
	}
	var w *tensor.Tensor
	var err error
	if kaiming {
		fanIn := float64(cin * kH * kW)
		k := float32(1.0 / math.Sqrt(fanIn))
		var wCPU *tensor.Tensor
		wCPU, err = tensor.Rand(dtype.F32, []int{cout, cin, kH, kW}, device.CPU())
		if err != nil {
			return nil, err
		}
		scaleU := 2 * k
		buf := wCPU.Storage().CPUBytes()
		vals := unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), len(buf)/4)
		for i := range vals {
			vals[i] = vals[i]*scaleU - k
		}
		if wdt != dtype.F32 {
			wCPU, err = ops.Cast(wCPU, wdt)
			if err != nil {
				return nil, err
			}
		}
		w = wCPU
		if !dev.SameAs(device.CPU()) {
			w, err = wCPU.To(dev)
			if err != nil {
				return nil, err
			}
		}
	} else {
		w, err = tensor.Zeros(wdt, []int{cout, cin, kH, kW}, dev)
		if err != nil {
			return nil, err
		}
	}
	c := &Conv2d{Weight: w, stride: stride, padding: padding, cin: cin, cout: cout, kH: kH, kW: kW}
	if bias {
		b, err := tensor.Zeros(wdt, []int{cout}, dev)
		if err != nil {
			return nil, err
		}
		c.Bias = b
	}
	return c, nil
}

func (c *Conv2d) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	return autograd.Conv2d(x, c.Weight, c.Bias, c.stride, c.padding)
}

func (c *Conv2d) Parameters() []*tensor.Tensor {
	if c.Bias != nil {
		return []*tensor.Tensor{c.Weight, c.Bias}
	}
	return []*tensor.Tensor{c.Weight}
}

// GroupedConv2d is a 2-D convolution with `groups > 1`.
//
//	Weight shape: (Cout, Cin/groups, kH, kW)
//	Depthwise:    groups == Cin == Cout
type GroupedConv2d struct {
	Weight          *tensor.Tensor
	Bias            *tensor.Tensor
	stride, padding [2]int
	cin, cout       int
	kH, kW          int
	groups          int
}

// NewGroupedConv2d constructs a grouped convolution. Use `groups=Cin=Cout`
// for depthwise convolutions (MobileNet-style).
func NewGroupedConv2d(cin, cout, kH, kW int, stride, padding [2]int, groups int, bias bool, dev device.Device) (*GroupedConv2d, error) {
	if groups <= 0 {
		return nil, fmt.Errorf("nn.GroupedConv2d: groups must be positive")
	}
	if cin%groups != 0 || cout%groups != 0 {
		return nil, fmt.Errorf("nn.GroupedConv2d: Cin=%d, Cout=%d must be divisible by groups=%d", cin, cout, groups)
	}
	cinPerG := cin / groups
	fanIn := float64(cinPerG * kH * kW)
	k := float32(1.0 / math.Sqrt(fanIn))
	w, err := tensor.Rand(dtype.F32, []int{cout, cinPerG, kH, kW}, device.CPU())
	if err != nil {
		return nil, err
	}
	buf := w.Storage().CPUBytes()
	vals := unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), len(buf)/4)
	for i := range vals {
		vals[i] = vals[i]*2*k - k
	}
	if !dev.SameAs(device.CPU()) {
		w, err = w.To(dev)
		if err != nil {
			return nil, err
		}
	}
	c := &GroupedConv2d{
		Weight: w, stride: stride, padding: padding, groups: groups,
		cin: cin, cout: cout, kH: kH, kW: kW,
	}
	if bias {
		b, err := tensor.Zeros(dtype.F32, []int{cout}, dev)
		if err != nil {
			return nil, err
		}
		c.Bias = b
	}
	return c, nil
}

// NewDepthwiseConv2d constructs a depthwise convolution (groups = channels).
// Requires Cin == Cout.
func NewDepthwiseConv2d(channels, kH, kW int, stride, padding [2]int, bias bool, dev device.Device) (*GroupedConv2d, error) {
	return NewGroupedConv2d(channels, channels, kH, kW, stride, padding, channels, bias, dev)
}

func (c *GroupedConv2d) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	return autograd.Conv2dGrouped(x, c.Weight, c.Bias, c.stride, c.padding, c.groups)
}

func (c *GroupedConv2d) Parameters() []*tensor.Tensor {
	if c.Bias != nil {
		return []*tensor.Tensor{c.Weight, c.Bias}
	}
	return []*tensor.Tensor{c.Weight}
}

// MaxPool2d module.
type MaxPool2d struct {
	kernel, stride, padding [2]int
}

func NewMaxPool2d(kernel, stride, padding [2]int) *MaxPool2d {
	return &MaxPool2d{kernel: kernel, stride: stride, padding: padding}
}

func (m *MaxPool2d) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	return autograd.MaxPool2d(x, m.kernel, m.stride, m.padding)
}
func (m *MaxPool2d) Parameters() []*tensor.Tensor { return nil }

// AvgPool2d module.
type AvgPool2d struct {
	kernel, stride, padding [2]int
}

func NewAvgPool2d(kernel, stride, padding [2]int) *AvgPool2d {
	return &AvgPool2d{kernel: kernel, stride: stride, padding: padding}
}

func (a *AvgPool2d) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	return autograd.AvgPool2d(x, a.kernel, a.stride, a.padding)
}
func (a *AvgPool2d) Parameters() []*tensor.Tensor { return nil }

// AdaptiveAvgPool2d pools any (N, C, H, W) input to (N, C, OutH, OutW) by
// averaging adaptive windows — ResNet uses (1, 1) as its global-average pool.
type AdaptiveAvgPool2d struct {
	outH, outW int
}

// NewAdaptiveAvgPool2d builds an adaptive-avg-pool module producing
// (outH, outW) spatial dims.
func NewAdaptiveAvgPool2d(outH, outW int) *AdaptiveAvgPool2d {
	return &AdaptiveAvgPool2d{outH: outH, outW: outW}
}

func (a *AdaptiveAvgPool2d) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	return autograd.AdaptiveAvgPool2d(x, a.outH, a.outW)
}
func (a *AdaptiveAvgPool2d) Parameters() []*tensor.Tensor { return nil }

// Flatten reshapes (N, ...) → (N, K) where K = product of trailing dims.
// Useful between conv/pool stacks and Linear classifiers.
type Flatten struct{}

func (Flatten) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Rank() < 2 {
		return x, nil
	}
	n := x.Dim(0)
	rest := 1
	for i := 1; i < x.Rank(); i++ {
		rest *= x.Dim(i)
	}
	return autograd.Reshape(x, n, rest)
}

func (Flatten) Parameters() []*tensor.Tensor { return nil }

// ConvTranspose2d is the transposed (fractionally strided) convolution
// module with PyTorch's weight layout (Cin, Cout/groups, kH, kW), so
// `nn.ConvTranspose2d` checkpoints (SD VAE decoders, U-Net up blocks,
// segmentation heads) load unchanged. Inference-only: Forward goes through
// ops.ConvTranspose2d without an autograd function.
type ConvTranspose2d struct {
	Weight *tensor.Tensor // (Cin, Cout/groups, kH, kW)
	Bias   *tensor.Tensor // (Cout,) or nil
	Opts   ops.ConvTranspose2dOpts
}

// NewConvTranspose2dUninit allocates zeroed weights for a loader to fill.
func NewConvTranspose2dUninit(cin, cout, kH, kW int, opts ops.ConvTranspose2dOpts, bias bool, wdt dtype.DType, dev device.Device) (*ConvTranspose2d, error) {
	groups := max(opts.Groups, 1)
	if cin <= 0 || cout <= 0 || kH <= 0 || kW <= 0 || cout%groups != 0 {
		return nil, fmt.Errorf("nn.ConvTranspose2d: bad dims (cin %d, cout %d, k %dx%d, groups %d)", cin, cout, kH, kW, groups)
	}
	w, err := tensor.Zeros(wdt, []int{cin, cout / groups, kH, kW}, dev)
	if err != nil {
		return nil, err
	}
	m := &ConvTranspose2d{Weight: w, Opts: opts}
	if bias {
		if m.Bias, err = tensor.Zeros(wdt, []int{cout}, dev); err != nil {
			return nil, err
		}
	}
	return m, nil
}

func (c *ConvTranspose2d) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	return ops.ConvTranspose2d(x, c.Weight, c.Bias, c.Opts)
}

func (c *ConvTranspose2d) Parameters() []*tensor.Tensor {
	if c.Bias != nil {
		return []*tensor.Tensor{c.Weight, c.Bias}
	}
	return []*tensor.Tensor{c.Weight}
}
