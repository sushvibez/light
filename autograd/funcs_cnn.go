package autograd

import (
	"fmt"

	"light/ops"
	"light/tensor"
)

// Conv2dFunction wraps ops.Conv2d. Accepts (input, weight) or (input, weight, bias).
type Conv2dFunction struct {
	Stride, Padding [2]int
}

func (Conv2dFunction) Name() string { return "Conv2d" }

func (f Conv2dFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) < 2 || len(inputs) > 3 {
		return nil, fmt.Errorf("Conv2d: need (input, weight[, bias])")
	}
	x, w := inputs[0], inputs[1]
	var b *tensor.Tensor
	if len(inputs) == 3 {
		b = inputs[2]
	}
	y, err := ops.Conv2d(x, w, b, f.Stride, f.Padding)
	if err != nil {
		return nil, err
	}
	ctx.SaveForBackward(x, w)
	ctx.Set("stride", f.Stride)
	ctx.Set("padding", f.Padding)
	ctx.Set("hasBias", b != nil)
	return []*tensor.Tensor{y}, nil
}

func (Conv2dFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	saved := ctx.Saved()
	x, w := saved[0], saved[1]
	strideAny, _ := ctx.Get("stride")
	padAny, _ := ctx.Get("padding")
	hasBAny, _ := ctx.Get("hasBias")
	stride := strideAny.([2]int)
	padding := padAny.([2]int)
	hasB := hasBAny.(bool)

	grads, err := ops.Conv2dBackward(x, w, g, stride, padding, true, true, hasB)
	if err != nil {
		return nil, err
	}
	if hasB {
		return []*tensor.Tensor{grads.GradInput, grads.GradWeight, grads.GradBias}, nil
	}
	return []*tensor.Tensor{grads.GradInput, grads.GradWeight}, nil
}

// Conv2d is the autograd-aware 2-D convolution. Pass bias=nil to skip bias.
func Conv2d(input, weight, bias *tensor.Tensor, stride, padding [2]int) (*tensor.Tensor, error) {
	fn := Conv2dFunction{Stride: stride, Padding: padding}
	ins := []*tensor.Tensor{input, weight}
	if bias != nil {
		ins = append(ins, bias)
	}
	outs, err := Apply(fn, ins)
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// Conv2dGroupedFunction is the autograd wrapper for grouped convolution.
type Conv2dGroupedFunction struct {
	Stride, Padding [2]int
	Groups          int
}

func (Conv2dGroupedFunction) Name() string { return "Conv2dGrouped" }

func (f Conv2dGroupedFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x, w := inputs[0], inputs[1]
	var b *tensor.Tensor
	if len(inputs) == 3 {
		b = inputs[2]
	}
	y, err := ops.Conv2dGrouped(x, w, b, f.Stride, f.Padding, f.Groups)
	if err != nil {
		return nil, err
	}
	ctx.SaveForBackward(x, w)
	ctx.Set("stride", f.Stride)
	ctx.Set("padding", f.Padding)
	ctx.Set("groups", f.Groups)
	ctx.Set("hasBias", b != nil)
	return []*tensor.Tensor{y}, nil
}

func (Conv2dGroupedFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	saved := ctx.Saved()
	x, w := saved[0], saved[1]
	strideAny, _ := ctx.Get("stride")
	padAny, _ := ctx.Get("padding")
	groupsAny, _ := ctx.Get("groups")
	hasBAny, _ := ctx.Get("hasBias")
	grads, err := ops.Conv2dGroupedBackward(x, w, g, strideAny.([2]int), padAny.([2]int),
		groupsAny.(int), true, true, hasBAny.(bool))
	if err != nil {
		return nil, err
	}
	if hasBAny.(bool) {
		return []*tensor.Tensor{grads.GradInput, grads.GradWeight, grads.GradBias}, nil
	}
	return []*tensor.Tensor{grads.GradInput, grads.GradWeight}, nil
}

// Conv2dGrouped is the autograd-aware grouped 2-D convolution. groups=1
// delegates to the regular Conv2d path.
func Conv2dGrouped(input, weight, bias *tensor.Tensor, stride, padding [2]int, groups int) (*tensor.Tensor, error) {
	if groups == 1 {
		return Conv2d(input, weight, bias, stride, padding)
	}
	fn := Conv2dGroupedFunction{Stride: stride, Padding: padding, Groups: groups}
	ins := []*tensor.Tensor{input, weight}
	if bias != nil {
		ins = append(ins, bias)
	}
	outs, err := Apply(fn, ins)
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// MaxPool2dFunction is the autograd wrapper around ops.MaxPool2d.
type MaxPool2dFunction struct {
	Kernel, Stride, Padding [2]int
}

func (MaxPool2dFunction) Name() string { return "MaxPool2d" }

func (f MaxPool2dFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := inputs[0]
	ctx.SaveForBackward(x)
	ctx.Set("kernel", f.Kernel)
	ctx.Set("stride", f.Stride)
	ctx.Set("padding", f.Padding)
	y, err := ops.MaxPool2d(x, f.Kernel, f.Stride, f.Padding)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{y}, nil
}

func (MaxPool2dFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	x := ctx.Saved()[0]
	kAny, _ := ctx.Get("kernel")
	sAny, _ := ctx.Get("stride")
	pAny, _ := ctx.Get("padding")
	gin, err := ops.MaxPool2dBackward(x, g, kAny.([2]int), sAny.([2]int), pAny.([2]int))
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gin}, nil
}

// MaxPool2d is the autograd-aware max pool.
func MaxPool2d(x *tensor.Tensor, kernel, stride, padding [2]int) (*tensor.Tensor, error) {
	outs, err := Apply(MaxPool2dFunction{Kernel: kernel, Stride: stride, Padding: padding}, []*tensor.Tensor{x})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// AvgPool2dFunction is the autograd wrapper around ops.AvgPool2d.
type AvgPool2dFunction struct {
	Kernel, Stride, Padding [2]int
}

func (AvgPool2dFunction) Name() string { return "AvgPool2d" }

func (f AvgPool2dFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := inputs[0]
	ctx.SaveForBackward(x)
	ctx.Set("kernel", f.Kernel)
	ctx.Set("stride", f.Stride)
	ctx.Set("padding", f.Padding)
	y, err := ops.AvgPool2d(x, f.Kernel, f.Stride, f.Padding)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{y}, nil
}

func (AvgPool2dFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	x := ctx.Saved()[0]
	kAny, _ := ctx.Get("kernel")
	sAny, _ := ctx.Get("stride")
	pAny, _ := ctx.Get("padding")
	gin, err := ops.AvgPool2dBackward(x, g, kAny.([2]int), sAny.([2]int), pAny.([2]int))
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gin}, nil
}

// AvgPool2d is the autograd-aware average pool.
func AvgPool2d(x *tensor.Tensor, kernel, stride, padding [2]int) (*tensor.Tensor, error) {
	outs, err := Apply(AvgPool2dFunction{Kernel: kernel, Stride: stride, Padding: padding}, []*tensor.Tensor{x})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// AdaptiveAvgPool2dFunction wraps ops.AdaptiveAvgPool2d.
type AdaptiveAvgPool2dFunction struct {
	OutH, OutW int
}

func (AdaptiveAvgPool2dFunction) Name() string { return "AdaptiveAvgPool2d" }

func (f AdaptiveAvgPool2dFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := inputs[0]
	ctx.SaveForBackward(x)
	ctx.Set("outH", f.OutH)
	ctx.Set("outW", f.OutW)
	y, err := ops.AdaptiveAvgPool2d(x, f.OutH, f.OutW)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{y}, nil
}

func (AdaptiveAvgPool2dFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	x := ctx.Saved()[0]
	ohAny, _ := ctx.Get("outH")
	owAny, _ := ctx.Get("outW")
	gin, err := ops.AdaptiveAvgPool2dBackward(x, g, ohAny.(int), owAny.(int))
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gin}, nil
}

// AdaptiveAvgPool2d is the autograd-aware adaptive average pool.
func AdaptiveAvgPool2d(x *tensor.Tensor, outH, outW int) (*tensor.Tensor, error) {
	outs, err := Apply(AdaptiveAvgPool2dFunction{OutH: outH, OutW: outW}, []*tensor.Tensor{x})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// BatchNorm2dFunction: differentiates training-mode BatchNorm2d.
//
// Inputs: (x, gamma, beta). Running stats live on the nn.BatchNorm2d module
// and are updated out-of-band by the caller (the module's Forward handles
// that). Autograd only propagates through x, gamma, beta.
type BatchNorm2dFunction struct {
	XHat, Sigma *tensor.Tensor // cached from forward
}

func (BatchNorm2dFunction) Name() string { return "BatchNorm2d" }

func (f *BatchNorm2dFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	// Forward is actually invoked via the module (which needs running stats
	// and the training flag). Callers here pass the already-computed output;
	// the cache (xHat, sigma) is supplied in-struct.
	return nil, fmt.Errorf("BatchNorm2dFunction: use autograd.ApplyBatchNorm2d instead")
}

func (f *BatchNorm2dFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	saved := ctx.Saved()
	gamma := saved[0]
	gIn, gGamma, gBeta, err := ops.BatchNorm2dBackward(g, f.XHat, f.Sigma, gamma)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gIn, gGamma, gBeta}, nil
}

// ApplyBatchNorm2d runs BatchNorm2d forward (training path), sets up a
// gradient node, and returns (y, newRunningMean, newRunningVar). `gamma` /
// `beta` are trainable parameters.
func ApplyBatchNorm2d(
	x, gamma, beta, runningMean, runningVar *tensor.Tensor,
	opts ops.BatchNormOptions,
) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	res, err := ops.BatchNorm2d(x, gamma, beta, runningMean, runningVar, opts)
	if err != nil {
		return nil, nil, nil, err
	}
	// Attach autograd only if any of x / gamma / beta require it.
	needsGrad := x.RequiresGrad() || x.GradFn() != nil ||
		gamma.RequiresGrad() || gamma.GradFn() != nil ||
		beta.RequiresGrad() || beta.GradFn() != nil
	if !needsGrad {
		return res.Out, res.RunningMean, res.RunningVar, nil
	}
	fn := &BatchNorm2dFunction{XHat: res.XHat, Sigma: res.Sigma}
	ctx := &Context{}
	ctx.SaveForBackward(gamma)
	node := &FunctionNode{
		fn:       fn,
		ctx:      ctx,
		inputs:   []*tensor.Tensor{x, gamma, beta},
		nOutputs: 1,
	}
	res.Out.SetRequiresGrad(true)
	res.Out.SetGradFn(node, 0)
	return res.Out, res.RunningMean, res.RunningVar, nil
}
