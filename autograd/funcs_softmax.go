package autograd

import (
	"light/ops"
	"light/tensor"
)

// SoftmaxFunction applies softmax along a given axis.
//
// Backward: grad_x = y * (grad_out - sum(grad_out * y, axis, keepdims=True))
type SoftmaxFunction struct {
	axis int
}

func (SoftmaxFunction) Name() string { return "Softmax" }

func (f SoftmaxFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	y, err := ops.Softmax(inputs[0], f.axis)
	if err != nil {
		return nil, err
	}
	ctx.SaveForBackward(y)
	ctx.Set("axis", f.axis)
	return []*tensor.Tensor{y}, nil
}

func (SoftmaxFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	y := ctx.Saved()[0]
	axisAny, _ := ctx.Get("axis")
	axis := axisAny.(int)
	gy, err := ops.Mul(g, y)
	if err != nil {
		return nil, err
	}
	s, err := ops.Sum(gy, ops.ReduceOptions{Axes: []int{axis}, KeepDims: true})
	if err != nil {
		return nil, err
	}
	gMinusS, err := ops.Sub(g, s)
	if err != nil {
		return nil, err
	}
	gx, err := ops.Mul(y, gMinusS)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gx}, nil
}

// Softmax is the autograd-aware softmax along `axis`.
func Softmax(a *tensor.Tensor, axis int) (*tensor.Tensor, error) {
	outs, err := Apply(SoftmaxFunction{axis: axis}, []*tensor.Tensor{a})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// LogSoftmaxFunction applies log-softmax along a given axis.
//
// Backward: grad_x = grad_out - exp(y) * sum(grad_out, axis, keepdims=True)
// where y = logsoftmax(x) and exp(y) = softmax(x).
type LogSoftmaxFunction struct {
	axis int
}

func (LogSoftmaxFunction) Name() string { return "LogSoftmax" }

func (f LogSoftmaxFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	y, err := ops.LogSoftmax(inputs[0], f.axis)
	if err != nil {
		return nil, err
	}
	ctx.SaveForBackward(y)
	ctx.Set("axis", f.axis)
	return []*tensor.Tensor{y}, nil
}

func (LogSoftmaxFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	y := ctx.Saved()[0]
	axisAny, _ := ctx.Get("axis")
	axis := axisAny.(int)
	sm, err := ops.Exp(y)
	if err != nil {
		return nil, err
	}
	sumG, err := ops.Sum(g, ops.ReduceOptions{Axes: []int{axis}, KeepDims: true})
	if err != nil {
		return nil, err
	}
	smTimesSum, err := ops.Mul(sm, sumG)
	if err != nil {
		return nil, err
	}
	gx, err := ops.Sub(g, smTimesSum)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gx}, nil
}

// LogSoftmax is the autograd-aware log-softmax along `axis`.
func LogSoftmax(a *tensor.Tensor, axis int) (*tensor.Tensor, error) {
	outs, err := Apply(LogSoftmaxFunction{axis: axis}, []*tensor.Tensor{a})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}
