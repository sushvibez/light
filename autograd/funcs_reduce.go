package autograd

import (
	"fmt"

	"light/dtype"
	"light/ops"
	"light/tensor"
)

// SumFunction implements reduce-all Sum (no axes, no keepdims). Backward
// broadcasts the scalar gradient back to the input shape.
type SumFunction struct{}

func (SumFunction) Name() string { return "Sum" }

func (SumFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	a := inputs[0]
	ctx.Set("shape", a.Shape().Dims())
	out, err := ops.Sum(a, ops.ReduceOptions{})
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{out}, nil
}

func (SumFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	sh, ok := ctx.Get("shape")
	if !ok {
		return nil, fmt.Errorf("Sum backward: missing shape")
	}
	shape := sh.([]int)
	ones, err := fullLike(1, shape, g)
	if err != nil {
		return nil, err
	}
	gin, err := ops.Mul(ones, g) // broadcasts scalar g across ones
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gin}, nil
}

// Sum is the autograd-aware reduce-all sum.
func Sum(a *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(SumFunction{}, []*tensor.Tensor{a})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// MeanFunction implements reduce-all Mean. Backward = grad * (1/n) * ones.
type MeanFunction struct{}

func (MeanFunction) Name() string { return "Mean" }

func (MeanFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	a := inputs[0]
	ctx.Set("shape", a.Shape().Dims())
	ctx.Set("n", a.NumElements())
	out, err := ops.Mean(a, ops.ReduceOptions{})
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{out}, nil
}

func (MeanFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	sh, _ := ctx.Get("shape")
	n, _ := ctx.Get("n")
	shape := sh.([]int)
	var scale *tensor.Tensor
	var err error
	if g.DType() == dtype.F32 {
		// Keep the F32 path bit-identical: 1/n rounded in float32.
		scale, err = tensor.Full(float32(1.0)/float32(n.(int)), shape, g.Device())
	} else {
		scale, err = fullLike(1/float64(n.(int)), shape, g)
	}
	if err != nil {
		return nil, err
	}
	gin, err := ops.Mul(scale, g)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gin}, nil
}

func Mean(a *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(MeanFunction{}, []*tensor.Tensor{a})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// AxisSumFunction reduces a tensor by summing along one or more axes,
// optionally keeping the reduced axes as size-1.
//
// Backward: broadcast grad_out back to the input shape (after first
// re-inserting size-1 axes if KeepDims=false).
type AxisSumFunction struct {
	axes     []int
	keepDims bool
}

func (AxisSumFunction) Name() string { return "AxisSum" }

func (f AxisSumFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := inputs[0]
	ctx.Set("x_shape", x.Shape().Dims())
	ctx.Set("axes", f.axes)
	ctx.Set("keepDims", f.keepDims)
	y, err := ops.Sum(x, ops.ReduceOptions{Axes: f.axes, KeepDims: f.keepDims})
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{y}, nil
}

func (AxisSumFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	xShapeAny, _ := ctx.Get("x_shape")
	axesAny, _ := ctx.Get("axes")
	keepDimsAny, _ := ctx.Get("keepDims")
	xShape := xShapeAny.([]int)
	axes := axesAny.([]int)
	keepDims := keepDimsAny.(bool)

	// If KeepDims was false, re-insert size-1 axes so `g` has the same rank
	// as the input; then ops.Mul broadcasts it across those axes.
	if !keepDims && len(axes) > 0 {
		norm, err := normalizeAxes(axes, len(xShape))
		if err != nil {
			return nil, fmt.Errorf("AxisSum backward: %w", err)
		}
		kept := make([]int, len(xShape))
		copy(kept, xShape)
		for _, a := range norm {
			kept[a] = 1
		}
		cont, err := g.Contiguous()
		if err != nil {
			return nil, err
		}
		g, err = cont.Reshape(kept...)
		if err != nil {
			return nil, err
		}
	}
	ones, err := fullLike(1, xShape, g)
	if err != nil {
		return nil, err
	}
	gin, err := ops.Mul(ones, g)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gin}, nil
}

// AxisSum is the autograd-aware axis-reducing sum.
func AxisSum(x *tensor.Tensor, axes []int, keepDims bool) (*tensor.Tensor, error) {
	outs, err := Apply(AxisSumFunction{axes: axes, keepDims: keepDims}, []*tensor.Tensor{x})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// AxisMean is autograd-aware mean reduction over the given axes.
// Implemented as AxisSum divided by the product of the reduced axis sizes.
// Empty axes reduce over every axis, as in AxisSum.
func AxisMean(x *tensor.Tensor, axes []int, keepDims bool) (*tensor.Tensor, error) {
	dims := x.Shape().Dims()
	norm, err := normalizeAxes(axes, len(dims))
	if err != nil {
		return nil, fmt.Errorf("AxisMean: %w", err)
	}
	n := 1
	for _, a := range norm {
		n *= dims[a]
	}
	s, err := AxisSum(x, axes, keepDims)
	if err != nil {
		return nil, err
	}
	var inv *tensor.Tensor
	if x.DType() == dtype.F32 {
		inv, err = tensor.Full(1.0/float32(n), []int{1}, x.Device())
	} else {
		inv, err = fullLike(1/float64(n), []int{1}, x)
	}
	if err != nil {
		return nil, err
	}
	return Mul(s, inv)
}

// normalizeAxes resolves negative axes against rank and rejects
// out-of-range or duplicate entries. Empty axes mean every axis.
func normalizeAxes(axes []int, rank int) ([]int, error) {
	if len(axes) == 0 {
		all := make([]int, rank)
		for i := range all {
			all[i] = i
		}
		return all, nil
	}
	out := make([]int, len(axes))
	seen := make(map[int]bool, len(axes))
	for i, a := range axes {
		na := a
		if na < 0 {
			na += rank
		}
		if na < 0 || na >= rank {
			return nil, fmt.Errorf("axis %d out of range for rank %d", a, rank)
		}
		if seen[na] {
			return nil, fmt.Errorf("duplicate axis %d", a)
		}
		seen[na] = true
		out[i] = na
	}
	return out, nil
}
