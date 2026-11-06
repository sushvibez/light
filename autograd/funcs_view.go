package autograd

import (
	"fmt"

	"light/ops"
	"light/tensor"
)

// ReshapeFunction changes a tensor's logical shape without touching storage
// (post-Contiguous'd). Backward reshapes the gradient back to the input shape.
type ReshapeFunction struct {
	shape []int
}

func (ReshapeFunction) Name() string { return "Reshape" }

func (f ReshapeFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := inputs[0]
	ctx.Set("input_shape", x.Shape().Dims())
	cont, err := x.Contiguous()
	if err != nil {
		return nil, err
	}
	y, err := cont.Reshape(f.shape...)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{y}, nil
}

func (ReshapeFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	orig, _ := ctx.Get("input_shape")
	cont, err := g.Contiguous()
	if err != nil {
		return nil, err
	}
	gin, err := cont.Reshape(orig.([]int)...)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gin}, nil
}

// Reshape is the autograd-aware shape change.
func Reshape(x *tensor.Tensor, shape ...int) (*tensor.Tensor, error) {
	outs, err := Apply(ReshapeFunction{shape: shape}, []*tensor.Tensor{x})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// TransposeFunction swaps axes i and j. Self-inverse.
type TransposeFunction struct {
	i, j int
}

func (TransposeFunction) Name() string { return "Transpose" }

func (f TransposeFunction) Forward(_ *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	y, err := inputs[0].Transpose(f.i, f.j)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{y}, nil
}

func (f TransposeFunction) Backward(_ *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	gin, err := gradOutputs[0].Transpose(f.i, f.j)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gin}, nil
}

// Transpose is the autograd-aware axis swap.
func Transpose(x *tensor.Tensor, i, j int) (*tensor.Tensor, error) {
	outs, err := Apply(TransposeFunction{i: i, j: j}, []*tensor.Tensor{x})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// PermuteFunction reorders tensor axes. Backward permutes the gradient by
// the inverse permutation.
type PermuteFunction struct {
	perm []int
}

func (PermuteFunction) Name() string { return "Permute" }

func (f PermuteFunction) Forward(_ *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	y, err := inputs[0].Permute(f.perm...)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{y}, nil
}

func (f PermuteFunction) Backward(_ *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	inv := make([]int, len(f.perm))
	for i, p := range f.perm {
		inv[p] = i
	}
	gin, err := gradOutputs[0].Permute(inv...)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gin}, nil
}

// Permute is the autograd-aware axis reorder.
func Permute(x *tensor.Tensor, perm ...int) (*tensor.Tensor, error) {
	outs, err := Apply(PermuteFunction{perm: perm}, []*tensor.Tensor{x})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// ContiguousFunction materialises a view into a dense, default-strided
// tensor. Shape is unchanged, so the gradient passes through unmodified.
type ContiguousFunction struct{}

func (ContiguousFunction) Name() string { return "Contiguous" }

func (ContiguousFunction) Forward(_ *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	y, err := inputs[0].Contiguous()
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{y}, nil
}

func (ContiguousFunction) Backward(_ *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	return []*tensor.Tensor{gradOutputs[0]}, nil
}

// Contiguous is the autograd-aware force-contiguify. If the input is
// already contiguous with zero offset, returns it unchanged — Apply would
// otherwise attach a new gradFn to the shared tensor and clobber the
// caller's graph.
func Contiguous(x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.IsContiguous() && x.Offset() == 0 {
		return x, nil
	}
	outs, err := Apply(ContiguousFunction{}, []*tensor.Tensor{x})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// ---- Narrow / Concat views with gradients (needed by RNN unrolling) ----
type narrowFn struct{ dim, start, length int }

func (narrowFn) Name() string { return "Narrow" }

func (f narrowFn) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := inputs[0]
	d, err := x.Shape().NormalizeAxis(f.dim)
	if err != nil {
		return nil, fmt.Errorf("Narrow: %w", err)
	}
	ctx.Set("dim", d)
	ctx.Set("n", x.Dim(d))
	v, err := x.Narrow(d, f.start, f.length)
	if err != nil {
		return nil, err
	}
	out, err := v.Contiguous()
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{out}, nil
}

func (f narrowFn) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	d, _ := ctx.Get("dim")
	n, _ := ctx.Get("n")
	dim, total := d.(int), n.(int)
	g, err := ops.Pad(gradOutputs[0], dim, f.start, total-f.start-f.length)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{g}, nil
}

// Narrow is a differentiable slice of `length` elements from `start` along
// dim; the gradient is zero-padded back to the input shape.
func Narrow(x *tensor.Tensor, dim, start, length int) (*tensor.Tensor, error) {
	outs, err := Apply(narrowFn{dim, start, length}, []*tensor.Tensor{x})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

type concatFn struct{ axis int }

func (concatFn) Name() string { return "Concat" }

func (f concatFn) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	d, err := inputs[0].Shape().NormalizeAxis(f.axis)
	if err != nil {
		return nil, fmt.Errorf("Concat: %w", err)
	}
	sizes := make([]int, len(inputs))
	for i, in := range inputs {
		sizes[i] = in.Dim(d)
	}
	ctx.Set("dim", d)
	ctx.Set("sizes", sizes)
	out, err := ops.Concat(inputs, d)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{out}, nil
}

func (f concatFn) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	d, _ := ctx.Get("dim")
	s, _ := ctx.Get("sizes")
	dim, sizes := d.(int), s.([]int)
	g := gradOutputs[0]
	grads := make([]*tensor.Tensor, len(sizes))
	off := 0
	for i, n := range sizes {
		v, err := g.Narrow(dim, off, n)
		if err != nil {
			return nil, err
		}
		if grads[i], err = v.Contiguous(); err != nil {
			return nil, err
		}
		off += n
	}
	return grads, nil
}

// Concat is a differentiable concatenation along axis.
func Concat(tensors []*tensor.Tensor, axis int) (*tensor.Tensor, error) {
	outs, err := Apply(concatFn{axis}, tensors)
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}
