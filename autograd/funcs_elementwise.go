package autograd

import (
	"fmt"

	"light/ops"
	"light/tensor"
)

// reduceToShape sums `grad` along any axes that were broadcast to produce
// its current shape, returning a tensor of shape `want`. Used in the
// backward of broadcasting ops (Add, Mul, etc.) so the gradient matches
// each input's shape.
func reduceToShape(grad *tensor.Tensor, want []int) (*tensor.Tensor, error) {
	gdims := grad.Shape().Dims()
	if equalDims(gdims, want) {
		return grad, nil
	}
	if len(gdims) < len(want) {
		return nil, fmt.Errorf("reduceToShape: grad rank %d < want rank %d", len(gdims), len(want))
	}
	padW := make([]int, len(gdims))
	off := len(gdims) - len(want)
	for i := range off {
		padW[i] = 1
	}
	for i, d := range want {
		padW[off+i] = d
	}
	var axes []int
	for i := range gdims {
		switch {
		case padW[i] == gdims[i]:
			// axis already matches — nothing to do
		case padW[i] == 1:
			axes = append(axes, i)
		default:
			return nil, fmt.Errorf("reduceToShape: grad %v not broadcast-compatible with want %v", gdims, want)
		}
	}
	if len(axes) > 0 {
		var err error
		grad, err = ops.Sum(grad, ops.ReduceOptions{Axes: axes, KeepDims: true})
		if err != nil {
			return nil, err
		}
	}
	if len(grad.Shape().Dims()) != len(want) {
		cont, err := grad.Contiguous()
		if err != nil {
			return nil, err
		}
		return cont.Reshape(want...)
	}
	return grad, nil
}

func equalDims(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ---- Add ----

// AddFunction computes elementwise a + b with broadcasting.
type AddFunction struct{}

func (AddFunction) Name() string { return "Add" }

func (AddFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Add: want 2 inputs, got %d", len(inputs))
	}
	a, b := inputs[0], inputs[1]
	ctx.Set("a_shape", a.Shape().Dims())
	ctx.Set("b_shape", b.Shape().Dims())
	out, err := ops.Add(a, b)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{out}, nil
}

func (AddFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	aShape, _ := ctx.Get("a_shape")
	bShape, _ := ctx.Get("b_shape")
	ga, err := reduceToShape(g, aShape.([]int))
	if err != nil {
		return nil, err
	}
	gb, err := reduceToShape(g, bShape.([]int))
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{ga, gb}, nil
}

// Add is the autograd-aware wrapper around ops.Add.
func Add(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(AddFunction{}, []*tensor.Tensor{a, b})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// ---- Sub ----

type SubFunction struct{}

func (SubFunction) Name() string { return "Sub" }

func (SubFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	a, b := inputs[0], inputs[1]
	ctx.Set("a_shape", a.Shape().Dims())
	ctx.Set("b_shape", b.Shape().Dims())
	out, err := ops.Sub(a, b)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{out}, nil
}

func (SubFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	aShape, _ := ctx.Get("a_shape")
	bShape, _ := ctx.Get("b_shape")
	ga, err := reduceToShape(g, aShape.([]int))
	if err != nil {
		return nil, err
	}
	gbRaw, err := ops.Neg(g)
	if err != nil {
		return nil, err
	}
	gb, err := reduceToShape(gbRaw, bShape.([]int))
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{ga, gb}, nil
}

func Sub(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(SubFunction{}, []*tensor.Tensor{a, b})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// ---- Mul ----

type MulFunction struct{}

func (MulFunction) Name() string { return "Mul" }

func (MulFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	a, b := inputs[0], inputs[1]
	ctx.SaveForBackward(a, b)
	out, err := ops.Mul(a, b)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{out}, nil
}

func (MulFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	saved := ctx.Saved()
	a, b := saved[0], saved[1]
	gaRaw, err := ops.Mul(g, b)
	if err != nil {
		return nil, err
	}
	gbRaw, err := ops.Mul(g, a)
	if err != nil {
		return nil, err
	}
	ga, err := reduceToShape(gaRaw, a.Shape().Dims())
	if err != nil {
		return nil, err
	}
	gb, err := reduceToShape(gbRaw, b.Shape().Dims())
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{ga, gb}, nil
}

func Mul(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(MulFunction{}, []*tensor.Tensor{a, b})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// ---- Div ----

type DivFunction struct{}

func (DivFunction) Name() string { return "Div" }

func (DivFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	a, b := inputs[0], inputs[1]
	ctx.SaveForBackward(a, b)
	out, err := ops.Div(a, b)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{out}, nil
}

func (DivFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	saved := ctx.Saved()
	a, b := saved[0], saved[1]
	// d(a/b)/da = 1/b; d(a/b)/db = -a/b^2.
	gaRaw, err := ops.Div(g, b)
	if err != nil {
		return nil, err
	}
	bSq, err := ops.Mul(b, b)
	if err != nil {
		return nil, err
	}
	negA, err := ops.Neg(a)
	if err != nil {
		return nil, err
	}
	gbPart, err := ops.Mul(g, negA)
	if err != nil {
		return nil, err
	}
	gbRaw, err := ops.Div(gbPart, bSq)
	if err != nil {
		return nil, err
	}
	ga, err := reduceToShape(gaRaw, a.Shape().Dims())
	if err != nil {
		return nil, err
	}
	gb, err := reduceToShape(gbRaw, b.Shape().Dims())
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{ga, gb}, nil
}

func Div(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(DivFunction{}, []*tensor.Tensor{a, b})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// ---- Neg ----

type NegFunction struct{}

func (NegFunction) Name() string { return "Neg" }

func (NegFunction) Forward(_ *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	out, err := ops.Neg(inputs[0])
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{out}, nil
}

func (NegFunction) Backward(_ *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g, err := ops.Neg(gradOutputs[0])
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{g}, nil
}

func Neg(a *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(NegFunction{}, []*tensor.Tensor{a})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// ---- Fused unary functions (2026-10-28): forward is the ops kernel,
// backward is g · deriv(x) from the saved input. ----

// unaryFn is a Function whose forward is fwd(x) and whose backward is
// g · deriv(x) (deriv returns dy/dx from the saved input).
type unaryFn struct {
	name  string
	fwd   func(*tensor.Tensor) (*tensor.Tensor, error)
	deriv func(x *tensor.Tensor) (*tensor.Tensor, error)
}

func (f unaryFn) Name() string { return f.name }

func (f unaryFn) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	ctx.SaveForBackward(inputs[0])
	out, err := f.fwd(inputs[0])
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{out}, nil
}

func (f unaryFn) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	d, err := f.deriv(ctx.Saved()[0])
	if err != nil {
		return nil, err
	}
	gin, err := ops.Mul(gradOutputs[0], d)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gin}, nil
}

func applyUnary(f unaryFn, x *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(f, []*tensor.Tensor{x})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// stepMask returns 1 where x > 0 else 0, at x's dtype.
func stepMask(x *tensor.Tensor) (*tensor.Tensor, error) {
	s, err := ops.Sign(x)
	if err != nil {
		return nil, err
	}
	return ops.Relu(s) // sign ∈ {-1,0,1} → relu keeps only +1
}

// Exp with d/dx = eˣ.
func Exp(x *tensor.Tensor) (*tensor.Tensor, error) {
	return applyUnary(unaryFn{"Exp", ops.Exp, ops.Exp}, x)
}

// Log with d/dx = 1/x.
func Log(x *tensor.Tensor) (*tensor.Tensor, error) {
	return applyUnary(unaryFn{"Log", ops.Log, ops.Recip}, x)
}

// Abs with d/dx = sign(x).
func Abs(x *tensor.Tensor) (*tensor.Tensor, error) {
	return applyUnary(unaryFn{"Abs", ops.Abs, ops.Sign}, x)
}

// Sqr with d/dx = 2x.
func Sqr(x *tensor.Tensor) (*tensor.Tensor, error) {
	return applyUnary(unaryFn{"Sqr", ops.Sqr, func(x *tensor.Tensor) (*tensor.Tensor, error) {
		return ops.Affine(x, 2, 0)
	}}, x)
}

// Clamp with d/dx = 1 inside [lo, hi], 0 outside.
func Clamp(x *tensor.Tensor, lo, hi float64) (*tensor.Tensor, error) {
	return applyUnary(unaryFn{"Clamp",
		func(x *tensor.Tensor) (*tensor.Tensor, error) { return ops.Clamp(x, lo, hi) },
		func(x *tensor.Tensor) (*tensor.Tensor, error) {
			a, err := ops.Affine(x, 1, -lo) // x - lo > 0
			if err != nil {
				return nil, err
			}
			if a, err = stepMask(a); err != nil {
				return nil, err
			}
			b, err := ops.Affine(x, -1, hi) // hi - x > 0
			if err != nil {
				return nil, err
			}
			if b, err = stepMask(b); err != nil {
				return nil, err
			}
			return ops.Mul(a, b)
		}}, x)
}

// ---- Maximum / Minimum: the gradient flows to whichever operand won
// (ties go to `a`, as PyTorch does for maximum/minimum of equal values? No:
// PyTorch splits ties evenly; Candle routes to `a`. We follow Candle.)
type extremumFn struct {
	name string
	max  bool
}

func (f extremumFn) Name() string { return f.name }

func (f extremumFn) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	a, b := inputs[0], inputs[1]
	ctx.SaveForBackward(a, b)
	var out *tensor.Tensor
	var err error
	if f.max {
		out, err = ops.Maximum(a, b)
	} else {
		out, err = ops.Minimum(a, b)
	}
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{out}, nil
}

func (f extremumFn) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	a, b := ctx.Saved()[0], ctx.Saved()[1]
	// maskA = 1 where a wins (a ≥ b for max, a ≤ b for min).
	var cmp *tensor.Tensor
	var err error
	if f.max {
		cmp, err = ops.GE(a, b)
	} else {
		cmp, err = ops.LE(a, b)
	}
	if err != nil {
		return nil, err
	}
	maskA, err := ops.Cast(cmp, g.DType())
	if err != nil {
		return nil, err
	}
	maskB, err := ops.Affine(maskA, -1, 1)
	if err != nil {
		return nil, err
	}
	ga, err := ops.Mul(g, maskA)
	if err != nil {
		return nil, err
	}
	gb, err := ops.Mul(g, maskB)
	if err != nil {
		return nil, err
	}
	if ga, err = reduceToShape(ga, a.Shape().Dims()); err != nil {
		return nil, err
	}
	if gb, err = reduceToShape(gb, b.Shape().Dims()); err != nil {
		return nil, err
	}
	return []*tensor.Tensor{ga, gb}, nil
}

// Maximum is the differentiable elementwise max (gradient to the winner).
func Maximum(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(extremumFn{"Maximum", true}, []*tensor.Tensor{a, b})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// Minimum is the differentiable elementwise min.
func Minimum(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(extremumFn{"Minimum", false}, []*tensor.Tensor{a, b})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}
