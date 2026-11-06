package autograd

import (
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// scalarLike returns a (1,) tensor of value `v` on `like`'s device, cast to
// `like`'s dtype. Used by composed activations (GELU, ReLU6, …) so their
// constants don't sit at F32 when the rest of the graph runs in BF16.
func scalarLike(v float32, like *tensor.Tensor) (*tensor.Tensor, error) {
	t, err := tensor.Full(v, []int{1}, like.Device())
	if err != nil {
		return nil, err
	}
	if like.DType() != dtype.F32 {
		t, err = ops.Cast(t, like.DType())
		if err != nil {
			return nil, err
		}
	}
	return t, nil
}

// ---- ReLU ----
// Backward: grad_in = grad_out * (input > 0).

type ReLUFunction struct{}

func (ReLUFunction) Name() string { return "ReLU" }

func (ReLUFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	a := inputs[0]
	ctx.SaveForBackward(a)
	out, err := ops.Relu(a)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{out}, nil
}

func (ReLUFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	a := ctx.Saved()[0]
	// mask = (a > 0); grad_in = mask ? g : 0. Expressed via ops.Where so it
	// works on CPU and CUDA without needing a U8→F32 cast (which CUDA
	// doesn't implement).
	zero, err := tensor.Zeros(a.DType(), a.Shape().Dims(), a.Device())
	if err != nil {
		return nil, err
	}
	mask, err := ops.GT(a, zero)
	if err != nil {
		return nil, err
	}
	gin, err := ops.Where(mask, g, zero)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gin}, nil
}

func ReLU(a *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(ReLUFunction{}, []*tensor.Tensor{a})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// ---- Sigmoid ----
// σ(x) = 1 / (1 + exp(-x)); σ'(x) = σ(x) * (1 - σ(x)).

type SigmoidFunction struct{}

func (SigmoidFunction) Name() string { return "Sigmoid" }

func (SigmoidFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	out, err := ops.Sigmoid(inputs[0])
	if err != nil {
		return nil, err
	}
	ctx.SaveForBackward(out)
	return []*tensor.Tensor{out}, nil
}

func (SigmoidFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	y := ctx.Saved()[0]
	one, err := tensor.Full[float32](1.0, []int{1}, y.Device())
	if err != nil {
		return nil, err
	}
	oneMinusY, err := ops.Sub(one, y)
	if err != nil {
		return nil, err
	}
	dy, err := ops.Mul(y, oneMinusY)
	if err != nil {
		return nil, err
	}
	gin, err := ops.Mul(g, dy)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gin}, nil
}

func Sigmoid(a *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(SigmoidFunction{}, []*tensor.Tensor{a})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// ---- Tanh ----
// tanh'(x) = 1 - tanh(x)^2.

type TanhFunction struct{}

func (TanhFunction) Name() string { return "Tanh" }

func (TanhFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	out, err := ops.Tanh(inputs[0])
	if err != nil {
		return nil, err
	}
	ctx.SaveForBackward(out)
	return []*tensor.Tensor{out}, nil
}

func (TanhFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	y := ctx.Saved()[0]
	ySq, err := ops.Mul(y, y)
	if err != nil {
		return nil, err
	}
	one, err := tensor.Full[float32](1.0, []int{1}, y.Device())
	if err != nil {
		return nil, err
	}
	dy, err := ops.Sub(one, ySq)
	if err != nil {
		return nil, err
	}
	gin, err := ops.Mul(g, dy)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gin}, nil
}

func Tanh(a *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(TanhFunction{}, []*tensor.Tensor{a})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// ---- SiLU ----
// SiLU(x) = x * σ(x); d/dx = σ(x) + x*σ(x)*(1-σ(x)) = σ(x)*(1 + x*(1-σ(x))).

type SiLUFunction struct{}

func (SiLUFunction) Name() string { return "SiLU" }

func (SiLUFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	a := inputs[0]
	ctx.SaveForBackward(a)
	out, err := ops.SiLU(a)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{out}, nil
}

func (SiLUFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	x := ctx.Saved()[0]
	sig, err := ops.Sigmoid(x)
	if err != nil {
		return nil, err
	}
	one, err := tensor.Full[float32](1.0, []int{1}, x.Device())
	if err != nil {
		return nil, err
	}
	oneMinusSig, err := ops.Sub(one, sig)
	if err != nil {
		return nil, err
	}
	xTimesOMS, err := ops.Mul(x, oneMinusSig)
	if err != nil {
		return nil, err
	}
	inner, err := ops.Add(one, xTimesOMS)
	if err != nil {
		return nil, err
	}
	dy, err := ops.Mul(sig, inner)
	if err != nil {
		return nil, err
	}
	gin, err := ops.Mul(g, dy)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gin}, nil
}

func SiLU(a *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(SiLUFunction{}, []*tensor.Tensor{a})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// GELU is the Gaussian Error Linear Unit, tanh approximation used by GPT-2:
//
//	GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
//
// Composed from autograd primitives (Mul, Add, Tanh) so gradients flow
// automatically. Matches PyTorch's `F.gelu(x, approximate="tanh")`.
func GELU(x *tensor.Tensor) (*tensor.Tensor, error) {
	const c1 = float32(0.7978845608) // sqrt(2/π)
	const c2 = float32(0.044715)
	// Constants must live at x's dtype — otherwise the very first
	// `Mul(x, c…)` errors with a dtype mismatch when callers pass BF16
	// (e.g. SD 1.5's GeGLU under -bf16). Always build the scalar at F32
	// then cast to match.
	one, err := scalarLike(1.0, x)
	if err != nil {
		return nil, err
	}
	half, err := scalarLike(0.5, x)
	if err != nil {
		return nil, err
	}
	c1T, err := scalarLike(c1, x)
	if err != nil {
		return nil, err
	}
	c2T, err := scalarLike(c2, x)
	if err != nil {
		return nil, err
	}
	xSq, err := Mul(x, x)
	if err != nil {
		return nil, err
	}
	xCube, err := Mul(xSq, x)
	if err != nil {
		return nil, err
	}
	c2xCube, err := Mul(c2T, xCube)
	if err != nil {
		return nil, err
	}
	xPlus, err := Add(x, c2xCube)
	if err != nil {
		return nil, err
	}
	inner, err := Mul(c1T, xPlus)
	if err != nil {
		return nil, err
	}
	tanhInner, err := Tanh(inner)
	if err != nil {
		return nil, err
	}
	onePlus, err := Add(one, tanhInner)
	if err != nil {
		return nil, err
	}
	halfX, err := Mul(half, x)
	if err != nil {
		return nil, err
	}
	return Mul(halfX, onePlus)
}

// ReLU6 is min(max(x, 0), 6), used by MobileNet v2/v3. Decomposed into
// `ReLU(x) - ReLU(x - 6)` so gradients flow through the existing autograd
// ReLU primitive without a new function.
func ReLU6(x *tensor.Tensor) (*tensor.Tensor, error) {
	six, err := scalarLike(6.0, x)
	if err != nil {
		return nil, err
	}
	clip, err := ReLU(x)
	if err != nil {
		return nil, err
	}
	xMinus6, err := Sub(x, six)
	if err != nil {
		return nil, err
	}
	overflow, err := ReLU(xMinus6)
	if err != nil {
		return nil, err
	}
	return Sub(clip, overflow)
}

// cast helper — unused externally but kept as a reference for future Cast fn.
var _ = dtype.F32

// ---- ELU / LeakyReLU / Mish / HardSigmoid / HardSwish / PReLU (2026-10-28) ----

// ELU with d/dx = 1 for x > 0, alpha·eˣ otherwise.
func ELU(x *tensor.Tensor, alpha float64) (*tensor.Tensor, error) {
	return applyUnary(unaryFn{"ELU",
		func(x *tensor.Tensor) (*tensor.Tensor, error) { return ops.ELU(x, alpha) },
		func(x *tensor.Tensor) (*tensor.Tensor, error) {
			// alpha·eˣ for x ≤ 0 equals elu(x)+alpha; select by the step mask.
			pos, err := stepMask(x)
			if err != nil {
				return nil, err
			}
			neg, err := ops.Affine(pos, -1, 1) // 1 - mask
			if err != nil {
				return nil, err
			}
			e, err := ops.ELU(x, alpha)
			if err != nil {
				return nil, err
			}
			if e, err = ops.Affine(e, 1, alpha); err != nil {
				return nil, err
			}
			if neg, err = ops.Mul(neg, e); err != nil {
				return nil, err
			}
			return ops.Add(pos, neg)
		}}, x)
}

// LeakyReLU with d/dx = 1 for x > 0, slope otherwise.
func LeakyReLU(x *tensor.Tensor, slope float64) (*tensor.Tensor, error) {
	return applyUnary(unaryFn{"LeakyReLU",
		func(x *tensor.Tensor) (*tensor.Tensor, error) { return ops.LeakyReLU(x, slope) },
		func(x *tensor.Tensor) (*tensor.Tensor, error) {
			pos, err := stepMask(x)
			if err != nil {
				return nil, err
			}
			return ops.Affine(pos, 1-slope, slope) // mask·(1-slope) + slope
		}}, x)
}

// Mish with d/dx = tanh(sp) + x·σ(x)·(1 - tanh²(sp)), sp = softplus(x).
func Mish(x *tensor.Tensor) (*tensor.Tensor, error) {
	return applyUnary(unaryFn{"Mish", ops.Mish, func(x *tensor.Tensor) (*tensor.Tensor, error) {
		m, err := ops.Mish(x) // x·tanh(sp)
		if err != nil {
			return nil, err
		}
		// tanh(sp) = mish/x is ill-conditioned at 0; recompute: sp = log1p(eˣ).
		e, err := ops.Exp(x)
		if err != nil {
			return nil, err
		}
		sp, err := ops.Affine(e, 1, 1)
		if err != nil {
			return nil, err
		}
		if sp, err = ops.Log(sp); err != nil {
			return nil, err
		}
		t, err := ops.Tanh(sp)
		if err != nil {
			return nil, err
		}
		t2, err := ops.Sqr(t)
		if err != nil {
			return nil, err
		}
		oneMinusT2, err := ops.Affine(t2, -1, 1)
		if err != nil {
			return nil, err
		}
		sig, err := ops.Sigmoid(x)
		if err != nil {
			return nil, err
		}
		xs, err := ops.Mul(x, sig)
		if err != nil {
			return nil, err
		}
		if xs, err = ops.Mul(xs, oneMinusT2); err != nil {
			return nil, err
		}
		_ = m
		return ops.Add(t, xs)
	}}, x)
}

// HardSigmoid with d/dx = 1/6 inside (-3, 3), 0 outside.
func HardSigmoid(x *tensor.Tensor) (*tensor.Tensor, error) {
	return applyUnary(unaryFn{"HardSigmoid", ops.HardSigmoid, hardSigmoidDeriv}, x)
}

func hardSigmoidDeriv(x *tensor.Tensor) (*tensor.Tensor, error) {
	// inside = step(x+3)·step(3-x)
	a, err := ops.Affine(x, 1, 3)
	if err != nil {
		return nil, err
	}
	if a, err = stepMask(a); err != nil {
		return nil, err
	}
	b, err := ops.Affine(x, -1, 3)
	if err != nil {
		return nil, err
	}
	if b, err = stepMask(b); err != nil {
		return nil, err
	}
	in, err := ops.Mul(a, b)
	if err != nil {
		return nil, err
	}
	return ops.Affine(in, 1.0/6, 0)
}

// HardSwish with d/dx = hard_sigmoid(x) + x·hard_sigmoid'(x).
func HardSwish(x *tensor.Tensor) (*tensor.Tensor, error) {
	return applyUnary(unaryFn{"HardSwish", ops.HardSwish, func(x *tensor.Tensor) (*tensor.Tensor, error) {
		hs, err := ops.HardSigmoid(x)
		if err != nil {
			return nil, err
		}
		d, err := hardSigmoidDeriv(x)
		if err != nil {
			return nil, err
		}
		xd, err := ops.Mul(x, d)
		if err != nil {
			return nil, err
		}
		return ops.Add(hs, xd)
	}}, x)
}

// PReLU is relu(x) + w·min(x, 0) with a learnable slope w broadcast against
// x (per-channel: w of shape (C, 1, 1) for NCHW input, or (1,) shared).
func PReLU(x, w *tensor.Tensor) (*tensor.Tensor, error) {
	pos, err := ReLU(x)
	if err != nil {
		return nil, err
	}
	negX, err := Sub(x, pos) // min(x, 0)
	if err != nil {
		return nil, err
	}
	scaled, err := Mul(w, negX)
	if err != nil {
		return nil, err
	}
	return Add(pos, scaled)
}
