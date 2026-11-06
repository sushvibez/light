package autograd

import (
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// DropoutFunction applies inverted dropout with probability `p`. At training
// time each element is independently zeroed with probability `p` and the
// survivors are scaled by 1/(1-p); at eval time the input is returned
// unchanged. Backward multiplies the upstream gradient by the same mask.
//
// Implementation: the mask is built explicitly as a (same-shape) F32 tensor
// holding 0 for dropped positions and 1/(1-p) for kept positions — all
// constructed via composed ops so CPU and GPU both work. Saving the mask
// for backward avoids re-rolling the RNG.
type DropoutFunction struct {
	p        float32
	training bool
}

func (DropoutFunction) Name() string { return "Dropout" }

func (f DropoutFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x := inputs[0]
	// Eval / p==0 paths never reach Forward — the Dropout() entry point
	// returns the input directly in those cases so the caller's gradFn is
	// preserved (Apply would otherwise clobber it by reusing the same
	// tensor as output).
	// r ~ Uniform[0, 1); keep = r >= p; scaled_mask = keep ? 1/(1-p) : 0.
	r, err := tensor.Rand(dtype.F32, x.Shape().Dims(), x.Device())
	if err != nil {
		return nil, err
	}
	thresh, err := tensor.Full(f.p, x.Shape().Dims(), x.Device())
	if err != nil {
		return nil, err
	}
	keep, err := ops.GE(r, thresh)
	if err != nil {
		return nil, err
	}
	scale := float32(1.0) / (1.0 - f.p)
	scaledOnes, err := tensor.Full(scale, x.Shape().Dims(), x.Device())
	if err != nil {
		return nil, err
	}
	zeros, err := tensor.Zeros(dtype.F32, x.Shape().Dims(), x.Device())
	if err != nil {
		return nil, err
	}
	mask, err := ops.Where(keep, scaledOnes, zeros)
	if err != nil {
		return nil, err
	}
	y, err := ops.Mul(x, mask)
	if err != nil {
		return nil, err
	}
	ctx.SaveForBackward(mask)
	return []*tensor.Tensor{y}, nil
}

func (DropoutFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	mask := ctx.Saved()[0]
	gin, err := ops.Mul(g, mask)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gin}, nil
}

// Dropout is the autograd-aware dropout. In eval mode (or p == 0) the input
// is returned unchanged so its gradFn is preserved.
func Dropout(x *tensor.Tensor, p float32, training bool) (*tensor.Tensor, error) {
	if !training || p == 0 {
		return x, nil
	}
	outs, err := Apply(DropoutFunction{p: p, training: training}, []*tensor.Tensor{x})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}
