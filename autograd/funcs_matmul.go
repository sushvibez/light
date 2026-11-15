package autograd

import (
	"fmt"

	"light/device"
	"light/ops"
	"light/tensor"
)

// MatMulFunction implements matmul with batched rank support. For inputs of
// rank r ≥ 2, the last two dims are the matrix axes and any leading dims
// are batch. Both operands must have the same rank.
//
//	dC/dA = grad_C @ B^T (last two axes transposed)
//	dC/dB = A^T @ grad_C
type MatMulFunction struct{}

func (MatMulFunction) Name() string { return "MatMul" }

func (MatMulFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	a, b := inputs[0], inputs[1]
	if a.Rank() < 2 || b.Rank() < 2 {
		return nil, fmt.Errorf("autograd MatMul: operands must be at least rank 2 (got %d, %d)", a.Rank(), b.Rank())
	}
	if a.Rank() != b.Rank() {
		return nil, fmt.Errorf("autograd MatMul: operand ranks must match (got %d vs %d)", a.Rank(), b.Rank())
	}
	ctx.SaveForBackward(a, b)
	out, err := ops.MatMul(a, b)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{out}, nil
}

func (MatMulFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	saved := ctx.Saved()
	a, b := saved[0], saved[1]
	needs := func(t *tensor.Tensor) bool { return t.RequiresGrad() || t.GradFn() != nil }
	grads := make([]*tensor.Tensor, 2) // nil: the engine skips that input
	if needs(a) {
		// dA = g @ Bᵀ. The CPU kernels are stride-aware and read Bᵀ (a
		// transposed view: unit row stride) on their dot-product path, so
		// it is passed as a view there; other backends get a contiguous
		// copy.
		bT, err := matmulTransposed(b, device.IsCPU(b.Device()))
		if err != nil {
			return nil, err
		}
		if grads[0], err = ops.MatMul(g, bT); err != nil {
			return nil, err
		}
	}
	if needs(b) {
		// dB = Aᵀ @ g. Aᵀ as a view has a non-unit inner stride, which the
		// CPU F32 kernels only handle on their naive fallback, so it is
		// always materialised.
		aT, err := matmulTransposed(a, false)
		if err != nil {
			return nil, err
		}
		if grads[1], err = ops.MatMul(aT, g); err != nil {
			return nil, err
		}
	}
	return grads, nil
}

// matmulTransposed swaps x's last two axes, as a view or a contiguous copy.
func matmulTransposed(x *tensor.Tensor, view bool) (*tensor.Tensor, error) {
	r := x.Rank()
	xT, err := x.Transpose(r-2, r-1)
	if err != nil || view {
		return xT, err
	}
	return xT.Contiguous()
}

// MatMul is the autograd-aware wrapper for 2-D matrix multiply.
func MatMul(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(MatMulFunction{}, []*tensor.Tensor{a, b})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}
