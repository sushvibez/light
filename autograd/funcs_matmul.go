package autograd

import (
	"fmt"

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
	rA, rB := a.Rank(), b.Rank()
	bT, err := b.Transpose(rB-2, rB-1)
	if err != nil {
		return nil, err
	}
	aT, err := a.Transpose(rA-2, rA-1)
	if err != nil {
		return nil, err
	}
	// Batched matmul requires contiguous operands (the CUDA/CPU kernels
	// assume default strides); materialise the transposed views.
	bTc, err := bT.Contiguous()
	if err != nil {
		return nil, err
	}
	aTc, err := aT.Contiguous()
	if err != nil {
		return nil, err
	}
	ga, err := ops.MatMul(g, bTc)
	if err != nil {
		return nil, err
	}
	gb, err := ops.MatMul(aTc, g)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{ga, gb}, nil
}

// MatMul is the autograd-aware wrapper for 2-D matrix multiply.
func MatMul(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(MatMulFunction{}, []*tensor.Tensor{a, b})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}
