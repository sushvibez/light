package autograd

import (
	"fmt"

	"light/dtype"
	"light/ops"
	"light/tensor"
)

// IndexSelectFunction looks up rows of `src` at positions given by `indices`
// (axis 0 only, matching ops.IndexSelect).
//
// Forward:  out[k, ...] = src[indices[k], ...]
// Backward: grad_src[j, ...] += Σ_k 1{indices[k]=j} · grad_out[k, ...]
//
// The backward uses ops.ScatterAdd with a broadcast-expanded indices tensor
// of shape (K, D_tail...) so one element per output position accumulates into
// the correct row of src.
type IndexSelectFunction struct{}

func (IndexSelectFunction) Name() string { return "IndexSelect" }

func (IndexSelectFunction) Forward(ctx *Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("IndexSelect: need (src, indices)")
	}
	src, indices := inputs[0], inputs[1]
	if indices.DType() != dtype.I64 {
		return nil, fmt.Errorf("IndexSelect: indices must be I64 (got %v)", indices.DType())
	}
	ctx.SaveForBackward(src, indices)
	out, err := ops.IndexSelect(src, indices)
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{out}, nil
}

func (IndexSelectFunction) Backward(ctx *Context, gradOutputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := gradOutputs[0]
	saved := ctx.Saved()
	src, indices := saved[0], saved[1]

	// Build an expanded index tensor of the same shape as g (and gradients):
	// each trailing-axis element along a row gets the same scalar index.
	K := indices.Dim(0)
	tailShape := src.Shape().Dims()[1:]
	idxExpandedShape := make([]int, 1+len(tailShape))
	idxExpandedShape[0] = K
	for i, d := range tailShape {
		idxExpandedShape[1+i] = d
	}
	// indices is 1-D of length K. Reshape to [K, 1, 1, …] then Expand to g's shape.
	idxR := make([]int, 1+len(tailShape))
	idxR[0] = K
	for i := range tailShape {
		idxR[1+i] = 1
	}
	idxReshaped, err := indices.Reshape(idxR...)
	if err != nil {
		return nil, err
	}
	idxExpanded, err := idxReshaped.Expand(idxExpandedShape...)
	if err != nil {
		return nil, err
	}
	idxContig, err := idxExpanded.Contiguous()
	if err != nil {
		return nil, err
	}
	zeroBase, err := tensor.Zeros(src.DType(), src.Shape().Dims(), src.Device())
	if err != nil {
		return nil, err
	}
	gradSrc, err := ops.ScatterAdd(zeroBase, idxContig, g, 0)
	if err != nil {
		return nil, err
	}
	// indices has no gradient.
	return []*tensor.Tensor{gradSrc, nil}, nil
}

// IndexSelect is the autograd-aware row selection along axis 0.
func IndexSelect(src, indices *tensor.Tensor) (*tensor.Tensor, error) {
	outs, err := Apply(IndexSelectFunction{}, []*tensor.Tensor{src, indices})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}
