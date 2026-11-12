package ops

import (
	"fmt"

	"light/internal/registry"
	"light/tensor"
)

// IndexSelect selects rows along axis 0 of `src` by the integer indices in
// `indices` (I64). Returns a tensor of shape (len(indices), src.shape[1:]...).
//
//	src: (N, ...)  indices: (K,)  →  out: (K, ...)
//
// Currently axis 0 only; generalised indexing lands later.
func IndexSelect(src, indices *tensor.Tensor) (*tensor.Tensor, error) {
	if src.Rank() < 1 {
		return nil, fmt.Errorf("IndexSelect: src must have rank >= 1")
	}
	if indices.Rank() != 1 {
		return nil, fmt.Errorf("IndexSelect: indices must be 1-D (got rank %d)", indices.Rank())
	}
	if !src.Device().SameAs(indices.Device()) {
		return nil, fmt.Errorf("IndexSelect: device mismatch (%v vs %v)", src.Device(), indices.Device())
	}
	src, err := ensureContig(src)
	if err != nil {
		return nil, err
	}
	indices, err = ensureContig(indices)
	if err != nil {
		return nil, err
	}
	// Output shape: replace first dim of src with len(indices).
	outDims := append([]int{indices.Dim(0)}, src.Shape().Dims()[1:]...)
	out, err := tensor.Empty(src.DType(), outDims, src.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "index_select", Dev: src.Device().Kind(), DType: src.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{src, indices},
		Output: out,
	}); err != nil {
		return nil, err
	}
	return out, nil
}
