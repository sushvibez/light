package tensor

import (
	"fmt"

	"light/shape"
)

// Narrow returns a view into the receiver with the given axis narrowed to
// [start, start+length). The new tensor shares the underlying storage; only
// the shape and offset change.
//
// Equivalent to PyTorch's `t.narrow(dim, start, length)`.
func (t *Tensor) Narrow(dim, start, length int) (*Tensor, error) {
	rank := t.Rank()
	if dim < 0 {
		dim += rank
	}
	if dim < 0 || dim >= rank {
		return nil, fmt.Errorf("Narrow: dim %d out of range for rank %d", dim, rank)
	}
	if length < 0 {
		return nil, fmt.Errorf("Narrow: length must be non-negative, got %d", length)
	}
	if start < 0 || start+length > t.Dim(dim) {
		return nil, fmt.Errorf("Narrow: slice [%d, %d) out of range for dim size %d", start, start+length, t.Dim(dim))
	}
	newDims := t.shape.Dims()
	newDims[dim] = length
	newOffset := t.offset + start*t.strides.Get(dim)
	return newTensor(t.storage, shape.New(newDims...), shape.NewStrides(t.strides.Slice()...), newOffset), nil
}
