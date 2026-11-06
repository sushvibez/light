package ops

import (
	"fmt"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// Gather selects values from `src` along `dim` at positions given by `indices`.
// The output has the same shape as `indices`. src and indices must have the
// same rank; indices must be I64; indices must match src.shape except along `dim`.
//
//	out[i_0, …, i_k, …, i_n] = src[i_0, …, indices[i_0, …, i_k, …, i_n], …, i_n]
//
// (k = dim)
func Gather(src, indices *tensor.Tensor, dim int) (*tensor.Tensor, error) {
	if indices.DType() != dtype.I64 {
		return nil, fmt.Errorf("gather: indices must be I64")
	}
	if src.Rank() != indices.Rank() {
		return nil, fmt.Errorf("gather: src rank %d != indices rank %d", src.Rank(), indices.Rank())
	}
	d, err := src.Shape().NormalizeAxis(dim)
	if err != nil {
		return nil, fmt.Errorf("gather: %w", err)
	}
	if !src.Device().SameAs(indices.Device()) {
		return nil, fmt.Errorf("gather: device mismatch")
	}
	srcC, err := ensureContig(src)
	if err != nil {
		return nil, err
	}
	idxC, err := ensureContig(indices)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(src.DType(), indices.Shape().Dims(), src.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "gather", Dev: src.Device().Kind(), DType: src.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{srcC, idxC},
		Output: out,
		Attrs:  map[string]any{"dim": d},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// ScatterAdd computes out = base; out[…, indices[j], …] += src[…, j, …] along `dim`.
// src and indices have identical shape; out has the same shape as `base`.
// If `base` is nil, `out` starts at zero.
func ScatterAdd(base *tensor.Tensor, indices, src *tensor.Tensor, dim int) (*tensor.Tensor, error) {
	if indices.DType() != dtype.I64 {
		return nil, fmt.Errorf("scatter_add: indices must be I64")
	}
	if !src.Shape().Equal(indices.Shape()) {
		return nil, fmt.Errorf("scatter_add: src and indices must have identical shape (got %v vs %v)", src.Shape(), indices.Shape())
	}
	if base.Rank() != src.Rank() {
		return nil, fmt.Errorf("scatter_add: base rank %d != src rank %d", base.Rank(), src.Rank())
	}
	if base.DType() != src.DType() {
		return nil, fmt.Errorf("scatter_add: base/src dtype mismatch")
	}
	d, err := base.Shape().NormalizeAxis(dim)
	if err != nil {
		return nil, fmt.Errorf("scatter_add: %w", err)
	}

	baseC, err := ensureContig(base)
	if err != nil {
		return nil, err
	}
	srcC, err := ensureContig(src)
	if err != nil {
		return nil, err
	}
	idxC, err := ensureContig(indices)
	if err != nil {
		return nil, err
	}

	// Start from a copy of base (since scatter_add is an accumulation).
	out, err := tensor.Zeros(base.DType(), base.Shape().Dims(), base.Device())
	if err != nil {
		return nil, err
	}
	if err := out.CopyFrom(baseC); err != nil {
		return nil, err
	}

	k, err := registry.Lookup(registry.Key{Op: "scatter_add", Dev: base.Device().Kind(), DType: base.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{srcC, idxC},
		Output: out,
		Attrs:  map[string]any{"dim": d},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// indexOp validates the shared contract of the scatter-style ops and runs
// the kernel on a copy of base.
func indexOp(op string, base, indices, src *tensor.Tensor, dim int, oneD bool) (*tensor.Tensor, error) {
	if indices.DType() != dtype.I64 {
		return nil, fmt.Errorf("%s: indices must be I64", op)
	}
	if base.Rank() != src.Rank() {
		return nil, fmt.Errorf("%s: base rank %d != src rank %d", op, base.Rank(), src.Rank())
	}
	if base.DType() != src.DType() {
		return nil, fmt.Errorf("%s: base/src dtype mismatch (%v vs %v)", op, base.DType(), src.DType())
	}
	if !base.Device().SameAs(src.Device()) || !base.Device().SameAs(indices.Device()) {
		return nil, fmt.Errorf("%s: device mismatch", op)
	}
	d, err := base.Shape().NormalizeAxis(dim)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	if oneD {
		if indices.Rank() != 1 || indices.Dim(0) != src.Dim(d) {
			return nil, fmt.Errorf("%s: indices must be 1-D with length src.dim(%d)=%d, got %v", op, d, src.Dim(d), indices.Shape())
		}
	} else if !src.Shape().Equal(indices.Shape()) {
		return nil, fmt.Errorf("%s: src and indices must have identical shape (got %v vs %v)", op, src.Shape(), indices.Shape())
	}
	for i := range base.Rank() {
		if i != d && src.Dim(i) != base.Dim(i) {
			return nil, fmt.Errorf("%s: src %v must match base %v except along dim %d", op, src.Shape(), base.Shape(), d)
		}
	}
	baseC, err := ensureContig(base)
	if err != nil {
		return nil, err
	}
	srcC, err := ensureContig(src)
	if err != nil {
		return nil, err
	}
	idxC, err := ensureContig(indices)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(base.DType(), base.Shape().Dims(), base.Device())
	if err != nil {
		return nil, err
	}
	if err := out.CopyFrom(baseC); err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: op, Dev: base.Device().Kind(), DType: base.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{srcC, idxC},
		Output: out,
		Attrs:  map[string]any{"dim": d},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// Scatter writes src into a copy of base along dim (PyTorch `scatter`,
// Candle `scatter_set`): out[…, indices[…, j, …], …] = src[…, j, …].
// indices has src's shape. With duplicate indices the CPU keeps the last
// write; CUDA keeps an arbitrary one.
func Scatter(base, indices, src *tensor.Tensor, dim int) (*tensor.Tensor, error) {
	return indexOp("scatter", base, indices, src, dim, false)
}

// IndexAdd adds whole slices of src into a copy of base (PyTorch
// `index_add`): out[…, indices[j], …] += src[…, j, …], with a 1-D index of
// length src.dim(dim). Unlike ScatterAdd the index addresses slices, not
// elements.
func IndexAdd(base, indices, src *tensor.Tensor, dim int) (*tensor.Tensor, error) {
	return indexOp("index_add", base, indices, src, dim, true)
}

// SliceScatter returns a copy of base with base.narrow(dim, start,
// src.dim(dim)) replaced by src (Candle `slice_scatter`). Built from Narrow +
// Concat so it works on every device and dtype that Concat supports.
func SliceScatter(base, src *tensor.Tensor, dim, start int) (*tensor.Tensor, error) {
	d, err := base.Shape().NormalizeAxis(dim)
	if err != nil {
		return nil, fmt.Errorf("slice_scatter: %w", err)
	}
	if base.Rank() != src.Rank() || base.DType() != src.DType() {
		return nil, fmt.Errorf("slice_scatter: base %v/%v and src %v/%v must share rank and dtype", base.Shape(), base.DType(), src.Shape(), src.DType())
	}
	n, l := base.Dim(d), src.Dim(d)
	if start < 0 || start+l > n {
		return nil, fmt.Errorf("slice_scatter: [%d, %d) out of range for dim %d of length %d", start, start+l, d, n)
	}
	for i := range base.Rank() {
		if i != d && src.Dim(i) != base.Dim(i) {
			return nil, fmt.Errorf("slice_scatter: src %v must match base %v except along dim %d", src.Shape(), base.Shape(), d)
		}
	}
	parts := make([]*tensor.Tensor, 0, 3)
	if start > 0 {
		head, err := base.Narrow(d, 0, start)
		if err != nil {
			return nil, err
		}
		parts = append(parts, head)
	}
	parts = append(parts, src)
	if start+l < n {
		tail, err := base.Narrow(d, start+l, n-start-l)
		if err != nil {
			return nil, err
		}
		parts = append(parts, tail)
	}
	if len(parts) == 1 {
		return ensureContig(src)
	}
	return Concat(parts, d)
}

// SliceAssign returns a copy of base with the block starting at `starts`
// (one offset per dim) replaced by src (Candle `slice_assign`): the
// multi-dimensional generalisation of SliceScatter.
func SliceAssign(base, src *tensor.Tensor, starts []int) (*tensor.Tensor, error) {
	if len(starts) != base.Rank() || base.Rank() != src.Rank() {
		return nil, fmt.Errorf("slice_assign: need one start per dim (%d), got %d for src rank %d", base.Rank(), len(starts), src.Rank())
	}
	return sliceAssignDim(base, src, starts, 0)
}

func sliceAssignDim(base, src *tensor.Tensor, starts []int, d int) (*tensor.Tensor, error) {
	if d == base.Rank() {
		return ensureContig(src)
	}
	// Replace, within base's block along dims ≥ d, the narrow window of dim d.
	window, err := base.Narrow(d, starts[d], src.Dim(d))
	if err != nil {
		return nil, fmt.Errorf("slice_assign dim %d: %w", d, err)
	}
	inner, err := sliceAssignDim(window, src, starts, d+1)
	if err != nil {
		return nil, err
	}
	return SliceScatter(base, inner, d, starts[d])
}
