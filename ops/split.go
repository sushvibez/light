package ops

import (
	"fmt"

	"light/tensor"
)

// Split divides `a` along `dim` into a slice of tensors with the given sizes.
// The sizes must sum to `a.Dim(dim)`. Returned tensors are views (no copy).
//
//	Split(a, []int{2, 3}, 0) on a (5, 4) returns [(2, 4), (3, 4)].
func Split(a *tensor.Tensor, sizes []int, dim int) ([]*tensor.Tensor, error) {
	if len(sizes) == 0 {
		return nil, fmt.Errorf("split: sizes must be non-empty")
	}
	d, err := a.Shape().NormalizeAxis(dim)
	if err != nil {
		return nil, fmt.Errorf("split: %w", err)
	}
	total := 0
	for i, s := range sizes {
		if s <= 0 {
			return nil, fmt.Errorf("split: size[%d]=%d must be positive", i, s)
		}
		total += s
	}
	if total != a.Dim(d) {
		return nil, fmt.Errorf("split: sizes sum to %d, want %d along dim %d", total, a.Dim(d), d)
	}
	out := make([]*tensor.Tensor, len(sizes))
	off := 0
	for i, s := range sizes {
		part, err := a.Narrow(d, off, s)
		if err != nil {
			return nil, err
		}
		out[i] = part
		off += s
	}
	return out, nil
}

// Chunk divides `a` along `dim` into `n` approximately-equal chunks. If the
// dim size isn't divisible by n, the last chunk gets the remainder. If the
// dim size is smaller than n, the result has fewer than n chunks.
//
// Matches PyTorch semantics: `torch.chunk(t, n, dim)`.
func Chunk(a *tensor.Tensor, n, dim int) ([]*tensor.Tensor, error) {
	if n <= 0 {
		return nil, fmt.Errorf("chunk: n must be positive (got %d)", n)
	}
	d, err := a.Shape().NormalizeAxis(dim)
	if err != nil {
		return nil, fmt.Errorf("chunk: %w", err)
	}
	size := a.Dim(d)
	if size == 0 {
		return []*tensor.Tensor{}, nil
	}
	// PyTorch's rule: chunk_size = ceil(size / n). Last chunk may be smaller.
	chunkSize := (size + n - 1) / n
	if chunkSize == 0 {
		chunkSize = 1
	}
	var out []*tensor.Tensor
	for off := 0; off < size; off += chunkSize {
		length := chunkSize
		if off+length > size {
			length = size - off
		}
		part, err := a.Narrow(d, off, length)
		if err != nil {
			return nil, err
		}
		out = append(out, part)
	}
	return out, nil
}
