package tensor

import (
	"fmt"

	"light/shape"
)

// Slice returns a view with the given start[i]:stop[i] range along every axis.
// Both start and stop must have length == Rank(). stop[i] == 0 is treated as
// "end of dim i" only when start[i] == 0 (i.e., a leading no-op); otherwise
// stop[i] must be > start[i].
//
// Semantics match PyTorch's `t[start[0]:stop[0], start[1]:stop[1], ...]`.
func (t *Tensor) Slice(start, stop []int) (*Tensor, error) {
	rank := t.Rank()
	if len(start) != rank || len(stop) != rank {
		return nil, fmt.Errorf("Slice: start and stop must have length %d", rank)
	}
	cur := t
	for d := range rank {
		s := start[d]
		e := stop[d]
		if s == 0 && e == 0 {
			// No-op along this axis.
			continue
		}
		if e <= s {
			return nil, fmt.Errorf("Slice: axis %d stop %d must be > start %d", d, e, s)
		}
		length := e - s
		n, err := cur.Narrow(d, s, length)
		if err != nil {
			return nil, fmt.Errorf("Slice axis %d: %w", d, err)
		}
		cur = n
	}
	return cur, nil
}

// Expand returns a view where each axis of size 1 in the receiver can be
// broadcast to the target size in `dims`. Axes with size > 1 must match the
// target exactly. Expansion is zero-cost: the resulting view has stride 0
// along expanded axes.
//
//	t.Expand([2, 3, 1]).Expand([2, 3, 5])  // legal: last dim expands 1 → 5
func (t *Tensor) Expand(dims ...int) (*Tensor, error) {
	if len(dims) != t.Rank() {
		return nil, fmt.Errorf("Expand: target rank %d != tensor rank %d", len(dims), t.Rank())
	}
	curDims := t.shape.Dims()
	curStrides := t.strides.Slice()
	newDims := make([]int, len(dims))
	newStrides := make([]int, len(dims))
	for i, d := range dims {
		if d < 0 {
			// -1 keeps the current size (PyTorch convention).
			newDims[i] = curDims[i]
			newStrides[i] = curStrides[i]
			continue
		}
		switch {
		case curDims[i] == d:
			newDims[i] = d
			newStrides[i] = curStrides[i]
		case curDims[i] == 1:
			newDims[i] = d
			newStrides[i] = 0
		default:
			return nil, fmt.Errorf("Expand: axis %d current size %d cannot expand to %d (must be 1 or equal)", i, curDims[i], d)
		}
	}
	return newTensor(t.storage, shape.New(newDims...), shape.NewStrides(newStrides...), t.offset), nil
}
