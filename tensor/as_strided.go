package tensor

import (
	"fmt"

	"light/shape"
)

// AsStrided returns a view over the receiver's storage with explicit dims,
// strides (in elements) and offset — PyTorch's `as_strided`. Strides may be
// zero (broadcast) or sum several of the receiver's strides (a diagonal:
// dims [n], strides [s0+s1]). Every element the view can reach must lie
// inside the storage; the receiver's own shape is not consulted beyond that.
func (t *Tensor) AsStrided(dims, strides []int, offset int) (*Tensor, error) {
	if len(dims) != len(strides) {
		return nil, fmt.Errorf("AsStrided: %d dims but %d strides", len(dims), len(strides))
	}
	if offset < 0 {
		return nil, fmt.Errorf("AsStrided: negative offset %d", offset)
	}
	maxIdx := offset
	n := 1
	for i, d := range dims {
		if d < 0 || strides[i] < 0 {
			return nil, fmt.Errorf("AsStrided: dim %d (size %d, stride %d) must be non-negative", i, d, strides[i])
		}
		if d == 0 {
			n = 0
		}
		if d > 0 {
			maxIdx += (d - 1) * strides[i]
		}
	}
	if n != 0 && maxIdx >= t.storage.Len() {
		return nil, fmt.Errorf("AsStrided: view reaches element %d but storage holds %d", maxIdx, t.storage.Len())
	}
	return newTensor(t.storage, shape.New(dims...), shape.NewStrides(strides...), offset), nil
}
