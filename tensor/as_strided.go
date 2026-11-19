package tensor

import (
	"fmt"
	"math"

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
	empty := false
	numel := 1 // saturates; only its overflow matters (zero strides can broadcast huge dims)
	for i, d := range dims {
		if d < 0 || strides[i] < 0 {
			return nil, fmt.Errorf("AsStrided: dim %d (size %d, stride %d) must be non-negative", i, d, strides[i])
		}
		if d == 0 {
			empty = true
		} else if numel > math.MaxInt/d {
			numel = math.MaxInt
		} else {
			numel *= d
		}
	}
	if !empty && numel == math.MaxInt {
		return nil, fmt.Errorf("AsStrided: dims %v element count overflows", dims)
	}
	if !empty {
		// The highest reachable element is offset + Σ (d-1)·stride. Track
		// the storage still free past it instead of the sum itself, so no
		// multiplication or addition can wrap and admit an invalid view.
		room := t.storage.Len() - 1 - offset // elements left after offset
		if room < 0 {
			return nil, fmt.Errorf("AsStrided: offset %d outside storage of %d elements", offset, t.storage.Len())
		}
		for i, d := range dims {
			if d <= 1 || strides[i] == 0 {
				continue
			}
			if strides[i] > room/(d-1) {
				return nil, fmt.Errorf("AsStrided: dims %v strides %v offset %d reach past storage of %d elements", dims, strides, offset, t.storage.Len())
			}
			room -= (d - 1) * strides[i]
		}
	}
	return newTensor(t.storage, shape.New(dims...), shape.NewStrides(strides...), offset), nil
}
