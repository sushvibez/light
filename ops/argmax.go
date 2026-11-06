package ops

import (
	"fmt"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// Argmax returns the indices of the maximum values along a single axis.
// Output dtype is always I64. If keepdims is false, the reduced axis is
// dropped; otherwise it has size 1.
func Argmax(a *tensor.Tensor, axis int, keepdims bool) (*tensor.Tensor, error) {
	return argDispatch("argmax", a, axis, keepdims)
}

// Argmin is the index-of-min counterpart of Argmax.
func Argmin(a *tensor.Tensor, axis int, keepdims bool) (*tensor.Tensor, error) {
	return argDispatch("argmin", a, axis, keepdims)
}

func argDispatch(op string, a *tensor.Tensor, axis int, keepdims bool) (*tensor.Tensor, error) {
	ax, err := a.Shape().NormalizeAxis(axis)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	a, err = ensureContig(a)
	if err != nil {
		return nil, err
	}
	// Output shape = a.shape with the reduced axis either dropped or set to 1.
	dims := a.Shape().Dims()
	var outDims []int
	for i, d := range dims {
		if i == ax {
			if keepdims {
				outDims = append(outDims, 1)
			}
		} else {
			outDims = append(outDims, d)
		}
	}
	out, err := tensor.Zeros(dtype.I64, outDims, a.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: op, Dev: a.Device().Kind(), DType: a.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{a},
		Output: out,
		Attrs:  map[string]any{"axes": []int{ax}, "keepdims": keepdims},
	}); err != nil {
		return nil, err
	}
	return out, nil
}
