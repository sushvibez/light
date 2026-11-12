package ops

import (
	"fmt"

	"light/internal/registry"
	"light/tensor"
)

// Softmax applies softmax along the given axis. Numerically stable
// (subtracts row-max before exponentiation).
func Softmax(a *tensor.Tensor, axis int) (*tensor.Tensor, error) {
	ax, err := a.Shape().NormalizeAxis(axis)
	if err != nil {
		return nil, fmt.Errorf("softmax: %w", err)
	}
	out, err := tensor.Empty(a.DType(), a.Shape().Dims(), a.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "softmax", Dev: a.Device().Kind(), DType: a.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{a},
		Output: out,
		Attrs:  map[string]any{"axis": ax},
	}); err != nil {
		return nil, err
	}
	return out, nil
}
