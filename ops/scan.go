package ops

import (
	"fmt"

	"light/internal/registry"
	"light/tensor"
)

// Cumsum returns the running prefix sum along `dim`. Output has the same
// shape and dtype as the input.
func Cumsum(a *tensor.Tensor, dim int) (*tensor.Tensor, error) { return scanDispatch("cumsum", a, dim) }

// Cumprod returns the running prefix product along `dim`.
func Cumprod(a *tensor.Tensor, dim int) (*tensor.Tensor, error) {
	return scanDispatch("cumprod", a, dim)
}

func scanDispatch(op string, a *tensor.Tensor, dim int) (*tensor.Tensor, error) {
	d, err := a.Shape().NormalizeAxis(dim)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	a, err = ensureContig(a)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(a.DType(), a.Shape().Dims(), a.Device())
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
		Attrs:  map[string]any{"dim": d},
	}); err != nil {
		return nil, err
	}
	return out, nil
}
