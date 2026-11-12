package ops

import (
	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// Cast converts `a` to a new dtype, returning a fresh tensor. No-ops when
// the dtype already matches (returns a contiguous copy).
func Cast(a *tensor.Tensor, to dtype.DType) (*tensor.Tensor, error) {
	src, err := ensureContig(a)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Empty(to, src.Shape().Dims(), src.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "cast", Dev: src.Device().Kind(), DType: src.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{Inputs: []*tensor.Tensor{src}, Output: out}); err != nil {
		return nil, err
	}
	return out, nil
}
