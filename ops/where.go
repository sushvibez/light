package ops

import (
	"fmt"

	"light/dtype"
	"light/internal/registry"
	"light/shape"
	"light/tensor"
)

// Where computes `cond ? a : b` elementwise with broadcasting. cond must be
// U8 (0 = false, non-zero = true); a and b must share dtype and device.
func Where(cond, a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if cond.DType() != dtype.U8 {
		return nil, fmt.Errorf("where: cond must be U8 (got %v)", cond.DType())
	}
	if a.DType() != b.DType() {
		return nil, fmt.Errorf("where: branch dtype mismatch (%v vs %v)", a.DType(), b.DType())
	}
	if !cond.Device().SameAs(a.Device()) || !a.Device().SameAs(b.Device()) {
		return nil, fmt.Errorf("where: device mismatch")
	}
	s, err := shape.Broadcast(cond.Shape(), a.Shape())
	if err != nil {
		return nil, fmt.Errorf("where: %w", err)
	}
	s, err = shape.Broadcast(s, b.Shape())
	if err != nil {
		return nil, fmt.Errorf("where: %w", err)
	}
	out, err := tensor.Zeros(a.DType(), s.Dims(), a.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "where", Dev: a.Device().Kind(), DType: a.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{Inputs: []*tensor.Tensor{cond, a, b}, Output: out}); err != nil {
		return nil, err
	}
	return out, nil
}
