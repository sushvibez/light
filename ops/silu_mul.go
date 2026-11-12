package ops

import (
	"errors"
	"fmt"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// SiLUMulBF16 is silu(g)·u rounded to BF16: a gated MLP's product handed
// straight to a 16-bit down projection (nn.Linear.ForwardMixed16). g and u
// are F32 of one shape. One fused kernel on CUDA; elsewhere SiLU, Mul and
// Cast.
func SiLUMulBF16(g, u *tensor.Tensor) (*tensor.Tensor, error) {
	if !g.Shape().Equal(u.Shape()) || g.DType() != dtype.F32 || u.DType() != dtype.F32 || !g.Device().SameAs(u.Device()) {
		return nil, fmt.Errorf("SiLUMulBF16: want F32 g, u of one shape on one device, got %v %v %v / %v %v %v",
			g.Shape(), g.DType(), g.Device(), u.Shape(), u.DType(), u.Device())
	}
	kern, err := registry.Lookup(registry.Key{Op: "silu_mul_bf16", Dev: g.Device().Kind(), DType: dtype.F32})
	if err != nil {
		if !errors.Is(err, registry.ErrKernelNotFound) {
			return nil, err
		}
		s, err := SiLU(g)
		if err != nil {
			return nil, err
		}
		if s, err = Mul(s, u); err != nil {
			return nil, err
		}
		return Cast(s, dtype.BF16)
	}
	gc, err := ensureContig(g)
	if err != nil {
		return nil, err
	}
	uc, err := ensureContig(u)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Empty(dtype.BF16, g.Shape().Dims(), g.Device())
	if err != nil {
		return nil, err
	}
	if err := kern(&registry.Context{Inputs: []*tensor.Tensor{gc, uc}, Output: out}); err != nil {
		return nil, err
	}
	return out, nil
}
