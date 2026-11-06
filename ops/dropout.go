package ops

import (
	"fmt"
	"math/rand/v2"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// Dropout applies inverted dropout: with probability p the element is zeroed,
// otherwise scaled by 1/(1-p). When training=false, returns x unchanged.
// Dispatches to CPU (via math/rand/v2) or CUDA (via a hash-based device RNG).
func Dropout(x *tensor.Tensor, p float64, training bool) (*tensor.Tensor, error) {
	if !training || p == 0 {
		return x, nil
	}
	if p < 0 || p >= 1 {
		return nil, fmt.Errorf("Dropout: p=%v must be in [0, 1)", p)
	}
	if x.DType() != dtype.F32 && x.DType() != dtype.F64 {
		return nil, fmt.Errorf("Dropout: dtype %v unsupported", x.DType())
	}
	src, err := ensureContig(x)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(x.DType(), x.Shape().Dims(), x.Device())
	if err != nil {
		return nil, err
	}
	scale := 1.0 / (1.0 - p)
	k, err := registry.Lookup(registry.Key{Op: "dropout", Dev: x.Device().Kind(), DType: x.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{src},
		Output: out,
		Attrs:  map[string]any{"p": p, "scale": scale, "seed": rand.Uint64()},
	}); err != nil {
		return nil, err
	}
	return out, nil
}
