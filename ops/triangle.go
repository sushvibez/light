package ops

import (
	"fmt"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

func init() {
	// Install the Tril/Triu dispatcher so `tensor.Tril(t, k)` / `tensor.Triu`
	// can call back into the ops registry (avoiding the tensor→ops cycle).
	tensor.RegisterTritriHook(tritriDispatch)
}

func tritriDispatch(t *tensor.Tensor, k int, op string) (*tensor.Tensor, error) {
	if t.Rank() < 2 {
		return nil, fmt.Errorf("%s: rank must be >= 2", op)
	}
	if t.DType() != dtype.F32 && t.DType() != dtype.F64 {
		return nil, fmt.Errorf("%s: dtype %v unsupported", op, t.DType())
	}
	src, err := ensureContig(t)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(t.DType(), t.Shape().Dims(), t.Device())
	if err != nil {
		return nil, err
	}
	kernel, err := registry.Lookup(registry.Key{Op: op, Dev: t.Device().Kind(), DType: t.DType()})
	if err != nil {
		return nil, err
	}
	if err := kernel(&registry.Context{
		Inputs: []*tensor.Tensor{src},
		Output: out,
		Attrs:  map[string]any{"k": k},
	}); err != nil {
		return nil, err
	}
	return out, nil
}
