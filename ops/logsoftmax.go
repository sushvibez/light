package ops

import (
	"fmt"

	"light/internal/registry"
	"light/tensor"
)

// LogSoftmax applies log-softmax along the given axis. Numerically stable
// (fused; doesn't just compose Softmax + Log).
func LogSoftmax(a *tensor.Tensor, axis int) (*tensor.Tensor, error) {
	ax, err := a.Shape().NormalizeAxis(axis)
	if err != nil {
		return nil, fmt.Errorf("logsoftmax: %w", err)
	}
	if ax != a.Rank()-1 {
		return nil, fmt.Errorf("logsoftmax: only last-axis supported (got %d)", ax)
	}
	a, err = ensureContig(a)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Empty(a.DType(), a.Shape().Dims(), a.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "logsoftmax", Dev: a.Device().Kind(), DType: a.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{Inputs: []*tensor.Tensor{a}, Output: out}); err != nil {
		return nil, err
	}
	return out, nil
}
