package ops

import (
	"fmt"

	"light/internal/registry"
	"light/tensor"
)

// Concat concatenates `tensors` along `axis`. All inputs must have the same
// shape except at `axis`; all must share dtype and device.
func Concat(tensors []*tensor.Tensor, axis int) (*tensor.Tensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("concat: empty input list")
	}
	first := tensors[0]
	dt := first.DType()
	dev := first.Device()
	rank := first.Rank()
	ax, err := first.Shape().NormalizeAxis(axis)
	if err != nil {
		return nil, fmt.Errorf("concat: %w", err)
	}
	// Validate + compute output dim along axis.
	baseDims := first.Shape().Dims()
	catSize := 0
	for i, t := range tensors {
		if t.DType() != dt {
			return nil, fmt.Errorf("concat: dtype mismatch at input %d (%v vs %v)", i, t.DType(), dt)
		}
		if !t.Device().SameAs(dev) {
			return nil, fmt.Errorf("concat: device mismatch at input %d", i)
		}
		if t.Rank() != rank {
			return nil, fmt.Errorf("concat: rank mismatch at input %d (%d vs %d)", i, t.Rank(), rank)
		}
		td := t.Shape().Dims()
		for d := range rank {
			if d == ax {
				continue
			}
			if td[d] != baseDims[d] {
				return nil, fmt.Errorf("concat: dim %d mismatch at input %d (%d vs %d)", d, i, td[d], baseDims[d])
			}
		}
		catSize += td[ax]
	}
	// Ensure all inputs are contiguous for the memcpy-based kernel.
	contigInputs := make([]*tensor.Tensor, len(tensors))
	for i, t := range tensors {
		c, err := ensureContig(t)
		if err != nil {
			return nil, err
		}
		contigInputs[i] = c
	}

	outDims := append([]int(nil), baseDims...)
	outDims[ax] = catSize
	out, err := tensor.Empty(dt, outDims, dev)
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "concat", Dev: dev.Kind(), DType: dt})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: contigInputs,
		Output: out,
		Attrs:  map[string]any{"axis": ax},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// Stack concatenates `tensors` along a new axis at position `axis`. All
// inputs must have identical shape, dtype and device.
func Stack(tensors []*tensor.Tensor, axis int) (*tensor.Tensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("stack: empty input list")
	}
	first := tensors[0]
	// Reshape each input to have a unit dim at `axis`, then Concat.
	// axis can be in range [0, rank] (where rank+1 is equivalent to append).
	newRank := first.Rank() + 1
	if axis < 0 {
		axis += newRank
	}
	if axis < 0 || axis > first.Rank() {
		return nil, fmt.Errorf("stack: axis %d out of range for rank %d (new rank %d)", axis, first.Rank(), newRank)
	}
	expanded := make([]*tensor.Tensor, len(tensors))
	for i, t := range tensors {
		dims := t.Shape().Dims()
		newDims := make([]int, 0, newRank)
		newDims = append(newDims, dims[:axis]...)
		newDims = append(newDims, 1)
		newDims = append(newDims, dims[axis:]...)
		c, err := ensureContig(t)
		if err != nil {
			return nil, err
		}
		r, err := c.Reshape(newDims...)
		if err != nil {
			return nil, err
		}
		expanded[i] = r
	}
	return Concat(expanded, axis)
}
