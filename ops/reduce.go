package ops

import (
	"fmt"
	"sort"

	"light/internal/registry"
	"light/shape"
	"light/tensor"
)

// ReduceOptions covers the subset of reduction options implemented in Phase 1.
type ReduceOptions struct {
	Axes     []int
	KeepDims bool
}

// Sum returns the elementwise sum of a along the given axes.
// If opts.Axes is empty, sums all elements (scalar result, or shape (1,) if
// KeepDims and rank==1, etc.).
func Sum(a *tensor.Tensor, opts ReduceOptions) (*tensor.Tensor, error) {
	return reduceDispatch("sum", a, opts)
}

// Mean returns the elementwise mean of a along the given axes.
// Only supported for float dtypes (F32, F64) in Phase 1.
func Mean(a *tensor.Tensor, opts ReduceOptions) (*tensor.Tensor, error) {
	return reduceDispatch("mean", a, opts)
}

// Max returns the elementwise max along the given axes.
func Max(a *tensor.Tensor, opts ReduceOptions) (*tensor.Tensor, error) {
	return reduceDispatch("max", a, opts)
}

// Min returns the elementwise min along the given axes.
func Min(a *tensor.Tensor, opts ReduceOptions) (*tensor.Tensor, error) {
	return reduceDispatch("min", a, opts)
}

// Prod returns the elementwise product along the given axes.
func Prod(a *tensor.Tensor, opts ReduceOptions) (*tensor.Tensor, error) {
	return reduceDispatch("prod", a, opts)
}

// L2Norm = sqrt(sum(x²)) along the given axes. F32/F64 only.
func L2Norm(a *tensor.Tensor, opts ReduceOptions) (*tensor.Tensor, error) {
	return reduceDispatch("l2norm", a, opts)
}

// All reduces a U8 tensor to 1 where every reduced element is non-zero.
func All(a *tensor.Tensor, opts ReduceOptions) (*tensor.Tensor, error) {
	return reduceDispatch("all", a, opts)
}

// Any reduces a U8 tensor to 1 where any reduced element is non-zero.
func Any(a *tensor.Tensor, opts ReduceOptions) (*tensor.Tensor, error) {
	return reduceDispatch("any", a, opts)
}

func reduceDispatch(op string, a *tensor.Tensor, opts ReduceOptions) (*tensor.Tensor, error) {
	rank := a.Rank()
	axes := append([]int(nil), opts.Axes...)
	if len(axes) == 0 {
		axes = make([]int, rank)
		for i := range axes {
			axes[i] = i
		}
	}
	// Normalize + validate
	seen := make(map[int]bool, len(axes))
	for i, ax := range axes {
		if ax < 0 {
			ax += rank
		}
		if ax < 0 || ax >= rank {
			return nil, fmt.Errorf("%s: axis %d out of range for rank %d", op, opts.Axes[i], rank)
		}
		if seen[ax] {
			return nil, fmt.Errorf("%s: duplicate axis %d", op, ax)
		}
		seen[ax] = true
		axes[i] = ax
	}
	sort.Ints(axes)

	// Compute output shape.
	inDims := a.Shape().Dims()
	isReduced := make([]bool, rank)
	for _, ax := range axes {
		isReduced[ax] = true
	}
	var outDims []int
	for ax := range rank {
		switch {
		case isReduced[ax] && opts.KeepDims:
			outDims = append(outDims, 1)
		case isReduced[ax]:
			// drop
		default:
			outDims = append(outDims, inDims[ax])
		}
	}
	if len(outDims) == 0 {
		outDims = []int{} // scalar
	}

	out, err := tensor.Zeros(a.DType(), outDims, a.Device())
	if err != nil {
		return nil, err
	}
	ctx := &registry.Context{
		Inputs: []*tensor.Tensor{a},
		Output: out,
		Attrs:  map[string]any{"axes": axes, "keepdims": opts.KeepDims},
	}
	k, err := registry.Lookup(registry.Key{Op: op, Dev: a.Device().Kind(), DType: a.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(ctx); err != nil {
		return nil, err
	}
	return out, nil
}

// Ensure unused import suppressed for helper rearrangements.
var _ = shape.DefaultStrides
