package cpu

import (
	"math"

	"light/dtype"
	"light/internal/registry"
)

// softmaxKernel implements numerically-stable softmax along a single axis:
//
//	out = exp(a - max(a, axis)) / sum(exp(a - max(a, axis)), axis)
//
// ctx.Attrs["axis"] is the positive axis index.
func softmaxKernel[T numericFloat](ctx *registry.Context) error {
	a := ctx.Inputs[0]
	out := ctx.Output
	axis, _ := ctx.Attrs["axis"].(int)

	aS := asSlice[T](a)
	outS := asSlice[T](out)

	dims := a.Shape().Dims()
	if axis < 0 {
		axis += len(dims)
	}
	rank := len(dims)

	// Stride of the softmax axis in elements.
	aStr := a.Strides().Slice()
	outStr := a.Strides().Slice() // out has same default strides since shape matches
	_ = outStr
	axStride := aStr[axis]
	axLen := dims[axis]

	// Iterate over all "rows" (product of non-axis dims).
	otherDims := make([]int, 0, rank-1)
	otherAxes := make([]int, 0, rank-1)
	for i := range rank {
		if i != axis {
			otherDims = append(otherDims, dims[i])
			otherAxes = append(otherAxes, i)
		}
	}
	idx := make([]int, rank)
	var rowCount int
	if len(otherDims) == 0 {
		rowCount = 1
	} else {
		rowCount = 1
		for _, d := range otherDims {
			rowCount *= d
		}
	}

	aBase := a.Offset()
	// output is contiguous, default strides.
	outDStr := make([]int, rank)
	// default strides of out
	{
		s := 1
		for k := rank - 1; k >= 0; k-- {
			outDStr[k] = s
			s *= dims[k]
		}
	}

	for r := range rowCount {
		// Compute base offsets for this row: axis index is 0.
		aRowOff := aBase
		outRowOff := 0
		for ax := range rank {
			if ax == axis {
				continue
			}
			aRowOff += idx[ax] * aStr[ax]
			outRowOff += idx[ax] * outDStr[ax]
		}

		// Stable softmax: subtract max.
		maxVal := aS[aRowOff]
		for k := 1; k < axLen; k++ {
			v := aS[aRowOff+k*axStride]
			if v > maxVal {
				maxVal = v
			}
		}
		var sum T
		for k := range axLen {
			e := T(math.Exp(float64(aS[aRowOff+k*axStride] - maxVal)))
			outS[outRowOff+k*outDStr[axis]] = e
			sum += e
		}
		for k := range axLen {
			outS[outRowOff+k*outDStr[axis]] /= sum
		}

		// Advance multi-index over non-axis dims.
		for i := len(otherAxes) - 1; i >= 0; i-- {
			ax := otherAxes[i]
			idx[ax]++
			if idx[ax] < dims[ax] {
				break
			}
			idx[ax] = 0
		}
		_ = r
	}
	return nil
}

// numericFloat narrows the generic constraint to floats that softmax uses.
type numericFloat interface {
	~float32 | ~float64
}

func init() {
	registry.Register(registry.Key{Op: "softmax", Dev: 0, DType: dtype.F32}, softmaxKernel[float32])
	registry.Register(registry.Key{Op: "softmax", Dev: 0, DType: dtype.F64}, softmaxKernel[float64])
}
