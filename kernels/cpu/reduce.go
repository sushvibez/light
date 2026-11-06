package cpu

import (
	"math"

	"light/dtype"
	"light/internal/registry"
	"light/shape"
)

// reduceConfig extracts common reduction parameters from the op context.
type reduceConfig struct {
	axes     []int
	keepdims bool
}

func mustReduceConfig(ctx *registry.Context) reduceConfig {
	axes, _ := ctx.Attrs["axes"].([]int)
	kd, _ := ctx.Attrs["keepdims"].(bool)
	return reduceConfig{axes: axes, keepdims: kd}
}

// reduceWalk visits every input element and calls fn(outIdx, val) where
// outIdx is the flat offset into the output tensor. Used by sum/max/min kernels.
func reduceWalk[T dtype.Numeric](ctx *registry.Context, fn func(outIdx int, v T)) {
	a := ctx.Inputs[0]
	out := ctx.Output
	cfg := mustReduceConfig(ctx)

	aS := asSlice[T](a)

	aDims := a.Shape().Dims()
	rank := len(aDims)
	isReduced := make([]bool, rank)
	for _, ax := range cfg.axes {
		if ax < 0 {
			ax += rank
		}
		isReduced[ax] = true
	}

	outStr := shape.DefaultStrides(out.Shape()).Slice()

	idx := make([]int, rank)
	aStr := a.Strides().Slice()
	aBase := a.Offset()
	n := a.NumElements()

	for range n {
		aOff := aBase
		outOff := 0
		outAx := 0
		for ax := range rank {
			aOff += idx[ax] * aStr[ax]
			if isReduced[ax] {
				if cfg.keepdims {
					outAx++
				}
			} else {
				outOff += idx[ax] * outStr[outAx]
				outAx++
			}
		}
		fn(outOff, aS[aOff])
		for ax := rank - 1; ax >= 0; ax-- {
			idx[ax]++
			if idx[ax] < aDims[ax] {
				break
			}
			idx[ax] = 0
		}
	}
}

// reduceSumKernel implements Sum for any Numeric dtype.
func reduceSumKernel[T dtype.Numeric](ctx *registry.Context) error {
	out := ctx.Output
	outS := asSlice[T](out)
	var zero T
	for i := range outS {
		outS[i] = zero
	}
	reduceWalk[T](ctx, func(i int, v T) {
		outS[i] += v
	})
	return nil
}

// reduceMeanKernel computes Sum / count where count is the product of reduced
// axis sizes.
func reduceMeanKernel[T dtype.Numeric](ctx *registry.Context) error {
	if err := reduceSumKernel[T](ctx); err != nil {
		return err
	}
	a := ctx.Inputs[0]
	cfg := mustReduceConfig(ctx)
	count := 1
	for _, ax := range cfg.axes {
		if ax < 0 {
			ax += a.Shape().Rank()
		}
		count *= a.Shape().Dim(ax)
	}
	if count == 0 {
		return nil
	}
	outS := asSlice[T](ctx.Output)
	divisor := T(count)
	for i := range outS {
		outS[i] /= divisor
	}
	return nil
}

// reduceMaxKernel with per-element identity = -Inf for floats, MinInt for ints.
func reduceMaxKernel[T dtype.Numeric](ctx *registry.Context) error {
	out := ctx.Output
	outS := asSlice[T](out)
	// Initialize with first input elements by scanning (avoids type-specific bounds).
	initialized := make([]bool, len(outS))
	reduceWalk[T](ctx, func(i int, v T) {
		if !initialized[i] || v > outS[i] {
			outS[i] = v
			initialized[i] = true
		}
	})
	return nil
}

// reduceMinKernel mirrors reduceMax.
func reduceMinKernel[T dtype.Numeric](ctx *registry.Context) error {
	out := ctx.Output
	outS := asSlice[T](out)
	initialized := make([]bool, len(outS))
	reduceWalk[T](ctx, func(i int, v T) {
		if !initialized[i] || v < outS[i] {
			outS[i] = v
			initialized[i] = true
		}
	})
	return nil
}

// reduceHalfSum goes through F32 for F16 / BF16 accumulators.
func reduceHalfSum(ctx *registry.Context, isBF16 bool) error {
	a := ctx.Inputs[0]
	out := ctx.Output
	aS := asSlice[uint16](a)
	outS := asSlice[uint16](out)

	cfg := mustReduceConfig(ctx)
	aDims := a.Shape().Dims()
	rank := len(aDims)
	isReduced := make([]bool, rank)
	for _, ax := range cfg.axes {
		if ax < 0 {
			ax += rank
		}
		isReduced[ax] = true
	}
	outStr := shape.DefaultStrides(out.Shape()).Slice()

	acc := make([]float32, len(outS))
	idx := make([]int, rank)
	aStr := a.Strides().Slice()
	aBase := a.Offset()
	n := a.NumElements()
	for range n {
		aOff := aBase
		outOff := 0
		outAx := 0
		for ax := range rank {
			aOff += idx[ax] * aStr[ax]
			if isReduced[ax] {
				if cfg.keepdims {
					outAx++
				}
			} else {
				outOff += idx[ax] * outStr[outAx]
				outAx++
			}
		}
		var v float32
		if isBF16 {
			v = dtype.BFloat16(aS[aOff]).ToF32()
		} else {
			v = dtype.Half(aS[aOff]).ToF32()
		}
		acc[outOff] += v
		for ax := rank - 1; ax >= 0; ax-- {
			idx[ax]++
			if idx[ax] < aDims[ax] {
				break
			}
			idx[ax] = 0
		}
	}
	for i, v := range acc {
		if isBF16 {
			outS[i] = uint16(dtype.BF16FromF32(v))
		} else {
			outS[i] = uint16(dtype.HalfFromF32(v))
		}
	}
	return nil
}

func init() {
	reg := func(op string, dt dtype.DType, k registry.Kernel) {
		registry.Register(registry.Key{Op: op, Dev: 0, DType: dt}, k)
	}
	// Sum
	reg("sum", dtype.F32, reduceSumKernel[float32])
	reg("sum", dtype.F64, reduceSumKernel[float64])
	reg("sum", dtype.I32, reduceSumKernel[int32])
	reg("sum", dtype.I64, reduceSumKernel[int64])
	reg("sum", dtype.F16, func(ctx *registry.Context) error { return reduceHalfSum(ctx, false) })
	reg("sum", dtype.BF16, func(ctx *registry.Context) error { return reduceHalfSum(ctx, true) })
	// Mean
	reg("mean", dtype.F32, reduceMeanKernel[float32])
	reg("mean", dtype.F64, reduceMeanKernel[float64])
	// Max
	reg("max", dtype.F32, reduceMaxKernel[float32])
	reg("max", dtype.F64, reduceMaxKernel[float64])
	reg("max", dtype.I32, reduceMaxKernel[int32])
	reg("max", dtype.I64, reduceMaxKernel[int64])
	// Min
	reg("min", dtype.F32, reduceMinKernel[float32])
	reg("min", dtype.F64, reduceMinKernel[float64])
	reg("min", dtype.I32, reduceMinKernel[int32])
	reg("min", dtype.I64, reduceMinKernel[int64])
	// Prod
	reg("prod", dtype.F32, reduceProdKernel[float32])
	reg("prod", dtype.F64, reduceProdKernel[float64])
	reg("prod", dtype.I32, reduceProdKernel[int32])
	reg("prod", dtype.I64, reduceProdKernel[int64])
	// L2Norm — sqrt(sum(x²))
	reg("l2norm", dtype.F32, reduceL2NormF32)
	reg("l2norm", dtype.F64, reduceL2NormF64)
	// All / Any on U8
	reg("all", dtype.U8, reduceAllKernel)
	reg("any", dtype.U8, reduceAnyKernel)
	_ = math.Inf
}

// reduceProdKernel computes elementwise product along the reduced axes.
func reduceProdKernel[T dtype.Numeric](ctx *registry.Context) error {
	out := ctx.Output
	outS := asSlice[T](out)
	var one T = 1
	for i := range outS {
		outS[i] = one
	}
	reduceWalk[T](ctx, func(i int, v T) {
		outS[i] *= v
	})
	return nil
}

// reduceL2NormF32 = sqrt(sum(x²)) along the reduced axes.
func reduceL2NormF32(ctx *registry.Context) error {
	out := ctx.Output
	outS := asSlice[float32](out)
	for i := range outS {
		outS[i] = 0
	}
	reduceWalk[float32](ctx, func(i int, v float32) {
		outS[i] += v * v
	})
	for i, v := range outS {
		outS[i] = float32(math.Sqrt(float64(v)))
	}
	return nil
}

func reduceL2NormF64(ctx *registry.Context) error {
	out := ctx.Output
	outS := asSlice[float64](out)
	for i := range outS {
		outS[i] = 0
	}
	reduceWalk[float64](ctx, func(i int, v float64) {
		outS[i] += v * v
	})
	for i, v := range outS {
		outS[i] = math.Sqrt(v)
	}
	return nil
}

// reduceAllKernel: U8 → U8. Output is 1 iff all inputs in the reduced slice
// are non-zero.
func reduceAllKernel(ctx *registry.Context) error {
	out := ctx.Output
	outS := asSlice[uint8](out)
	for i := range outS {
		outS[i] = 1
	}
	reduceWalk[uint8](ctx, func(i int, v uint8) {
		if v == 0 {
			outS[i] = 0
		}
	})
	return nil
}

// reduceAnyKernel: U8 → U8. Output is 1 iff any input in the reduced slice is non-zero.
func reduceAnyKernel(ctx *registry.Context) error {
	out := ctx.Output
	outS := asSlice[uint8](out)
	for i := range outS {
		outS[i] = 0
	}
	reduceWalk[uint8](ctx, func(i int, v uint8) {
		if v != 0 {
			outS[i] = 1
		}
	})
	return nil
}
