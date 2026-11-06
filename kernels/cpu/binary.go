package cpu

import (
	"math"

	"light/dtype"
	"light/internal/registry"
	"light/shape"
)

// binaryOp runs a broadcasting-aware elementwise binary kernel fn(a,b) over
// any Numeric dtype. Contiguous same-shape inputs take the fast linear path.
func binaryOp[T dtype.Numeric](ctx *registry.Context, fn func(a, b T) T) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output

	aStridesB, err := shape.BroadcastStrides(a.Shape(), a.Strides(), out.Shape())
	if err != nil {
		return err
	}
	bStridesB, err := shape.BroadcastStrides(b.Shape(), b.Strides(), out.Shape())
	if err != nil {
		return err
	}

	aS := asSlice[T](a)
	bS := asSlice[T](b)
	outS := asSlice[T](out)

	dims := out.Shape().Dims()
	n := out.NumElements()

	aFast := a.IsContiguous() && a.Offset() == 0 && a.Shape().Equal(out.Shape())
	bFast := b.IsContiguous() && b.Offset() == 0 && b.Shape().Equal(out.Shape())
	if aFast && bFast {
		for i := range n {
			outS[i] = fn(aS[i], bS[i])
		}
		return nil
	}

	rank := len(dims)
	idx := make([]int, rank)
	aStride := aStridesB.Slice()
	bStride := bStridesB.Slice()
	aBase := a.Offset()
	bBase := b.Offset()
	for i := range n {
		aOff := aBase
		bOff := bBase
		for ax := range rank {
			aOff += idx[ax] * aStride[ax]
			bOff += idx[ax] * bStride[ax]
		}
		outS[i] = fn(aS[aOff], bS[bOff])
		for ax := rank - 1; ax >= 0; ax-- {
			idx[ax]++
			if idx[ax] < dims[ax] {
				break
			}
			idx[ax] = 0
		}
	}
	return nil
}

// binaryHalfOp is the F16 version: computes in F32 internally, rounds back to F16.
func binaryHalfOp(ctx *registry.Context, fn func(a, b float32) float32) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	aS := asSlice[uint16](a)
	bS := asSlice[uint16](b)
	outS := asSlice[uint16](out)

	aStridesB, err := shape.BroadcastStrides(a.Shape(), a.Strides(), out.Shape())
	if err != nil {
		return err
	}
	bStridesB, err := shape.BroadcastStrides(b.Shape(), b.Strides(), out.Shape())
	if err != nil {
		return err
	}

	iterate(out, func(i int, idx []int) {
		aOff := a.Offset() + dotIdx(idx, aStridesB.Slice())
		bOff := b.Offset() + dotIdx(idx, bStridesB.Slice())
		av := dtype.Half(aS[aOff]).ToF32()
		bv := dtype.Half(bS[bOff]).ToF32()
		outS[i] = uint16(dtype.HalfFromF32(fn(av, bv)))
	})
	return nil
}

// binaryBF16Op is the BF16 version.
func binaryBF16Op(ctx *registry.Context, fn func(a, b float32) float32) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	aS := asSlice[uint16](a)
	bS := asSlice[uint16](b)
	outS := asSlice[uint16](out)

	aStridesB, err := shape.BroadcastStrides(a.Shape(), a.Strides(), out.Shape())
	if err != nil {
		return err
	}
	bStridesB, err := shape.BroadcastStrides(b.Shape(), b.Strides(), out.Shape())
	if err != nil {
		return err
	}

	iterate(out, func(i int, idx []int) {
		aOff := a.Offset() + dotIdx(idx, aStridesB.Slice())
		bOff := b.Offset() + dotIdx(idx, bStridesB.Slice())
		av := dtype.BFloat16(aS[aOff]).ToF32()
		bv := dtype.BFloat16(bS[bOff]).ToF32()
		outS[i] = uint16(dtype.BF16FromF32(fn(av, bv)))
	})
	return nil
}

// registerBinary wires a single binary op across every supported dtype.
// numericFn is called for non-half dtypes; halfFn is the F32-internal form used
// by F16 and BF16 kernels.
func registerBinary(
	op string,
	numericFn func(a, b any) any,
	f32 func(a, b float32) float32,
	f64 func(a, b float64) float64,
	i32 func(a, b int32) int32,
	i64 func(a, b int64) int64,
	u8 func(a, b uint8) uint8,
	i8 func(a, b int8) int8,
) {
	reg := func(dt dtype.DType, k registry.Kernel) {
		registry.Register(registry.Key{Op: op, Dev: 0, DType: dt}, k)
	}
	_ = numericFn
	reg(dtype.F32, func(ctx *registry.Context) error { return binaryOp(ctx, f32) })
	reg(dtype.F64, func(ctx *registry.Context) error { return binaryOp(ctx, f64) })
	reg(dtype.I32, func(ctx *registry.Context) error { return binaryOp(ctx, i32) })
	reg(dtype.I64, func(ctx *registry.Context) error { return binaryOp(ctx, i64) })
	reg(dtype.U8, func(ctx *registry.Context) error { return binaryOp(ctx, u8) })
	reg(dtype.I8, func(ctx *registry.Context) error { return binaryOp(ctx, i8) })
	reg(dtype.F16, func(ctx *registry.Context) error { return binaryHalfOp(ctx, f32) })
	reg(dtype.BF16, func(ctx *registry.Context) error { return binaryBF16Op(ctx, f32) })
}

func init() {
	registerBinary("sub", nil,
		func(a, b float32) float32 { return a - b },
		func(a, b float64) float64 { return a - b },
		func(a, b int32) int32 { return a - b },
		func(a, b int64) int64 { return a - b },
		func(a, b uint8) uint8 { return a - b },
		func(a, b int8) int8 { return a - b },
	)
	registerBinary("mul", nil,
		func(a, b float32) float32 { return a * b },
		func(a, b float64) float64 { return a * b },
		func(a, b int32) int32 { return a * b },
		func(a, b int64) int64 { return a * b },
		func(a, b uint8) uint8 { return a * b },
		func(a, b int8) int8 { return a * b },
	)
	registerBinary("div", nil,
		func(a, b float32) float32 { return a / b },
		func(a, b float64) float64 { return a / b },
		func(a, b int32) int32 {
			if b == 0 {
				return 0
			}
			return a / b
		},
		func(a, b int64) int64 {
			if b == 0 {
				return 0
			}
			return a / b
		},
		func(a, b uint8) uint8 {
			if b == 0 {
				return 0
			}
			return a / b
		},
		func(a, b int8) int8 {
			if b == 0 {
				return 0
			}
			return a / b
		},
	)
	registerBinary("pow", nil,
		func(a, b float32) float32 { return f32Pow(a, b) },
		func(a, b float64) float64 { return f64Pow(a, b) },
		func(a, b int32) int32 { return intPow(a, int32(b)) },
		func(a, b int64) int64 { return int64Pow(a, b) },
		func(a, b uint8) uint8 { return uint8(intPow(int32(a), int32(b))) },
		func(a, b int8) int8 { return int8(intPow(int32(a), int32(b))) },
	)
}

// maximum / minimum (2026-10-28).

func init() {
	registerBinary("maximum", nil,
		func(a, b float32) float32 { return float32(math.Max(float64(a), float64(b))) },
		math.Max,
		func(a, b int32) int32 { return max(a, b) },
		func(a, b int64) int64 { return max(a, b) },
		func(a, b uint8) uint8 { return max(a, b) },
		func(a, b int8) int8 { return max(a, b) },
	)
	registerBinary("minimum", nil,
		func(a, b float32) float32 { return float32(math.Min(float64(a), float64(b))) },
		math.Min,
		func(a, b int32) int32 { return min(a, b) },
		func(a, b int64) int64 { return min(a, b) },
		func(a, b uint8) uint8 { return min(a, b) },
		func(a, b int8) int8 { return min(a, b) },
	)
}
