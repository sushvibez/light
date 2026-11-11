package cpu

import (
	"light/device"
	"light/dtype"
	"light/internal/parallel"
	"light/internal/registry"
	"light/shape"
	"light/tensor"
)

func init() {
	registry.Register(registry.Key{Op: "add", Dev: device.KindCPU, DType: dtype.F32}, addKernel[float32])
	registry.Register(registry.Key{Op: "add", Dev: device.KindCPU, DType: dtype.F64}, addKernel[float64])
	registry.Register(registry.Key{Op: "add", Dev: device.KindCPU, DType: dtype.I32}, addKernel[int32])
	registry.Register(registry.Key{Op: "add", Dev: device.KindCPU, DType: dtype.I64}, addKernel[int64])
	registry.Register(registry.Key{Op: "add", Dev: device.KindCPU, DType: dtype.U8}, addKernel[uint8])
	registry.Register(registry.Key{Op: "add", Dev: device.KindCPU, DType: dtype.I8}, addKernel[int8])
	// F16 / BF16: emulated via F32 conversion on CPU.
	registry.Register(registry.Key{Op: "add", Dev: device.KindCPU, DType: dtype.F16}, addHalfKernel)
	registry.Register(registry.Key{Op: "add", Dev: device.KindCPU, DType: dtype.BF16}, addBF16Kernel)
}

// addKernel does a + b → out, honoring broadcasts. Fast path: if all three
// tensors are contiguous with offset 0 and have identical shape, it's a
// linear loop. Otherwise it iterates the multi-index.
func addKernel[T dtype.Numeric](ctx *registry.Context) error {
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

	// Fast path: a and b are same-shape as out and contiguous.
	aFast := a.IsContiguous() && a.Offset() == 0 && a.Shape().Equal(out.Shape())
	bFast := b.IsContiguous() && b.Offset() == 0 && b.Shape().Equal(out.Shape())
	if aFast && bFast {
		parallel.For(n, parallel.Grain, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				outS[i] = aS[i] + bS[i]
			}
		})
		return nil
	}
	aStr, bStr := aStridesB.Slice(), bStridesB.Slice()
	parallel.For(n, parallel.Grain, func(lo, hi int) {
		parallel.Strided(dims, aStr, bStr, a.Offset(), b.Offset(), lo, hi, func(i, ao, bo int) { outS[i] = aS[ao] + bS[bo] })
	})
	return nil
}

// addHalfKernel performs F16 + F16 by upcasting to F32, adding, and
// downcasting. Matches Candle's CPU semantics.
func addHalfKernel(ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	aS := asSlice[uint16](a) // Half uses uint16 backing
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
		outS[i] = uint16(dtype.HalfFromF32(av + bv))
	})
	return nil
}

// addBF16Kernel is the BF16 analogue.
func addBF16Kernel(ctx *registry.Context) error {
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
		outS[i] = uint16(dtype.BF16FromF32(av + bv))
	})
	return nil
}

// iterate walks every element of t's shape in row-major order, calling fn(i, idx).
func iterate(t *tensor.Tensor, fn func(i int, idx []int)) {
	dims := t.Shape().Dims()
	rank := len(dims)
	idx := make([]int, rank)
	n := t.NumElements()
	for i := range n {
		fn(i, idx)
		for ax := rank - 1; ax >= 0; ax-- {
			idx[ax]++
			if idx[ax] < dims[ax] {
				break
			}
			idx[ax] = 0
		}
	}
}

func dotIdx(idx, strides []int) int {
	s := 0
	for i, ix := range idx {
		s += ix * strides[i]
	}
	return s
}
