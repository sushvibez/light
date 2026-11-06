package cpu

import (
	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/shape"
)

func init() {
	reg := func(dt dtype.DType, fn registry.Kernel) {
		registry.Register(registry.Key{Op: "where", Dev: device.KindCPU, DType: dt}, fn)
	}
	// Register by the cond/branch dtype pair. cond is always U8; a/b share dtype,
	// which is the dispatch key here.
	reg(dtype.F32, whereKernel[float32])
	reg(dtype.F64, whereKernel[float64])
	reg(dtype.I32, whereKernel[int32])
	reg(dtype.I64, whereKernel[int64])
	reg(dtype.U8, whereKernel[uint8])
	reg(dtype.I8, whereKernel[int8])
	// Half types select raw 16-bit patterns; no arithmetic involved.
	reg(dtype.BF16, whereKernel[uint16])
	reg(dtype.F16, whereKernel[uint16])
}

// whereKernel computes out = cond ? a : b, broadcasting all three to the
// output shape. cond is U8 (0 = false, non-zero = true).
func whereKernel[T dtype.Numeric](ctx *registry.Context) error {
	cond, a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output

	cStr, err := shape.BroadcastStrides(cond.Shape(), cond.Strides(), out.Shape())
	if err != nil {
		return err
	}
	aStr, err := shape.BroadcastStrides(a.Shape(), a.Strides(), out.Shape())
	if err != nil {
		return err
	}
	bStr, err := shape.BroadcastStrides(b.Shape(), b.Strides(), out.Shape())
	if err != nil {
		return err
	}

	cS := asSlice[uint8](cond)
	aS := asSlice[T](a)
	bS := asSlice[T](b)
	outS := asSlice[T](out)

	dims := out.Shape().Dims()
	rank := len(dims)
	idx := make([]int, rank)
	cOffBase := cond.Offset()
	aOffBase := a.Offset()
	bOffBase := b.Offset()
	cStride := cStr.Slice()
	aStride := aStr.Slice()
	bStride := bStr.Slice()

	n := out.NumElements()
	for i := range n {
		cOff := cOffBase
		aOff := aOffBase
		bOff := bOffBase
		for ax := range rank {
			cOff += idx[ax] * cStride[ax]
			aOff += idx[ax] * aStride[ax]
			bOff += idx[ax] * bStride[ax]
		}
		if cS[cOff] != 0 {
			outS[i] = aS[aOff]
		} else {
			outS[i] = bS[bOff]
		}
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
