package cpu

import (
	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/shape"
)

// Comparison ops: broadcast two tensors of the same Numeric dtype and write
// 0/1 into a U8 output. Logical ops take U8 inputs (0/1) and operate the
// same way.

func init() {
	reg := func(op string, dt dtype.DType, fn registry.Kernel) {
		registry.Register(registry.Key{Op: op, Dev: device.KindCPU, DType: dt}, fn)
	}
	dts := []dtype.DType{
		dtype.F32, dtype.F64, dtype.F16, dtype.BF16, dtype.I32, dtype.I64,
		dtype.I8, dtype.U8, dtype.I16, dtype.U16, dtype.U32, dtype.U64,
	}
	for _, dt := range dts {
		reg("lt", dt, cmpKernelFor(dt, func(a, b float64) bool { return a < b }))
		reg("le", dt, cmpKernelFor(dt, func(a, b float64) bool { return a <= b }))
		reg("gt", dt, cmpKernelFor(dt, func(a, b float64) bool { return a > b }))
		reg("ge", dt, cmpKernelFor(dt, func(a, b float64) bool { return a >= b }))
		reg("eq", dt, cmpKernelFor(dt, func(a, b float64) bool { return a == b }))
		reg("ne", dt, cmpKernelFor(dt, func(a, b float64) bool { return a != b }))
	}
	// Logical ops: U8 in, U8 out. Reuse the existing binary/unary helpers
	// with uint8 callbacks.
	reg("logical_and", dtype.U8, func(ctx *registry.Context) error {
		return binaryOp(ctx, func(a, b uint8) uint8 {
			if a != 0 && b != 0 {
				return 1
			}
			return 0
		})
	})
	reg("logical_or", dtype.U8, func(ctx *registry.Context) error {
		return binaryOp(ctx, func(a, b uint8) uint8 {
			if a != 0 || b != 0 {
				return 1
			}
			return 0
		})
	})
	reg("logical_not", dtype.U8, func(ctx *registry.Context) error {
		return unaryOp(ctx, func(a uint8) uint8 {
			if a == 0 {
				return 1
			}
			return 0
		})
	})
}

// cmpKernelFor returns a kernel for the given input dtype.  The kernel does
// a broadcast-aware walk of the output, reads each position from `a` and `b`
// as float64 (via readF64Buf), applies `cmp`, and writes 0/1 to U8 output.
func cmpKernelFor(dt dtype.DType, cmp func(a, b float64) bool) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output

		aStr, err := shape.BroadcastStrides(a.Shape(), a.Strides(), out.Shape())
		if err != nil {
			return err
		}
		bStr, err := shape.BroadcastStrides(b.Shape(), b.Strides(), out.Shape())
		if err != nil {
			return err
		}

		aBuf := a.Storage().CPUBytes()
		bBuf := b.Storage().CPUBytes()
		outS := asSlice[uint8](out)

		dims := out.Shape().Dims()
		rank := len(dims)
		idx := make([]int, rank)
		aOffBase := a.Offset()
		bOffBase := b.Offset()
		aStride := aStr.Slice()
		bStride := bStr.Slice()

		n := out.NumElements()
		for i := range n {
			aOff := aOffBase
			bOff := bOffBase
			for ax := range rank {
				aOff += idx[ax] * aStride[ax]
				bOff += idx[ax] * bStride[ax]
			}
			av := readF64Buf(dt, aBuf, aOff)
			bv := readF64Buf(dt, bBuf, bOff)
			if cmp(av, bv) {
				outS[i] = 1
			} else {
				outS[i] = 0
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
}
