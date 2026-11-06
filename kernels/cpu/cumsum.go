package cpu

import (
	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	reg := func(op string, dt dtype.DType, k registry.Kernel) {
		registry.Register(registry.Key{Op: op, Dev: device.KindCPU, DType: dt}, k)
	}
	reg("cumsum", dtype.F32, cumsumKernel[float32])
	reg("cumsum", dtype.F64, cumsumKernel[float64])
	reg("cumsum", dtype.I32, cumsumKernel[int32])
	reg("cumsum", dtype.I64, cumsumKernel[int64])
	reg("cumprod", dtype.F32, cumprodKernel[float32])
	reg("cumprod", dtype.F64, cumprodKernel[float64])
	reg("cumprod", dtype.I32, cumprodKernel[int32])
	reg("cumprod", dtype.I64, cumprodKernel[int64])
}

// cumsumKernel writes running prefix sums along ctx.Attrs["dim"].
// Input and output have the same shape. Caller contiguifies both.
func cumsumKernel[T dtype.Numeric](ctx *registry.Context) error {
	return scanKernel[T](ctx, func(acc, v T) T { return acc + v }, 0)
}

func cumprodKernel[T dtype.Numeric](ctx *registry.Context) error {
	return scanKernel[T](ctx, func(acc, v T) T { return acc * v }, 1)
}

func scanKernel[T dtype.Numeric](ctx *registry.Context, combine func(T, T) T, init T) error {
	a := ctx.Inputs[0]
	out := ctx.Output
	dim := ctx.Attrs["dim"].(int)

	aS := asSlice[T](a)
	outS := asSlice[T](out)
	dims := a.Shape().Dims()
	rank := len(dims)
	axSize := dims[dim]

	outer := 1
	for i := 0; i < dim; i++ {
		outer *= dims[i]
	}
	inner := 1
	for i := dim + 1; i < rank; i++ {
		inner *= dims[i]
	}

	for o := range outer {
		for in := range inner {
			base := o*axSize*inner + in
			acc := init
			for j := range axSize {
				acc = combine(acc, aS[base+j*inner])
				outS[base+j*inner] = acc
			}
		}
	}
	return nil
}
