package cpu

import (
	"math"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/shape"
)

func init() {
	reg := func(op string, dt dtype.DType, k registry.Kernel) {
		registry.Register(registry.Key{Op: op, Dev: device.KindCPU, DType: dt}, k)
	}
	reg("argmax", dtype.F32, argmaxKernel[float32])
	reg("argmax", dtype.F64, argmaxKernel[float64])
	reg("argmax", dtype.I32, argmaxKernel[int32])
	reg("argmax", dtype.I64, argmaxKernel[int64])
	reg("argmin", dtype.F32, argminKernel[float32])
	reg("argmin", dtype.F64, argminKernel[float64])
	reg("argmin", dtype.I32, argminKernel[int32])
	reg("argmin", dtype.I64, argminKernel[int64])
}

// argmaxKernel scans input along the reduction axis and writes indices to the
// I64 output tensor. Attrs: "axes" (must contain exactly one), "keepdims".
func argmaxKernel[T dtype.Numeric](ctx *registry.Context) error {
	return argKernel[T](ctx, true)
}

func argminKernel[T dtype.Numeric](ctx *registry.Context) error {
	return argKernel[T](ctx, false)
}

func argKernel[T dtype.Numeric](ctx *registry.Context, findMax bool) error {
	a, out := ctx.Inputs[0], ctx.Output
	outS := asSlice[int64](out)
	aS := asSlice[T](a)

	cfg := mustReduceConfig(ctx)
	aDims := a.Shape().Dims()
	rank := len(aDims)

	// argmax/argmin reduces exactly one axis.
	if len(cfg.axes) != 1 {
		// default: reduce all axes → flatten output, but we expect one.
		// For robustness, treat multi-axis the same way: iterate flattened.
	}
	ax := cfg.axes[0]
	if ax < 0 {
		ax += rank
	}

	// Row size = product of axes != ax.
	outerSize := 1
	innerSize := 1
	for i, d := range aDims {
		if i < ax {
			outerSize *= d
		} else if i > ax {
			innerSize *= d
		}
	}
	axSize := aDims[ax]
	_ = shape.DefaultStrides // quiet import

	// For each output position (outer, inner), scan axis.
	for outer := range outerSize {
		for inner := range innerSize {
			bestIdx := int64(0)
			bestVal := aS[outer*axSize*innerSize+0*innerSize+inner]
			for j := 1; j < axSize; j++ {
				v := aS[outer*axSize*innerSize+j*innerSize+inner]
				if (findMax && v > bestVal) || (!findMax && v < bestVal) {
					bestVal = v
					bestIdx = int64(j)
				}
			}
			outS[outer*innerSize+inner] = bestIdx
		}
	}
	_ = math.Inf // quiet import
	return nil
}
