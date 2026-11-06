//go:build !nocuda && cgo

package cuda

import (
	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
)

func init() {
	// Register the same launcher for every supported dtype — the kernel is
	// pure memcpy so dtype only affects element size.
	for _, dt := range []dtype.DType{
		dtype.F32, dtype.F64, dtype.F16, dtype.BF16,
		dtype.I8, dtype.U8, dtype.I16, dtype.U16,
		dtype.I32, dtype.U32, dtype.I64, dtype.U64,
	} {
		registry.Register(registry.Key{Op: "concat", Dev: device.KindCUDA, DType: dt}, concatCUDALauncher)
	}
}

// concatCUDALauncher concatenates ctx.Inputs along Attrs["axis"]: one
// cudaMemcpy2D per input copies its `outer` rows of `axisLen·inner` elements
// into the output's wider rows. No kernel, and no per-row calls: a KV-cache
// append over 32 heads is two copies, not 64.
func concatCUDALauncher(ctx *registry.Context) error {
	axis := ctx.Attrs["axis"].(int)
	out := ctx.Output
	elemSize := out.DType().Size()

	outDims := out.Shape().Dims()
	outer := 1
	inner := 1
	for i, d := range outDims {
		if i < axis {
			outer *= d
		} else if i > axis {
			inner *= d
		}
	}
	innerBytes := uintptr(inner * elemSize)
	outAxisLen := outDims[axis]
	dstRowStride := uintptr(outAxisLen) * innerBytes

	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	dstBase := out.Storage().CUDAPtr()

	var dstOff uintptr
	for _, src := range ctx.Inputs {
		srcRowStride := uintptr(src.Shape().Dims()[axis]) * innerBytes
		srcBase := src.Storage().CUDAPtr() + uintptr(src.Offset()*elemSize)
		if err := cudart.Memcpy2DDtoD(dstBase+dstOff, dstRowStride, srcBase, srcRowStride, srcRowStride, uintptr(outer)); err != nil {
			return err
		}
		dstOff += srcRowStride
	}
	return nil
}
