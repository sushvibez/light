package cpu

import (
	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	// Concat accepts mixed-count inputs; dispatch by the first input's dtype.
	for _, dt := range []dtype.DType{
		dtype.F32, dtype.F64, dtype.F16, dtype.BF16,
		dtype.I8, dtype.U8, dtype.I16, dtype.U16,
		dtype.I32, dtype.U32, dtype.I64, dtype.U64,
	} {
		registry.Register(registry.Key{Op: "concat", Dev: device.KindCPU, DType: dt}, concatKernel)
	}
}

// concatKernel concatenates Inputs along the axis in Attrs["axis"]. All
// inputs must share the same shape along every axis EXCEPT the cat axis.
// Tensors are assumed contiguous (ops.Concat enforces).
func concatKernel(ctx *registry.Context) error {
	axis := ctx.Attrs["axis"].(int)
	out := ctx.Output
	elemSize := out.DType().Size()
	dstBuf := out.Storage().CPUBytes()

	outDims := out.Shape().Dims()
	// "outer" = product of axes before `axis`. "inner" = product of axes after.
	outer := 1
	inner := 1
	for i, d := range outDims {
		if i < axis {
			outer *= d
		} else if i > axis {
			inner *= d
		}
	}
	// For each outer slice, interleave the inputs along the cat axis.
	innerBytes := inner * elemSize
	dstRowStride := outDims[axis] * innerBytes

	for o := range outer {
		dstOff := o * dstRowStride
		for _, src := range ctx.Inputs {
			srcDims := src.Shape().Dims()
			srcCount := srcDims[axis]
			srcBuf := src.Storage().CPUBytes()
			srcRowStride := srcCount * innerBytes
			srcOff := o * srcRowStride
			copy(dstBuf[dstOff:dstOff+srcRowStride], srcBuf[srcOff:srcOff+srcRowStride])
			dstOff += srcRowStride
		}
	}
	return nil
}
