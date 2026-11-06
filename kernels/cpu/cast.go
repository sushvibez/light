package cpu

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "cast", Dev: device.KindCPU, DType: dtype.F32}, castKernel)
	registry.Register(registry.Key{Op: "cast", Dev: device.KindCPU, DType: dtype.F64}, castKernel)
	registry.Register(registry.Key{Op: "cast", Dev: device.KindCPU, DType: dtype.F16}, castKernel)
	registry.Register(registry.Key{Op: "cast", Dev: device.KindCPU, DType: dtype.BF16}, castKernel)
	registry.Register(registry.Key{Op: "cast", Dev: device.KindCPU, DType: dtype.I8}, castKernel)
	registry.Register(registry.Key{Op: "cast", Dev: device.KindCPU, DType: dtype.U8}, castKernel)
	registry.Register(registry.Key{Op: "cast", Dev: device.KindCPU, DType: dtype.I16}, castKernel)
	registry.Register(registry.Key{Op: "cast", Dev: device.KindCPU, DType: dtype.U16}, castKernel)
	registry.Register(registry.Key{Op: "cast", Dev: device.KindCPU, DType: dtype.I32}, castKernel)
	registry.Register(registry.Key{Op: "cast", Dev: device.KindCPU, DType: dtype.U32}, castKernel)
	registry.Register(registry.Key{Op: "cast", Dev: device.KindCPU, DType: dtype.I64}, castKernel)
	registry.Register(registry.Key{Op: "cast", Dev: device.KindCPU, DType: dtype.U64}, castKernel)
	registry.Register(registry.Key{Op: "cast", Dev: device.KindCPU, DType: dtype.F8E4M3}, castKernel)
}

// castKernel reads each element of the input as a float64 (or int64 for
// integer → integer casts), then writes it in the output's dtype. Registered
// by input dtype; the output dtype is derived from ctx.Output.DType().
//
// Lossy conversions (e.g. I64 → F64 for values > 2^53) are by design — the
// caller is trusted to know their dtype choices. Matches PyTorch / Candle.
func castKernel(ctx *registry.Context) error {
	a, out := ctx.Inputs[0], ctx.Output
	from, to := a.DType(), out.DType()
	aBuf := a.Storage().CPUBytes()
	outBuf := out.Storage().CPUBytes()
	if from == to {
		copy(outBuf, aBuf)
		return nil
	}
	n := a.NumElements()
	for i := range n {
		v := readF64Buf(from, aBuf, i)
		if err := writeFromF64Buf(to, outBuf, i, v); err != nil {
			return fmt.Errorf("cast %v→%v at %d: %w", from, to, i, err)
		}
	}
	return nil
}

// readF64Buf reads the i-th element of the dtype-typed buffer as a float64.
// Quantized dtypes are not supported.
func readF64Buf(dt dtype.DType, buf []byte, i int) float64 {
	switch dt {
	case dtype.F32:
		return float64(asSliceBytes[float32](buf)[i])
	case dtype.F64:
		return asSliceBytes[float64](buf)[i]
	case dtype.F16:
		return float64(dtype.Half(asSliceBytes[uint16](buf)[i]).ToF32())
	case dtype.BF16:
		return float64(dtype.BFloat16(asSliceBytes[uint16](buf)[i]).ToF32())
	case dtype.F8E4M3:
		return float64(dtype.Float8(buf[i]).ToF32())
	case dtype.I8:
		return float64(asSliceBytes[int8](buf)[i])
	case dtype.U8:
		return float64(asSliceBytes[uint8](buf)[i])
	case dtype.I16:
		return float64(asSliceBytes[int16](buf)[i])
	case dtype.U16:
		return float64(asSliceBytes[uint16](buf)[i])
	case dtype.I32:
		return float64(asSliceBytes[int32](buf)[i])
	case dtype.U32:
		return float64(asSliceBytes[uint32](buf)[i])
	case dtype.I64:
		return float64(asSliceBytes[int64](buf)[i])
	case dtype.U64:
		return float64(asSliceBytes[uint64](buf)[i])
	}
	return 0
}

func writeFromF64Buf(dt dtype.DType, buf []byte, i int, v float64) error {
	switch dt {
	case dtype.F32:
		asSliceBytes[float32](buf)[i] = float32(v)
	case dtype.F64:
		asSliceBytes[float64](buf)[i] = v
	case dtype.F16:
		asSliceBytes[uint16](buf)[i] = uint16(dtype.HalfFromF32(float32(v)))
	case dtype.BF16:
		asSliceBytes[uint16](buf)[i] = uint16(dtype.BF16FromF32(float32(v)))
	case dtype.F8E4M3:
		buf[i] = uint8(dtype.F8FromF32(float32(v)))
	case dtype.I8:
		asSliceBytes[int8](buf)[i] = int8(v)
	case dtype.U8:
		asSliceBytes[uint8](buf)[i] = uint8(v)
	case dtype.I16:
		asSliceBytes[int16](buf)[i] = int16(v)
	case dtype.U16:
		asSliceBytes[uint16](buf)[i] = uint16(v)
	case dtype.I32:
		asSliceBytes[int32](buf)[i] = int32(v)
	case dtype.U32:
		asSliceBytes[uint32](buf)[i] = uint32(v)
	case dtype.I64:
		asSliceBytes[int64](buf)[i] = int64(v)
	case dtype.U64:
		asSliceBytes[uint64](buf)[i] = uint64(v)
	default:
		return fmt.Errorf("unsupported write dtype %v", dt)
	}
	return nil
}

// asSliceBytes reinterprets a byte slice as []T. Used in the cast kernel
// where we cannot easily instantiate asSlice through tensor.Tensor (we have
// raw bytes).
func asSliceBytes[T any](buf []byte) []T {
	if len(buf) == 0 {
		return nil
	}
	var zero T
	sz := int(unsafeSizeof(zero))
	return unsafeSliceFromBytes[T](buf, len(buf)/sz)
}
