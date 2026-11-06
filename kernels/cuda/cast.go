//go:build !nocuda

package cuda

import (
	"fmt"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
)

func init() {
	// Register Cast under the INPUT dtype; output dtype is in ctx.Output.
	for _, dt := range []dtype.DType{dtype.F32, dtype.F64, dtype.F16, dtype.BF16, dtype.I8, dtype.U8, dtype.I16, dtype.U16, dtype.I32, dtype.U32, dtype.I64, dtype.U64, dtype.F8E4M3} {
		registry.Register(registry.Key{Op: "cast", Dev: device.KindCUDA, DType: dt}, castLauncher)
	}
}

var castKernelTable = map[struct{ from, to dtype.DType }]string{
	{dtype.F32, dtype.F16}:     "cast_f32_to_f16",
	{dtype.F16, dtype.F32}:     "cast_f16_to_f32",
	{dtype.F32, dtype.BF16}:    "cast_f32_to_bf16",
	{dtype.BF16, dtype.F32}:    "cast_bf16_to_f32",
	{dtype.F32, dtype.F64}:     "cast_f32_to_f64",
	{dtype.F64, dtype.F32}:     "cast_f64_to_f32",
	{dtype.F32, dtype.I32}:     "cast_f32_to_i32",
	{dtype.I32, dtype.F32}:     "cast_i32_to_f32",
	{dtype.F32, dtype.I64}:     "cast_f32_to_i64",
	{dtype.I64, dtype.F32}:     "cast_i64_to_f32",
	{dtype.F32, dtype.F8E4M3}:  "cast_f32_to_f8e4m3",
	{dtype.F8E4M3, dtype.F32}:  "cast_f8e4m3_to_f32",
	{dtype.BF16, dtype.F8E4M3}: "cast_bf16_to_f8e4m3",
	{dtype.F8E4M3, dtype.BF16}: "cast_f8e4m3_to_bf16",
	{dtype.F16, dtype.F8E4M3}:  "cast_f16_to_f8e4m3",
	{dtype.F8E4M3, dtype.F16}:  "cast_f8e4m3_to_f16",
	{dtype.I8, dtype.F32}:      "cast_i8_to_f32",
	{dtype.F32, dtype.I8}:      "cast_f32_to_i8",
	{dtype.U8, dtype.F32}:      "cast_u8_to_f32",
	{dtype.F32, dtype.U8}:      "cast_f32_to_u8",
	{dtype.I16, dtype.F32}:     "cast_i16_to_f32",
	{dtype.F32, dtype.I16}:     "cast_f32_to_i16",
	{dtype.U16, dtype.F32}:     "cast_u16_to_f32",
	{dtype.F32, dtype.U16}:     "cast_f32_to_u16",
	{dtype.U32, dtype.F32}:     "cast_u32_to_f32",
	{dtype.F32, dtype.U32}:     "cast_f32_to_u32",
	{dtype.U64, dtype.F32}:     "cast_u64_to_f32",
	{dtype.F32, dtype.U64}:     "cast_f32_to_u64",
}

func castLauncher(ctx *registry.Context) error {
	a, out := ctx.Inputs[0], ctx.Output
	from, to := a.DType(), out.DType()
	if from == to {
		// Same-dtype cast: just memcpy.
		bytes := uintptr(a.NumElements()) * uintptr(from.Size())
		return cudart.MemcpyDtoD(out.Storage().CUDAPtr(), a.Storage().CUDAPtr(), bytes)
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	n := uint64(a.NumElements())
	aPtr := a.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	if _, ok := castKernelTable[struct{ from, to dtype.DType }{from, to}]; ok {
		if err := launchCast(from, to, aPtr, outPtr, n); err != nil {
			return err
		}
		return nil
	}
	// Any other pair hops through an F32 scratch buffer (every dtype has a
	// kernel to and from F32).
	if n == 0 {
		return nil
	}
	tmp, err := cudart.Malloc(uintptr(n) * 4)
	if err != nil {
		return err
	}
	defer func() { _ = cudart.Free(tmp) }()
	if err := launchCast(from, dtype.F32, aPtr, tmp, n); err != nil {
		return err
	}
	if err := launchCast(dtype.F32, to, tmp, outPtr, n); err != nil {
		return err
	}
	return nil
}

func launchCast(from, to dtype.DType, src, dst uintptr, n uint64) error {
	kname, ok := castKernelTable[struct{ from, to dtype.DType }{from, to}]
	if !ok {
		return fmt.Errorf("cuda cast: unsupported pair %v → %v", from, to)
	}
	fn, err := GetKernel(kname)
	if err != nil {
		return err
	}
	args := []unsafe.Pointer{unsafe.Pointer(&src), unsafe.Pointer(&dst), unsafe.Pointer(&n)}
	return Launch1D(fn, n, args)
}
