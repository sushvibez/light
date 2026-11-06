//go:build !nocuda && cgo

package cuda

import (
	"fmt"
	"unsafe"

	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
	"light/shape"
)

// stridedBinaryName maps a plain op name to its strided-kernel variant,
// per dtype.
var stridedBinaryName = map[struct {
	op string
	dt dtype.DType
}]string{
	{"add", dtype.F32}: "add_strided_f32", {"add", dtype.F64}: "add_strided_f64", {"add", dtype.BF16}: "add_strided_bf16",
	{"sub", dtype.F32}: "sub_strided_f32", {"sub", dtype.F64}: "sub_strided_f64", {"sub", dtype.BF16}: "sub_strided_bf16",
	{"mul", dtype.F32}: "mul_strided_f32", {"mul", dtype.F64}: "mul_strided_f64", {"mul", dtype.BF16}: "mul_strided_bf16",
	{"div", dtype.F32}: "div_strided_f32", {"div", dtype.F64}: "div_strided_f64", {"div", dtype.BF16}: "div_strided_bf16",
	{"maximum", dtype.F32}: "maximum_strided_f32", {"maximum", dtype.F64}: "maximum_strided_f64", {"maximum", dtype.BF16}: "maximum_strided_bf16",
	{"maximum", dtype.I32}: "maximum_strided_i32", {"maximum", dtype.I64}: "maximum_strided_i64",
	{"minimum", dtype.F32}: "minimum_strided_f32", {"minimum", dtype.F64}: "minimum_strided_f64", {"minimum", dtype.BF16}: "minimum_strided_bf16",
	{"minimum", dtype.I32}: "minimum_strided_i32", {"minimum", dtype.I64}: "minimum_strided_i64",
	{"add", dtype.F16}: "add_strided_f16", {"sub", dtype.F16}: "sub_strided_f16", {"mul", dtype.F16}: "mul_strided_f16",
	{"div", dtype.F16}: "div_strided_f16", {"maximum", dtype.F16}: "maximum_strided_f16", {"minimum", dtype.F16}: "minimum_strided_f16",
}

// launchStridedBinary executes a broadcast-aware binary kernel. `name` is
// the plain op+dtype identifier (e.g., "add_f32"); we map it to the strided
// variant internally.
func launchStridedBinary(name string, ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	op, dtSuffix := splitNameDType(name)
	stridedName, ok := stridedBinaryName[struct {
		op string
		dt dtype.DType
	}{op, parseDType(dtSuffix)}]
	if !ok {
		return fmt.Errorf("cuda %s: broadcast/strided not supported for this dtype yet", name)
	}

	// Compute broadcast-adjusted strides for a and b against out.Shape.
	aStr, err := shape.BroadcastStrides(a.Shape(), a.Strides(), out.Shape())
	if err != nil {
		return err
	}
	bStr, err := shape.BroadcastStrides(b.Shape(), b.Strides(), out.Shape())
	if err != nil {
		return err
	}

	dims := out.Shape().Dims()
	rank := int32(len(dims))
	// Pack meta [dims…, aStr…, bStr…] as int32.
	meta := make([]int32, 3*len(dims))
	for i, d := range dims {
		meta[i] = int32(d)
		meta[len(dims)+i] = int32(aStr.Get(i))
		meta[2*len(dims)+i] = int32(bStr.Get(i))
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	metaDev, err := uploadMeta(out.Device().Ordinal(), meta)
	if err != nil {
		return err
	}

	fn, err := GetKernel(stridedName)
	if err != nil {
		return err
	}

	elemSize := a.DType().Size()
	aPtr := a.Storage().CUDAPtr() + uintptr(a.Offset()*elemSize)
	bPtr := b.Storage().CUDAPtr() + uintptr(b.Offset()*elemSize)
	outPtr := out.Storage().CUDAPtr()
	n := uint64(out.NumElements())

	args := []unsafe.Pointer{
		unsafe.Pointer(&aPtr),
		unsafe.Pointer(&bPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&n),
		unsafe.Pointer(&rank),
		unsafe.Pointer(&metaDev),
	}
	if err := Launch1D(fn, n, args); err != nil {
		return err
	}
	return nil
}

// splitNameDType splits "add_f32" → ("add", "f32"). Works for any op+dtype
// with the "name_dtype" convention used in our .cu source.
func splitNameDType(kernelName string) (string, string) {
	for i := len(kernelName) - 1; i > 0; i-- {
		if kernelName[i] == '_' {
			return kernelName[:i], kernelName[i+1:]
		}
	}
	return kernelName, ""
}

// parseDType converts "f32"/"f64"/etc. → dtype.DType.
func parseDType(s string) dtype.DType {
	switch s {
	case "f32":
		return dtype.F32
	case "f64":
		return dtype.F64
	case "f16":
		return dtype.F16
	case "bf16":
		return dtype.BF16
	case "i32":
		return dtype.I32
	case "i64":
		return dtype.I64
	case "u8":
		return dtype.U8
	case "i8":
		return dtype.I8
	}
	return dtype.F32
}
