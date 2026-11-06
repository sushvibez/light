//go:build !nocuda && cgo

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
	for _, op := range []string{"lt", "le", "gt", "ge", "eq", "ne"} {
		registry.Register(registry.Key{Op: op, Dev: device.KindCUDA, DType: dtype.F32}, compareLauncher(op+"_f32"))
	}
	registry.Register(registry.Key{Op: "where", Dev: device.KindCUDA, DType: dtype.F32}, whereLauncher("where_f32"))
	registry.Register(registry.Key{Op: "where", Dev: device.KindCUDA, DType: dtype.F64}, whereLauncher("where_f64"))
	// F64 / I32 / I64 / U8 / BF16 (2026-10-28): elementwise.cu + bf16_ops.cu.
	suffix := map[dtype.DType]string{dtype.F64: "f64", dtype.I32: "i32", dtype.I64: "i64", dtype.U8: "u8", dtype.BF16: "bf16"}
	for _, op := range []string{"lt", "le", "gt", "ge", "eq", "ne"} {
		for _, dt := range []dtype.DType{dtype.F64, dtype.I32, dtype.I64, dtype.U8, dtype.BF16} {
			registry.Register(registry.Key{Op: op, Dev: device.KindCUDA, DType: dt}, compareLauncher(op+"_"+suffix[dt]))
		}
	}
	for _, dt := range []dtype.DType{dtype.I32, dtype.I64, dtype.U8, dtype.BF16} {
		registry.Register(registry.Key{Op: "where", Dev: device.KindCUDA, DType: dt}, whereLauncher("where_"+suffix[dt]))
	}
}

// compareLauncher wraps a compare_<op>_f32 kernel. Inputs a and b must be
// contiguous + same shape; output is U8, same shape.
func compareLauncher(kname string) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
		if !a.Shape().Equal(out.Shape()) || !b.Shape().Equal(out.Shape()) {
			return fmt.Errorf("cuda %s: broadcast not yet supported on CUDA (contiguous same-shape only)", kname)
		}
		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(kname)
		if err != nil {
			return err
		}
		aPtr := a.Storage().CUDAPtr()
		bPtr := b.Storage().CUDAPtr()
		outPtr := out.Storage().CUDAPtr()
		n := uint64(out.NumElements())
		args := []unsafe.Pointer{
			unsafe.Pointer(&aPtr),
			unsafe.Pointer(&bPtr),
			unsafe.Pointer(&outPtr),
			unsafe.Pointer(&n),
		}
		if err := Launch1D(fn, n, args); err != nil {
			return err
		}
		return nil
	}
}

func whereLauncher(kname string) registry.Kernel {
	return func(ctx *registry.Context) error {
		cond, a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
		if !cond.Shape().Equal(out.Shape()) || !a.Shape().Equal(out.Shape()) || !b.Shape().Equal(out.Shape()) {
			return fmt.Errorf("cuda %s: broadcast not yet supported (contiguous same-shape only)", kname)
		}
		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(kname)
		if err != nil {
			return err
		}
		cPtr := cond.Storage().CUDAPtr()
		aPtr := a.Storage().CUDAPtr()
		bPtr := b.Storage().CUDAPtr()
		outPtr := out.Storage().CUDAPtr()
		n := uint64(out.NumElements())
		args := []unsafe.Pointer{
			unsafe.Pointer(&cPtr),
			unsafe.Pointer(&aPtr),
			unsafe.Pointer(&bPtr),
			unsafe.Pointer(&outPtr),
			unsafe.Pointer(&n),
		}
		if err := Launch1D(fn, n, args); err != nil {
			return err
		}
		return nil
	}
}
