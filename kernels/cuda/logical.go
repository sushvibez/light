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
	registry.Register(registry.Key{Op: "logical_and", Dev: device.KindCUDA, DType: dtype.U8}, logicalBinLauncher("logical_and_u8"))
	registry.Register(registry.Key{Op: "logical_or", Dev: device.KindCUDA, DType: dtype.U8}, logicalBinLauncher("logical_or_u8"))
	registry.Register(registry.Key{Op: "logical_not", Dev: device.KindCUDA, DType: dtype.U8}, logicalNotLauncher)
}

func logicalBinLauncher(kname string) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
		if !a.Shape().Equal(out.Shape()) || !b.Shape().Equal(out.Shape()) {
			return fmt.Errorf("cuda %s: broadcasting not supported (contiguous same-shape only)", kname)
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

func logicalNotLauncher(ctx *registry.Context) error {
	a, out := ctx.Inputs[0], ctx.Output
	if !a.Shape().Equal(out.Shape()) {
		return fmt.Errorf("cuda logical_not: shape mismatch")
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("logical_not_u8")
	if err != nil {
		return err
	}
	aPtr := a.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	n := uint64(out.NumElements())
	args := []unsafe.Pointer{
		unsafe.Pointer(&aPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&n),
	}
	if err := Launch1D(fn, n, args); err != nil {
		return err
	}
	return nil
}
