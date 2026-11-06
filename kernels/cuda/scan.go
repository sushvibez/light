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
	registry.Register(registry.Key{Op: "cumsum", Dev: device.KindCUDA, DType: dtype.F32}, scanLauncher("cumsum_f32"))
	registry.Register(registry.Key{Op: "cumprod", Dev: device.KindCUDA, DType: dtype.F32}, scanLauncher("cumprod_f32"))
}

func scanLauncher(kname string) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, out := ctx.Inputs[0], ctx.Output
		dim := ctx.Attrs["dim"].(int)
		dims := a.Shape().Dims()
		outer := 1
		for i := 0; i < dim; i++ {
			outer *= dims[i]
		}
		inner := 1
		for i := dim + 1; i < len(dims); i++ {
			inner *= dims[i]
		}
		axSize := uint64(dims[dim])
		outerU := uint64(outer)
		innerU := uint64(inner)

		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(kname)
		if err != nil {
			return err
		}
		aPtr := a.Storage().CUDAPtr()
		outPtr := out.Storage().CUDAPtr()
		args := []unsafe.Pointer{
			unsafe.Pointer(&aPtr),
			unsafe.Pointer(&outPtr),
			unsafe.Pointer(&outerU),
			unsafe.Pointer(&axSize),
			unsafe.Pointer(&innerU),
		}
		n := outerU * innerU
		if err := Launch1D(fn, n, args); err != nil {
			return err
		}
		return nil
	}
}

// Ensure fmt is used somewhere (reserved for future error messages).
var _ = fmt.Errorf
