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
	registry.Register(registry.Key{Op: "geglu", Dev: device.KindCUDA, DType: dtype.F32}, gegluLauncher("geglu_fwd_f32"))
	registry.Register(registry.Key{Op: "geglu", Dev: device.KindCUDA, DType: dtype.BF16}, gegluLauncher("geglu_fwd_bf16"))
}

// gegluLauncher invokes the fused GeGLU forward kernel. Input shape is
// (..., 2*C); output is (..., C). The activation collapses HF SD's
// FeedForward.net.0 (Linear → split → hidden·GELU(gate)) from a 16-launch
// chain (Linear + 2 Narrows + 2 Contiguous + 10-op composed GELU + Mul)
// into one device kernel, which both removes the dtype-mismatch traps in
// the composed path and saves the two `(B, T, inner)` scratch tensors.
func gegluLauncher(kname string) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, out := ctx.Inputs[0], ctx.Output
		if a.Rank() < 2 {
			return fmt.Errorf("cuda %s: input rank must be ≥ 2, got %d", kname, a.Rank())
		}
		twoC := a.Dim(-1)
		if twoC%2 != 0 {
			return fmt.Errorf("cuda %s: last dim %d must be even", kname, twoC)
		}
		C := twoC / 2
		if out.Dim(-1) != C {
			return fmt.Errorf("cuda %s: output last dim %d != %d", kname, out.Dim(-1), C)
		}
		rows := uint64(a.NumElements()) / uint64(twoC)
		cVal := uint64(C)
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
			unsafe.Pointer(&rows),
			unsafe.Pointer(&cVal),
		}
		if err := Launch1D(fn, rows*cVal, args); err != nil {
			return err
		}
		return nil
	}
}
