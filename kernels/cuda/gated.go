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
	registry.Register(registry.Key{Op: "silu_mul_bf16", Dev: device.KindCUDA, DType: dtype.F32}, siluMulBF16CUDA)
}

// siluMulBF16CUDA: Inputs g, u contiguous F32 of one shape, Output BF16.
func siluMulBF16CUDA(ctx *registry.Context) error {
	g, u, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	if g.DType() != dtype.F32 || u.DType() != dtype.F32 || out.DType() != dtype.BF16 {
		return fmt.Errorf("cuda silu_mul_bf16: want F32 g, u and BF16 out, got %v, %v, %v", g.DType(), u.DType(), out.DType())
	}
	n := int64(out.NumElements())
	if n == 0 {
		return nil
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("silu_mul_f32_bf16")
	if err != nil {
		return err
	}
	gp, up, op := g.Storage().CUDAPtr(), u.Storage().CUDAPtr(), out.Storage().CUDAPtr()
	args := []unsafe.Pointer{unsafe.Pointer(&gp), unsafe.Pointer(&up), unsafe.Pointer(&op), unsafe.Pointer(&n)}
	return Launch1D(fn, uint64((n+3)/4), args)
}
