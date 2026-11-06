//go:build !nocuda && cgo

package cuda

import (
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "matmul_int8", Dev: device.KindCUDA, DType: dtype.F32}, matmulInt8F32CUDA)
}

// matmulInt8F32CUDA launches the per-channel INT8 weight matmul on a 16x16
// thread grid over (N, M).
//
//	Inputs[0]: x F32 (M, K)   Inputs[1]: wq I8 (K, N)   Inputs[2]: scales F32 (N,)
//	Output:    F32 (M, N)
func matmulInt8F32CUDA(ctx *registry.Context) error {
	x, wq, scales, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	M, K := int32(x.Dim(0)), int32(x.Dim(1))
	N := int32(wq.Dim(1))
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("matmul_int8_perchan_f32")
	if err != nil {
		return err
	}
	xPtr, wPtr, sPtr, oPtr := x.Storage().CUDAPtr(), wq.Storage().CUDAPtr(), scales.Storage().CUDAPtr(), out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&xPtr), unsafe.Pointer(&wPtr), unsafe.Pointer(&sPtr), unsafe.Pointer(&oPtr),
		unsafe.Pointer(&M), unsafe.Pointer(&K), unsafe.Pointer(&N),
	}
	const tile = 16
	gridX := uint32((int(N) + tile - 1) / tile)
	gridY := uint32((int(M) + tile - 1) / tile)
	if gridX == 0 {
		gridX = 1
	}
	if gridY == 0 {
		gridY = 1
	}
	if err := cudart.LaunchKernel(fn, gridX, gridY, 1, tile, tile, 1, 0, 0, args); err != nil {
		return err
	}
	return nil
}
