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
	for _, dt := range []dtype.DType{dtype.F32, dtype.BF16} {
		registry.Register(registry.Key{Op: "int4_dequant", Dev: device.KindCUDA, DType: dt}, int4DequantLauncher)
		registry.Register(registry.Key{Op: "int4_gemv", Dev: device.KindCUDA, DType: dt}, int4GEMVLauncher)
	}
}

// int4GEMVWarps / int4GEMVMaxM are INT4_GEMV_WARPS / INT4_GEMV_MAXM in int4.cu.
const (
	int4GEMVWarps = 8
	int4GEMVMaxM  = 8
)

func int4Suffix(dt dtype.DType) (string, error) {
	switch dt {
	case dtype.F32:
		return "f32", nil
	case dtype.BF16:
		return "bf16", nil
	}
	return "", fmt.Errorf("int4: dtype %v not supported (F32, BF16)", dt)
}

// int4DequantLauncher: see ops.DequantizeInt4.
//
//	Inputs: codes U32 (N, K/8), scales F32 (N, K/G), zeros F32 (N, K/G)
//	Output: (N, K) F32 / BF16
//	Attrs:  "group" int
func int4DequantLauncher(ctx *registry.Context) error {
	codes, s, z, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	N, K, G := codes.Dim(0), codes.Dim(1)*8, ctx.Attrs["group"].(int)
	if G%8 != 0 {
		return fmt.Errorf("cuda int4_dequant: group %d not a multiple of 8", G)
	}
	suf, err := int4Suffix(out.DType())
	if err != nil {
		return err
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("int4_dequant_" + suf)
	if err != nil {
		return err
	}
	cP, sP, zP, oP := codes.Storage().CUDAPtr(), s.Storage().CUDAPtr(), z.Storage().CUDAPtr(), out.Storage().CUDAPtr()
	n32, k32, g32 := int32(N), int32(K), int32(G)
	return Launch1D(fn, uint64(N*K/8), []unsafe.Pointer{
		unsafe.Pointer(&cP), unsafe.Pointer(&sP), unsafe.Pointer(&zP), unsafe.Pointer(&oP),
		unsafe.Pointer(&n32), unsafe.Pointer(&k32), unsafe.Pointer(&g32),
	})
}

// int4GEMVLauncher: y (M, N) = x (M, K) · Wᵀ for M ≤ int4GEMVMaxM.
//
//	Inputs: x (M, K) F32 / BF16, codes U32 (N, K/8), scales, zeros F32 (N, K/G)
//	Output: (M, N) at x's dtype
//	Attrs:  "group" int
func int4GEMVLauncher(ctx *registry.Context) error {
	x, codes, s, z, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Inputs[3], ctx.Output
	M, K, N, G := x.Dim(0), x.Dim(1), codes.Dim(0), ctx.Attrs["group"].(int)
	if M > int4GEMVMaxM || K%32 != 0 || G%32 != 0 {
		return fmt.Errorf("cuda int4_gemv: M %d (≤ %d), K %d and group %d (multiples of 32) unsupported", M, int4GEMVMaxM, K, G)
	}
	suf, err := int4Suffix(x.DType())
	if err != nil {
		return err
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("int4_gemv_" + suf)
	if err != nil {
		return err
	}
	xP, cP, sP, zP, oP := x.Storage().CUDAPtr(), codes.Storage().CUDAPtr(), s.Storage().CUDAPtr(), z.Storage().CUDAPtr(), out.Storage().CUDAPtr()
	m32, n32, k32, g32 := int32(M), int32(N), int32(K), int32(G)
	args := []unsafe.Pointer{
		unsafe.Pointer(&xP), unsafe.Pointer(&cP), unsafe.Pointer(&sP), unsafe.Pointer(&zP), unsafe.Pointer(&oP),
		unsafe.Pointer(&m32), unsafe.Pointer(&n32), unsafe.Pointer(&k32), unsafe.Pointer(&g32),
	}
	grid := uint32((N + int4GEMVWarps - 1) / int4GEMVWarps)
	return cudart.LaunchKernel(fn, grid, 1, 1, int4GEMVWarps*32, 1, 1, 0, 0, args)
}
