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
		registry.Register(registry.Key{Op: "fp8_block_dequant", Dev: device.KindCUDA, DType: dt}, fp8BlockDequantLauncher)
		registry.Register(registry.Key{Op: "fp8_block_gemv", Dev: device.KindCUDA, DType: dt}, fp8BlockGEMVLauncher)
	}
}

// fp8GEMVWarps / fp8GEMVMaxM are FP8_GEMV_WARPS / FP8_GEMV_MAXM in
// fp8_block.cu.
const (
	fp8GEMVWarps = 8
	fp8GEMVMaxM  = 8
)

func fp8Suffix(dt dtype.DType) (string, error) {
	switch dt {
	case dtype.F32:
		return "f32", nil
	case dtype.BF16:
		return "bf16", nil
	}
	return "", fmt.Errorf("fp8_block: output dtype %v not supported (F32, BF16)", dt)
}

// fp8BlockDequantLauncher: see ops.DequantizeF8Block.
//
//	Inputs: w (R, C) F8E4M3 contiguous, scale (R/bR, C/bC) F32 contiguous
//	Output: (R, C) F32 or BF16
//	Attrs:  "block_r", "block_c" int
func fp8BlockDequantLauncher(ctx *registry.Context) error {
	w, s, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	R, C := w.Dim(0), w.Dim(1)
	bR, bC := ctx.Attrs["block_r"].(int), ctx.Attrs["block_c"].(int)
	if C%4 != 0 || bC%4 != 0 {
		return fmt.Errorf("cuda fp8_block_dequant: columns %d / block %d not multiples of 4", C, bC)
	}
	suf, err := fp8Suffix(out.DType())
	if err != nil {
		return err
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("fp8_block_dequant_" + suf)
	if err != nil {
		return err
	}
	wP, sP, oP := w.Storage().CUDAPtr(), s.Storage().CUDAPtr(), out.Storage().CUDAPtr()
	r32, c32, br32, bc32 := int32(R), int32(C), int32(bR), int32(bC)
	return Launch1D(fn, uint64(R*C/4), []unsafe.Pointer{
		unsafe.Pointer(&wP), unsafe.Pointer(&sP), unsafe.Pointer(&oP),
		unsafe.Pointer(&r32), unsafe.Pointer(&c32), unsafe.Pointer(&br32), unsafe.Pointer(&bc32),
	})
}

// fp8BlockGEMVLauncher: y (M, N) = x (M, K) · Wᵀ with W block-scaled FP8,
// for M ≤ fp8GEMVMaxM. See ops.MatMulF8Block.
//
//	Inputs: x (M, K) contiguous F32 / BF16, w (N, K) F8E4M3, scale (N/bR, K/bC) F32
//	Output: (M, N) at x's dtype
//	Attrs:  "block_r", "block_c" int
func fp8BlockGEMVLauncher(ctx *registry.Context) error {
	x, w, s, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	M, K, N := x.Dim(0), x.Dim(1), w.Dim(0)
	bR, bC := ctx.Attrs["block_r"].(int), ctx.Attrs["block_c"].(int)
	if M > fp8GEMVMaxM {
		return fmt.Errorf("cuda fp8_block_gemv: %d rows > %d", M, fp8GEMVMaxM)
	}
	if K%16 != 0 || bC%16 != 0 {
		return fmt.Errorf("cuda fp8_block_gemv: K %d / block %d not multiples of 16", K, bC)
	}
	suf, err := fp8Suffix(x.DType())
	if err != nil {
		return err
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("fp8_block_gemv_" + suf)
	if err != nil {
		return err
	}
	xP, wP, sP, oP := x.Storage().CUDAPtr(), w.Storage().CUDAPtr(), s.Storage().CUDAPtr(), out.Storage().CUDAPtr()
	m32, n32, k32, br32, bc32 := int32(M), int32(N), int32(K), int32(bR), int32(bC)
	args := []unsafe.Pointer{
		unsafe.Pointer(&xP), unsafe.Pointer(&wP), unsafe.Pointer(&sP), unsafe.Pointer(&oP),
		unsafe.Pointer(&m32), unsafe.Pointer(&n32), unsafe.Pointer(&k32), unsafe.Pointer(&br32), unsafe.Pointer(&bc32),
	}
	grid := uint32((N + fp8GEMVWarps - 1) / fp8GEMVWarps)
	return cudart.LaunchKernel(fn, grid, 1, 1, fp8GEMVWarps*32, 1, 1, 0, 0, args)
}
