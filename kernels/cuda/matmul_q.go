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

// matmulQKernels lists the quant formats with a fused dequant-matmul kernel.
// Formats missing here fall back in ops.MatMulQ to dequant + cuBLAS.
var matmulQKernels = map[dtype.DType]string{
	dtype.Q4_0: "matmul_q4_0_f32",
	dtype.Q5_0: "matmul_q5_0_f32",
	dtype.Q8_0: "matmul_q8_0_f32",
	dtype.Q4_K: "matmul_q4_k_f32",
	dtype.Q5_K: "matmul_q5_k_f32",
	dtype.Q6_K: "matmul_q6_k_f32",
}

// matmulQMVKernels are the single-row (decode) specialisations, used when
// M == 1 and K % 64 == 0 (rows 4-byte aligned).
var matmulQMVKernels = map[dtype.DType]string{
	dtype.Q8_0: "matmul_q8_0_f32_mv",
	dtype.Q4_K: "matmul_q4_k_f32_mv",
	dtype.Q5_K: "matmul_q5_k_f32_mv",
	dtype.Q6_K: "matmul_q6_k_f32_mv",
}

func init() {
	for dt, kname := range matmulQKernels {
		registry.Register(registry.Key{Op: "matmul_q", Dev: device.KindCUDA, DType: dt}, matmulQLauncher(kname))
	}
}

// matmulQLauncher runs out (M,N) = x (M,K) F32 × wᵀ with w (N,K) quantized.
// Grid: one warp per output column n (8 warps per 256-thread block), one
// block row per 4-row tile of M (MT in matmul_q.cu).
func matmulQLauncher(kname string) registry.Kernel {
	return func(ctx *registry.Context) error {
		x, w, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
		M, K := int32(x.Dim(0)), int32(x.Dim(1))
		N := int32(w.Dim(0))
		if w.Dim(1) != int(K) {
			return fmt.Errorf("matmul_q: K mismatch (x %d, w %d)", K, w.Dim(1))
		}
		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(kname)
		if err != nil {
			return err
		}
		xPtr, wPtr, oPtr := x.Storage().CUDAPtr(), w.Storage().CUDAPtr(), out.Storage().CUDAPtr()
		if mv, ok := matmulQMVKernels[w.DType()]; ok && M == 1 && K%64 == 0 && x.Offset() == 0 && x.IsContiguous() {
			fn, err := GetKernel(mv)
			if err != nil {
				return err
			}
			args := []unsafe.Pointer{unsafe.Pointer(&xPtr), unsafe.Pointer(&wPtr), unsafe.Pointer(&oPtr), unsafe.Pointer(&N), unsafe.Pointer(&K)}
			const warpsPerBlock = 8
			grid := uint32((int(N) + warpsPerBlock - 1) / warpsPerBlock)
			if err := cudart.LaunchKernel(fn, grid, 1, 1, 32*warpsPerBlock, 1, 1, 0, 0, args); err != nil {
				return err
			}
			return nil
		}
		args := []unsafe.Pointer{
			unsafe.Pointer(&xPtr), unsafe.Pointer(&wPtr), unsafe.Pointer(&oPtr),
			unsafe.Pointer(&M), unsafe.Pointer(&N), unsafe.Pointer(&K),
		}
		const warpsPerBlock = 8
		gridX := uint32((int(N) + warpsPerBlock - 1) / warpsPerBlock)
		gridY := uint32((int(M) + 3) / 4)
		if gridX == 0 {
			gridX = 1
		}
		if gridY == 0 {
			gridY = 1
		}
		if err := cudart.LaunchKernel(fn, gridX, gridY, 1, 32*warpsPerBlock, 1, 1, 0, 0, args); err != nil {
			return err
		}
		return nil
	}
}
