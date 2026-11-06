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
	for dt, s := range map[dtype.DType]string{dtype.F32: "f32", dtype.BF16: "bf16", dtype.F16: "f16"} {
		registry.Register(registry.Key{Op: "moe_gemm", Dev: device.KindCUDA, DType: dt}, moeGemmLauncher("moe_gemm_"+s))
	}
}

// moeGemmLauncher: Inputs = [input, weights (E, N, K), sortedPairs I64 (M),
// expertIDs I64 (M), counts I32 (E) scratch, offsets I32 (E+1) scratch,
// topkWeights F32 (M) or nil]; Attrs topk, row_per_pair. See src/moe.cu.
func moeGemmLauncher(name string) registry.Kernel {
	return func(ctx *registry.Context) error {
		in, w, pairs, experts, counts, offsets, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Inputs[3], ctx.Inputs[4], ctx.Inputs[5], ctx.Output
		var topkW uintptr
		if len(ctx.Inputs) > 6 && ctx.Inputs[6] != nil {
			topkW = ctx.Inputs[6].Storage().CUDAPtr()
		}
		topk := int32(ctx.Attrs["topk"].(int))
		rowPerPair := int32(0)
		if ctx.Attrs["row_per_pair"].(bool) {
			rowPerPair = 1
		}
		E, N, K := w.Dim(0), uint64(w.Dim(1)), uint64(w.Dim(2))
		M := uint64(pairs.Dim(0))
		sizeM := uint64(out.Dim(0))
		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		countFn, err := GetKernel("moe_count_experts")
		if err != nil {
			return err
		}
		offFn, err := GetKernel("moe_expert_offsets")
		if err != nil {
			return err
		}
		fn, err := GetKernel(name)
		if err != nil {
			return err
		}
		expPtr, cntPtr, offPtr := experts.Storage().CUDAPtr(), counts.Storage().CUDAPtr(), offsets.Storage().CUDAPtr()
		if err := Launch1D(countFn, M, []unsafe.Pointer{unsafe.Pointer(&expPtr), unsafe.Pointer(&cntPtr), unsafe.Pointer(&M)}); err != nil {
			return err
		}
		e32 := int32(E)
		if err := cudart.LaunchKernel(offFn, 1, 1, 1, 32, 1, 1, 0, 0, []unsafe.Pointer{unsafe.Pointer(&cntPtr), unsafe.Pointer(&offPtr), unsafe.Pointer(&e32)}); err != nil {
			return err
		}
		inPtr, wPtr, pairPtr, outPtr := in.Storage().CUDAPtr(), w.Storage().CUDAPtr(), pairs.Storage().CUDAPtr(), out.Storage().CUDAPtr()
		args := []unsafe.Pointer{
			unsafe.Pointer(&inPtr), unsafe.Pointer(&wPtr), unsafe.Pointer(&pairPtr), unsafe.Pointer(&offPtr),
			unsafe.Pointer(&topkW), unsafe.Pointer(&outPtr), unsafe.Pointer(&topk), unsafe.Pointer(&sizeM),
			unsafe.Pointer(&N), unsafe.Pointer(&K), unsafe.Pointer(&rowPerPair),
		}
		gridN := uint32((N + 31) / 32)
		if err := cudart.LaunchKernel(fn, uint32(E), gridN, 1, 128, 1, 1, 0, 0, args); err != nil {
			return fmt.Errorf("cuda %s: %w", name, err)
		}
		return nil
	}
}

// ---- quantized experts (src/moe_gguf.cu) ----

func init() {
	for dt, kname := range matmulQKernels { // same six GGUF formats as matmul_q
		registry.Register(registry.Key{Op: "moe_gemm_q", Dev: device.KindCUDA, DType: dt}, moeGemmQLauncher("moe_gemm_"+kname[len("matmul_"):]))
	}
}

// moeGemmQLauncher: same Inputs/Attrs contract as moeGemmLauncher with x F32
// and w (E, N, K) block-quantized; output F32 (M, N).
func moeGemmQLauncher(name string) registry.Kernel {
	return func(ctx *registry.Context) error {
		x, w, pairs, experts, counts, offsets, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Inputs[3], ctx.Inputs[4], ctx.Inputs[5], ctx.Output
		var topkW uintptr
		if len(ctx.Inputs) > 6 && ctx.Inputs[6] != nil {
			topkW = ctx.Inputs[6].Storage().CUDAPtr()
		}
		topk := int32(ctx.Attrs["topk"].(int))
		rowPerPair := int32(0)
		if ctx.Attrs["row_per_pair"].(bool) {
			rowPerPair = 1
		}
		E, N, K := w.Dim(0), int32(w.Dim(1)), int32(w.Dim(2))
		M := uint64(pairs.Dim(0))
		sizeM := uint64(out.Dim(0))
		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		countFn, err := GetKernel("moe_count_experts")
		if err != nil {
			return err
		}
		offFn, err := GetKernel("moe_expert_offsets")
		if err != nil {
			return err
		}
		fn, err := GetKernel(name)
		if err != nil {
			return err
		}
		expPtr, cntPtr, offPtr := experts.Storage().CUDAPtr(), counts.Storage().CUDAPtr(), offsets.Storage().CUDAPtr()
		if err := Launch1D(countFn, M, []unsafe.Pointer{unsafe.Pointer(&expPtr), unsafe.Pointer(&cntPtr), unsafe.Pointer(&M)}); err != nil {
			return err
		}
		e32 := int32(E)
		if err := cudart.LaunchKernel(offFn, 1, 1, 1, 32, 1, 1, 0, 0, []unsafe.Pointer{unsafe.Pointer(&cntPtr), unsafe.Pointer(&offPtr), unsafe.Pointer(&e32)}); err != nil {
			return err
		}
		xPtr, wPtr, pairPtr, outPtr := x.Storage().CUDAPtr(), w.Storage().CUDAPtr(), pairs.Storage().CUDAPtr(), out.Storage().CUDAPtr()
		args := []unsafe.Pointer{
			unsafe.Pointer(&xPtr), unsafe.Pointer(&wPtr), unsafe.Pointer(&pairPtr), unsafe.Pointer(&offPtr),
			unsafe.Pointer(&topkW), unsafe.Pointer(&outPtr), unsafe.Pointer(&topk), unsafe.Pointer(&sizeM),
			unsafe.Pointer(&N), unsafe.Pointer(&K), unsafe.Pointer(&rowPerPair),
		}
		const warpsPerBlock = 8
		gridX := uint32((int(N) + warpsPerBlock - 1) / warpsPerBlock)
		gridY := uint32(min((int(M)+3)/4, 1024))
		if gridY == 0 {
			gridY = 1
		}
		if err := cudart.LaunchKernel(fn, gridX, gridY, uint32(E), 32*warpsPerBlock, 1, 1, 0, 0, args); err != nil {
			return fmt.Errorf("cuda %s: %w", name, err)
		}
		return nil
	}
}
