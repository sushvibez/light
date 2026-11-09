//go:build !nocuda && cgo

package cuda

import (
	"fmt"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
	"light/tensor"
)

func init() {
	registry.Register(registry.Key{Op: "matmul_w8a8", Dev: device.KindCUDA, DType: dtype.F32}, matmulW8A8CUDA)
}

// matmulW8A8CUDA: out (M, N) F32 = x (M, K) F32 × wqᵀ with wq (N, K) int8
// and per-output-channel scales ws (N,) F32. The activation is quantised
// per row to int8 on the device, the product runs as an int8 × int8 → int32
// cuBLAS GEMM (tensor-core IMMA), and the epilogue applies the row and
// column scales. Three launches, K and N multiples of 4.
//
//	Inputs[0]: x F32 (M, K)   Inputs[1]: wq I8 (N, K)   Inputs[2]: ws F32 (N,)
func matmulW8A8CUDA(ctx *registry.Context) error {
	x, wq, ws, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	M, K, N := x.Dim(0), x.Dim(1), wq.Dim(0)
	if K%4 != 0 || N%4 != 0 {
		return fmt.Errorf("cuda matmul_w8a8: K (%d) and N (%d) must be multiples of 4", K, N)
	}
	dev := out.Device()
	if err := cudart.SetDevice(dev.Ordinal()); err != nil {
		return err
	}
	xq, err := tensor.Empty(dtype.I8, []int{M, K}, dev)
	if err != nil {
		return err
	}
	xs, err := tensor.Empty(dtype.F32, []int{M}, dev)
	if err != nil {
		return err
	}
	acc, err := tensor.Empty(dtype.I32, []int{M, N}, dev)
	if err != nil {
		return err
	}
	qfn, err := GetKernel("quantize_rows_i8_f32")
	if err != nil {
		return err
	}
	xPtr, xqPtr, xsPtr := x.Storage().CUDAPtr(), xq.Storage().CUDAPtr(), xs.Storage().CUDAPtr()
	m32, k32 := int32(M), int32(K)
	qargs := []unsafe.Pointer{unsafe.Pointer(&xPtr), unsafe.Pointer(&xqPtr), unsafe.Pointer(&xsPtr), unsafe.Pointer(&m32), unsafe.Pointer(&k32)}
	if err := cudart.LaunchKernel(qfn, uint32(M), 1, 1, 256, 1, 1, 0, 0, qargs); err != nil {
		return err
	}
	h, err := cudart.GetBlasHandle(dev.Ordinal())
	if err != nil {
		return err
	}
	if err := cudart.GemmInt8RowMajorTransB(h, M, N, K, xqPtr, wq.Storage().CUDAPtr(), acc.Storage().CUDAPtr()); err != nil {
		return err
	}
	dfn, err := GetKernel("dequant_i32_rowcol_f32")
	if err != nil {
		return err
	}
	accPtr, wsPtr, oPtr := acc.Storage().CUDAPtr(), ws.Storage().CUDAPtr(), out.Storage().CUDAPtr()
	mm, nn := int64(M), int64(N)
	dargs := []unsafe.Pointer{unsafe.Pointer(&accPtr), unsafe.Pointer(&xsPtr), unsafe.Pointer(&wsPtr), unsafe.Pointer(&oPtr), unsafe.Pointer(&mm), unsafe.Pointer(&nn)}
	total := uint64(M) * uint64(N)
	return cudart.LaunchKernel(dfn, uint32((total+255)/256), 1, 1, 256, 1, 1, 0, 0, dargs)
}
