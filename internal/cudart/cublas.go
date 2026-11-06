//go:build !nocuda

package cudart

/*
#cgo LDFLAGS: -lcublas

#include <cublas_v2.h>
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// BlasHandle wraps a cublasHandle_t. Handles are per-device; use GetBlasHandle.
type BlasHandle struct {
	handle C.cublasHandle_t
}

var blasHandleMu sync.Mutex
var blasHandleByOrd = map[int]*BlasHandle{}

// GetBlasHandle returns a cached cuBLAS handle for the given device ordinal.
// Creates one on first use; the handle lives for the process lifetime.
func GetBlasHandle(ordinal int) (*BlasHandle, error) {
	blasHandleMu.Lock()
	defer blasHandleMu.Unlock()
	if bh, ok := blasHandleByOrd[ordinal]; ok {
		return bh, nil
	}
	if err := SetDevice(ordinal); err != nil {
		return nil, err
	}
	var h C.cublasHandle_t
	if rc := C.cublasCreate(&h); rc != C.CUBLAS_STATUS_SUCCESS {
		return nil, fmt.Errorf("cublasCreate: status %d", int(rc))
	}
	bh := &BlasHandle{handle: h}
	blasHandleByOrd[ordinal] = bh
	return bh, nil
}

// SgemmRowMajor computes C = A @ B in row-major semantics.
//
//	A: (M, K), B: (K, N), C: (M, N) — all row-major contiguous on device.
//
// cuBLAS is column-major, so we swap the call order: asking cuBLAS to compute
// the column-major gemm (B_col)·(A_col) yields the transpose we want.
func SgemmRowMajor(h *BlasHandle, m, n, k int, a, b, c uintptr) error {
	alpha := C.float(1.0)
	beta := C.float(0.0)
	rc := C.cublasSgemm(
		h.handle,
		C.CUBLAS_OP_N, C.CUBLAS_OP_N,
		C.int(n), C.int(m), C.int(k),
		&alpha,
		(*C.float)(unsafe.Pointer(b)), C.int(n),
		(*C.float)(unsafe.Pointer(a)), C.int(k),
		&beta,
		(*C.float)(unsafe.Pointer(c)), C.int(n),
	)
	if rc != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasSgemm: status %d", int(rc))
	}
	return nil
}

// SgemmRowMajorTransB computes C = A @ B^T in row-major semantics, without
// allocating an explicit transpose of B. This is the workhorse for nn.Linear,
// where B is the weight matrix W of shape (out, in) and we want y = x @ W^T.
//
//	A: (M, K), B: (N, K), C: (M, N) — all row-major contiguous on device.
//
// In column-major terms M_A=A^T (K,M), M_B=B^T (K,N), M_C=C^T (N,M); the
// row-major C = A @ B^T corresponds to M_C = M_B^T @ M_A.
func SgemmRowMajorTransB(h *BlasHandle, m, n, k int, a, b, c uintptr) error {
	alpha := C.float(1.0)
	beta := C.float(0.0)
	rc := C.cublasSgemm(
		h.handle,
		C.CUBLAS_OP_T, C.CUBLAS_OP_N,
		C.int(n), C.int(m), C.int(k),
		&alpha,
		(*C.float)(unsafe.Pointer(b)), C.int(k),
		(*C.float)(unsafe.Pointer(a)), C.int(k),
		&beta,
		(*C.float)(unsafe.Pointer(c)), C.int(n),
	)
	if rc != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasSgemm(T,N): status %d", int(rc))
	}
	return nil
}

// GemmExBF16RowMajorTransB computes C = A @ B^T in row-major semantics with
// BF16 inputs and F32 accumulation/output via cublasGemmEx. Used for
// mixed-precision inference: the activation `a` is cast to BF16 by the
// caller, the model weight `b` stays in BF16 storage, and the result lands
// in F32 — so downstream norms / softmax / etc. still run in F32.
//
//	A: (M, K), B: (N, K), C: (M, N) — all row-major contiguous on device.
//	Atype=Btype=BF16, Ctype=F32, computeType=F32.
func GemmExBF16RowMajorTransB(h *BlasHandle, m, n, k int, a, b, c uintptr) error {
	alpha := C.float(1.0)
	beta := C.float(0.0)
	rc := C.cublasGemmEx(
		h.handle,
		C.CUBLAS_OP_T, C.CUBLAS_OP_N,
		C.int(n), C.int(m), C.int(k),
		unsafe.Pointer(&alpha),
		unsafe.Pointer(b), C.CUDA_R_16BF, C.int(k),
		unsafe.Pointer(a), C.CUDA_R_16BF, C.int(k),
		unsafe.Pointer(&beta),
		unsafe.Pointer(c), C.CUDA_R_32F, C.int(n),
		C.CUBLAS_COMPUTE_32F,
		C.CUBLAS_GEMM_DEFAULT,
	)
	if rc != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasGemmEx(BF16,BF16→F32,T,N): status %d", int(rc))
	}
	return nil
}

// GemmExBF16OutBF16RowMajorTransB is the same shape contract as
// GemmExBF16RowMajorTransB but the output also lands in BF16. F32 accumulate
// is still used internally so precision matches cuBLAS's tensor-core path —
// only the output store converts back to BF16. This is the "fully BF16"
// inference path for nn.Linear.
func GemmExBF16OutBF16RowMajorTransB(h *BlasHandle, m, n, k int, a, b, c uintptr) error {
	alpha := C.float(1.0)
	beta := C.float(0.0)
	rc := C.cublasGemmEx(
		h.handle,
		C.CUBLAS_OP_T, C.CUBLAS_OP_N,
		C.int(n), C.int(m), C.int(k),
		unsafe.Pointer(&alpha),
		unsafe.Pointer(b), C.CUDA_R_16BF, C.int(k),
		unsafe.Pointer(a), C.CUDA_R_16BF, C.int(k),
		unsafe.Pointer(&beta),
		unsafe.Pointer(c), C.CUDA_R_16BF, C.int(n),
		C.CUBLAS_COMPUTE_32F,
		C.CUBLAS_GEMM_DEFAULT,
	)
	if rc != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasGemmEx(BF16,BF16→BF16,T,N): status %d", int(rc))
	}
	return nil
}

// GemmExBF16BatchedRowMajor computes a stack of `count` row-major BF16 GEMMs
// with BF16 outputs (no transpose; both operands taken as-is). Used for
// attention's batched Q @ Kᵀ and attn @ V — both fully BF16 with F32
// internal accumulation.
//
//	A: (count, M, K), B: (count, K, N), C: (count, M, N)
//	C[i] = A[i] @ B[i]
func GemmExBF16BatchedRowMajor(h *BlasHandle, m, n, k int, a, b, c uintptr, count int) error {
	alpha := C.float(1.0)
	beta := C.float(0.0)
	rc := C.cublasGemmStridedBatchedEx(
		h.handle,
		C.CUBLAS_OP_N, C.CUBLAS_OP_N,
		C.int(n), C.int(m), C.int(k),
		unsafe.Pointer(&alpha),
		unsafe.Pointer(b), C.CUDA_R_16BF, C.int(n), C.longlong(k*n),
		unsafe.Pointer(a), C.CUDA_R_16BF, C.int(k), C.longlong(m*k),
		unsafe.Pointer(&beta),
		unsafe.Pointer(c), C.CUDA_R_16BF, C.int(n), C.longlong(m*n),
		C.int(count),
		C.CUBLAS_COMPUTE_32F,
		C.CUBLAS_GEMM_DEFAULT,
	)
	if rc != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasGemmStridedBatchedEx(BF16): status %d", int(rc))
	}
	return nil
}

// SgemmStridedBatchedRowMajor computes a stack of `count` row-major gemms:
//
//	A: (count, M, K), B: (count, K, N), C: (count, M, N)
//	C[i] = A[i] @ B[i]
//
// Per-slice strides are M*K / K*N / M*N floats. cuBLAS is column-major so we
// apply the same swap trick used by SgemmRowMajor.
func SgemmStridedBatchedRowMajor(h *BlasHandle, m, n, k int, a, b, c uintptr, count int) error {
	alpha := C.float(1.0)
	beta := C.float(0.0)
	rc := C.cublasSgemmStridedBatched(
		h.handle,
		C.CUBLAS_OP_N, C.CUBLAS_OP_N,
		C.int(n), C.int(m), C.int(k),
		&alpha,
		(*C.float)(unsafe.Pointer(b)), C.int(n), C.longlong(k*n),
		(*C.float)(unsafe.Pointer(a)), C.int(k), C.longlong(m*k),
		&beta,
		(*C.float)(unsafe.Pointer(c)), C.int(n), C.longlong(m*n),
		C.int(count),
	)
	if rc != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasSgemmStridedBatched: status %d", int(rc))
	}
	return nil
}

// DgemmRowMajor is the F64 counterpart of SgemmRowMajor.
func DgemmRowMajor(h *BlasHandle, m, n, k int, a, b, c uintptr) error {
	alpha := C.double(1.0)
	beta := C.double(0.0)
	rc := C.cublasDgemm(
		h.handle,
		C.CUBLAS_OP_N, C.CUBLAS_OP_N,
		C.int(n), C.int(m), C.int(k),
		&alpha,
		(*C.double)(unsafe.Pointer(b)), C.int(n),
		(*C.double)(unsafe.Pointer(a)), C.int(k),
		&beta,
		(*C.double)(unsafe.Pointer(c)), C.int(n),
	)
	if rc != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasDgemm: status %d", int(rc))
	}
	return nil
}

// DataType selects the cuBLAS storage type of a GEMM operand.
type DataType int

const (
	DataF16 DataType = iota
	DataBF16
	DataF32
)

func (d DataType) cuda() C.cudaDataType {
	switch d {
	case DataF16:
		return C.CUDA_R_16F
	case DataBF16:
		return C.CUDA_R_16BF
	default:
		return C.CUDA_R_32F
	}
}

// Size is the element width in bytes.
func (d DataType) Size() int {
	if d == DataF32 {
		return 4
	}
	return 2
}

// GemmExRowMajor computes C = A @ B (transB false; B is (K, N)) or
// C = A @ Bᵀ (transB true; B is (N, K)) in row-major semantics with F32
// accumulation, for any mix of F16 / BF16 / F32 operand and output types
// cuBLAS accepts (e.g. F16×F16→F16, F16×F16→F32, BF16×BF16→BF16).
//
//	A: (M, K), C: (M, N) — row-major contiguous on device.
func GemmExRowMajor(h *BlasHandle, m, n, k int, a, b, c uintptr, aType, bType, cType DataType, transB bool) error {
	alpha := C.float(1.0)
	beta := C.float(0.0)
	opB, ldb := C.cublasOperation_t(C.CUBLAS_OP_N), C.int(n)
	if transB {
		opB, ldb = C.cublasOperation_t(C.CUBLAS_OP_T), C.int(k)
	}
	rc := C.cublasGemmEx(
		h.handle,
		opB, C.CUBLAS_OP_N,
		C.int(n), C.int(m), C.int(k),
		unsafe.Pointer(&alpha),
		unsafe.Pointer(b), bType.cuda(), ldb,
		unsafe.Pointer(a), aType.cuda(), C.int(k),
		unsafe.Pointer(&beta),
		unsafe.Pointer(c), cType.cuda(), C.int(n),
		C.CUBLAS_COMPUTE_32F,
		C.CUBLAS_GEMM_DEFAULT,
	)
	if rc != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasGemmEx(%v×%v→%v, transB=%v): status %d", aType, bType, cType, transB, int(rc))
	}
	return nil
}

// GemmStridedBatchedExRowMajor computes C[i] = A[i] @ B[i] for `count`
// row-major slices with a single storage dtype (F16 / BF16 / F32) and F32
// accumulation.
//
//	A: (count, M, K), B: (count, K, N), C: (count, M, N)
func GemmStridedBatchedExRowMajor(h *BlasHandle, m, n, k int, a, b, c uintptr, dt DataType, count int) error {
	alpha := C.float(1.0)
	beta := C.float(0.0)
	rc := C.cublasGemmStridedBatchedEx(
		h.handle,
		C.CUBLAS_OP_N, C.CUBLAS_OP_N,
		C.int(n), C.int(m), C.int(k),
		unsafe.Pointer(&alpha),
		unsafe.Pointer(b), dt.cuda(), C.int(n), C.longlong(k*n),
		unsafe.Pointer(a), dt.cuda(), C.int(k), C.longlong(m*k),
		unsafe.Pointer(&beta),
		unsafe.Pointer(c), dt.cuda(), C.int(n), C.longlong(m*n),
		C.int(count),
		C.CUBLAS_COMPUTE_32F,
		C.CUBLAS_GEMM_DEFAULT,
	)
	if rc != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasGemmStridedBatchedEx(%v): status %d", dt, int(rc))
	}
	return nil
}

func (d DataType) String() string {
	switch d {
	case DataF16:
		return "F16"
	case DataBF16:
		return "BF16"
	default:
		return "F32"
	}
}
