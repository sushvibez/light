//go:build !nocuda && cgo

package cudart

/*
#cgo LDFLAGS: -lcublasLt

#include <cublasLt.h>
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// LtHandle wraps a cublasLtHandle_t plus a device workspace for the
// heuristic-selected algorithms. One per device, process lifetime.
type LtHandle struct {
	handle    C.cublasLtHandle_t
	workspace uintptr
	wsBytes   uintptr
}

const ltWorkspaceBytes = 32 << 20 // cuBLASLt's recommended size for Hopper+

var ltHandleMu sync.Mutex
var ltHandleByOrd = map[int]*LtHandle{}

// GetLtHandle returns the cached cuBLASLt handle for a device ordinal.
func GetLtHandle(ordinal int) (*LtHandle, error) {
	ltHandleMu.Lock()
	defer ltHandleMu.Unlock()
	if h, ok := ltHandleByOrd[ordinal]; ok {
		return h, nil
	}
	if err := SetDevice(ordinal); err != nil {
		return nil, err
	}
	var h C.cublasLtHandle_t
	if rc := C.cublasLtCreate(&h); rc != C.CUBLAS_STATUS_SUCCESS {
		return nil, fmt.Errorf("cublasLtCreate: status %d", int(rc))
	}
	ws, err := Malloc(ltWorkspaceBytes)
	if err != nil {
		C.cublasLtDestroy(h)
		return nil, err
	}
	lt := &LtHandle{handle: h, workspace: ws, wsBytes: ltWorkspaceBytes}
	ltHandleByOrd[ordinal] = lt
	return lt, nil
}

// MatmulFP8RowMajor computes, in row-major semantics,
//
//	C (M, N) = aScale · bScale · A (M, K) @ B (N, K)ᵀ
//
// with A and B stored as F8E4M3 and C at cType (F16, BF16 or F32), F32
// accumulation. aScale / bScale are device pointers to one float each (the
// de-quantisation factors; pass 0 for an implicit 1.0). This is cuBLASLt's
// "TN" FP8 GEMM: in column-major terms the first operand is Bᵀ, which is why
// the (N, K) weight layout of nn.Linear is the natural one here.
//
// Constraints from cuBLASLt: K must be a multiple of 16, N·sizeof(C) a
// multiple of 16 bytes, all pointers 16-byte aligned.
func MatmulFP8RowMajor(lt *LtHandle, m, n, k int, a, b, c uintptr, aScale, bScale uintptr, cType DataType) error {
	tally(&counters.Gemms)
	if k%16 != 0 {
		return fmt.Errorf("cublasLt fp8: K=%d must be a multiple of 16", k)
	}
	if (n*cType.Size())%16 != 0 {
		return fmt.Errorf("cublasLt fp8: N=%d at %v must give a 16-byte row stride", n, cType)
	}
	var opDesc C.cublasLtMatmulDesc_t
	if rc := C.cublasLtMatmulDescCreate(&opDesc, C.CUBLAS_COMPUTE_32F, C.CUDA_R_32F); rc != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasLtMatmulDescCreate: status %d", int(rc))
	}
	defer C.cublasLtMatmulDescDestroy(opDesc)
	setDesc := func(attr C.cublasLtMatmulDescAttributes_t, p unsafe.Pointer, size uintptr) error {
		if rc := C.cublasLtMatmulDescSetAttribute(opDesc, attr, p, C.size_t(size)); rc != C.CUBLAS_STATUS_SUCCESS {
			return fmt.Errorf("cublasLtMatmulDescSetAttribute(%d): status %d", int(attr), int(rc))
		}
		return nil
	}
	transA, transB := C.cublasOperation_t(C.CUBLAS_OP_T), C.cublasOperation_t(C.CUBLAS_OP_N)
	if err := setDesc(C.CUBLASLT_MATMUL_DESC_TRANSA, unsafe.Pointer(&transA), unsafe.Sizeof(transA)); err != nil {
		return err
	}
	if err := setDesc(C.CUBLASLT_MATMUL_DESC_TRANSB, unsafe.Pointer(&transB), unsafe.Sizeof(transB)); err != nil {
		return err
	}
	// First cuBLASLt operand is our B (the weight), second is our A.
	if bScale != 0 {
		if err := setDesc(C.CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, unsafe.Pointer(&bScale), unsafe.Sizeof(bScale)); err != nil {
			return err
		}
	}
	if aScale != 0 {
		if err := setDesc(C.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, unsafe.Pointer(&aScale), unsafe.Sizeof(aScale)); err != nil {
			return err
		}
	}
	layout := func(dt C.cudaDataType, rows, cols, ld int) (C.cublasLtMatrixLayout_t, error) {
		var l C.cublasLtMatrixLayout_t
		if rc := C.cublasLtMatrixLayoutCreate(&l, dt, C.uint64_t(rows), C.uint64_t(cols), C.int64_t(ld)); rc != C.CUBLAS_STATUS_SUCCESS {
			return nil, fmt.Errorf("cublasLtMatrixLayoutCreate: status %d", int(rc))
		}
		return l, nil
	}
	// Stored (pre-op) shapes in column-major: weight (K, N) ld K; activation
	// (K, M) ld K; result (N, M) ld N.
	wDesc, err := layout(C.CUDA_R_8F_E4M3, k, n, k)
	if err != nil {
		return err
	}
	defer C.cublasLtMatrixLayoutDestroy(wDesc)
	xDesc, err := layout(C.CUDA_R_8F_E4M3, k, m, k)
	if err != nil {
		return err
	}
	defer C.cublasLtMatrixLayoutDestroy(xDesc)
	cDesc, err := layout(cType.cuda(), n, m, n)
	if err != nil {
		return err
	}
	defer C.cublasLtMatrixLayoutDestroy(cDesc)

	var pref C.cublasLtMatmulPreference_t
	if rc := C.cublasLtMatmulPreferenceCreate(&pref); rc != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasLtMatmulPreferenceCreate: status %d", int(rc))
	}
	defer C.cublasLtMatmulPreferenceDestroy(pref)
	ws := C.size_t(lt.wsBytes)
	if rc := C.cublasLtMatmulPreferenceSetAttribute(pref, C.CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, unsafe.Pointer(&ws), C.size_t(unsafe.Sizeof(ws))); rc != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasLtMatmulPreferenceSetAttribute: status %d", int(rc))
	}
	var heur C.cublasLtMatmulHeuristicResult_t
	var found C.int
	if rc := C.cublasLtMatmulAlgoGetHeuristic(lt.handle, opDesc, wDesc, xDesc, cDesc, cDesc, pref, 1, &heur, &found); rc != C.CUBLAS_STATUS_SUCCESS || found == 0 {
		return fmt.Errorf("cublasLtMatmulAlgoGetHeuristic: status %d, %d algos (FP8 GEMM needs sm_89+ and K%%16==0)", int(rc), int(found))
	}
	alpha := C.float(1.0)
	beta := C.float(0.0)
	rc := C.cublasLtMatmul(lt.handle, opDesc,
		unsafe.Pointer(&alpha),
		unsafe.Pointer(b), wDesc,
		unsafe.Pointer(a), xDesc,
		unsafe.Pointer(&beta),
		unsafe.Pointer(c), cDesc,
		unsafe.Pointer(c), cDesc,
		&heur.algo, unsafe.Pointer(lt.workspace), ws, nil)
	if rc != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasLtMatmul(FP8→%v): status %d", cType, int(rc))
	}
	return nil
}
