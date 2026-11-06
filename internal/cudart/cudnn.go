//go:build !nocuda

package cudart

/*
#cgo LDFLAGS: -lcudnn
#cgo CFLAGS: -I/usr/local/cuda/include

#include <cudnn.h>
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// DnnHandle wraps a cudnnHandle_t. One per device, cached for the process
// lifetime. cuDNN handles are heavyweight (tens of MB of allocations on
// first kernel call), so creating one per launch would dominate runtime.
type DnnHandle struct {
	handle C.cudnnHandle_t
}

var dnnHandleMu sync.Mutex
var dnnHandleByOrd = map[int]*DnnHandle{}

// GetDnnHandle returns a cached cuDNN handle for the given device ordinal.
// Creates one on first use.
func GetDnnHandle(ordinal int) (*DnnHandle, error) {
	dnnHandleMu.Lock()
	defer dnnHandleMu.Unlock()
	if dh, ok := dnnHandleByOrd[ordinal]; ok {
		return dh, nil
	}
	if err := SetDevice(ordinal); err != nil {
		return nil, err
	}
	var h C.cudnnHandle_t
	if rc := C.cudnnCreate(&h); rc != C.CUDNN_STATUS_SUCCESS {
		return nil, fmt.Errorf("cudnnCreate: status %d (%s)", int(rc), C.GoString(C.cudnnGetErrorString(rc)))
	}
	dh := &DnnHandle{handle: h}
	dnnHandleByOrd[ordinal] = dh
	return dh, nil
}

// DnnDType is the subset of cudnnDataType_t we need.
type DnnDType int

const (
	DnnF32  DnnDType = iota // CUDNN_DATA_FLOAT
	DnnBF16                 // CUDNN_DATA_BFLOAT16
)

func (d DnnDType) cudnn() C.cudnnDataType_t {
	switch d {
	case DnnF32:
		return C.CUDNN_DATA_FLOAT
	case DnnBF16:
		return C.CUDNN_DATA_BFLOAT16
	}
	return C.CUDNN_DATA_FLOAT
}

// Conv2DForward runs y = conv2d(x, w) + bias (bias optional). Shapes are
// row-major NCHW:
//
//	x: (N, Cin,  H,    W)
//	w: (Cout, Cin, kH, kW)
//	bias: (Cout,) or nil
//	y: (N, Cout, Hout, Wout)
//
// All four storage tensors carry the same dtype `dt`. Internal compute is
// always F32 (cuDNN uses tensor cores when dt=BF16, plain SGEMM when F32).
//
// Algorithm is left to cuDNN's heuristic (CUDNN_CONVOLUTION_FWD_ALGO_*),
// which on Ampere/Ada/Blackwell typically picks an implicit-GEMM or
// Winograd path for 3×3 convs. Workspace is allocated per call — for SD's
// ~50 convs/step that's 50 cudaMallocs, but they're under 100 MB each and
// the alternative (caching by shape) would carry persistent device memory.
func Conv2DForward(
	h *DnnHandle,
	dt DnnDType,
	xPtr, wPtr, biasPtr, yPtr uintptr,
	n, cin, hIn, wIn int,
	cout, kH, kW int,
	hOut, wOut int,
	sH, sW, pH, pW int,
) error {
	dataType := dt.cudnn()

	// Tensor descriptors.
	var xDesc, yDesc, biasDesc C.cudnnTensorDescriptor_t
	var wDesc C.cudnnFilterDescriptor_t
	var convDesc C.cudnnConvolutionDescriptor_t

	if rc := C.cudnnCreateTensorDescriptor(&xDesc); rc != C.CUDNN_STATUS_SUCCESS {
		return cudnnErr("CreateTensorDescriptor(x)", rc)
	}
	defer C.cudnnDestroyTensorDescriptor(xDesc)
	if rc := C.cudnnCreateTensorDescriptor(&yDesc); rc != C.CUDNN_STATUS_SUCCESS {
		return cudnnErr("CreateTensorDescriptor(y)", rc)
	}
	defer C.cudnnDestroyTensorDescriptor(yDesc)
	if rc := C.cudnnCreateFilterDescriptor(&wDesc); rc != C.CUDNN_STATUS_SUCCESS {
		return cudnnErr("CreateFilterDescriptor", rc)
	}
	defer C.cudnnDestroyFilterDescriptor(wDesc)
	if rc := C.cudnnCreateConvolutionDescriptor(&convDesc); rc != C.CUDNN_STATUS_SUCCESS {
		return cudnnErr("CreateConvolutionDescriptor", rc)
	}
	defer C.cudnnDestroyConvolutionDescriptor(convDesc)

	if rc := C.cudnnSetTensor4dDescriptor(
		xDesc, C.CUDNN_TENSOR_NCHW, dataType,
		C.int(n), C.int(cin), C.int(hIn), C.int(wIn),
	); rc != C.CUDNN_STATUS_SUCCESS {
		return cudnnErr("SetTensor4dDescriptor(x)", rc)
	}
	if rc := C.cudnnSetTensor4dDescriptor(
		yDesc, C.CUDNN_TENSOR_NCHW, dataType,
		C.int(n), C.int(cout), C.int(hOut), C.int(wOut),
	); rc != C.CUDNN_STATUS_SUCCESS {
		return cudnnErr("SetTensor4dDescriptor(y)", rc)
	}
	if rc := C.cudnnSetFilter4dDescriptor(
		wDesc, dataType, C.CUDNN_TENSOR_NCHW,
		C.int(cout), C.int(cin), C.int(kH), C.int(kW),
	); rc != C.CUDNN_STATUS_SUCCESS {
		return cudnnErr("SetFilter4dDescriptor", rc)
	}
	// Conv math is always F32 for stability; tensor cores kick in
	// automatically when filter/input dtype is BF16.
	if rc := C.cudnnSetConvolution2dDescriptor(
		convDesc,
		C.int(pH), C.int(pW), C.int(sH), C.int(sW),
		C.int(1), C.int(1), // dilation
		C.CUDNN_CROSS_CORRELATION,
		C.CUDNN_DATA_FLOAT,
	); rc != C.CUDNN_STATUS_SUCCESS {
		return cudnnErr("SetConvolution2dDescriptor", rc)
	}
	if dt == DnnBF16 {
		// Allow tensor-core math for BF16 conv. cuDNN may still pick a
		// non-tensor-core path when the shape doesn't fit; this is just a
		// permission flag.
		if rc := C.cudnnSetConvolutionMathType(convDesc, C.CUDNN_TENSOR_OP_MATH); rc != C.CUDNN_STATUS_SUCCESS {
			return cudnnErr("SetConvolutionMathType", rc)
		}
	}

	// Pick the fastest algorithm cuDNN's heuristic returns. Asking for the
	// top-1 is effectively free — the heuristic doesn't run kernels.
	var algoPerf C.cudnnConvolutionFwdAlgoPerf_t
	var returned C.int
	if rc := C.cudnnGetConvolutionForwardAlgorithm_v7(
		h.handle, xDesc, wDesc, convDesc, yDesc,
		C.int(1), &returned, &algoPerf,
	); rc != C.CUDNN_STATUS_SUCCESS || returned < 1 {
		return cudnnErr("GetConvolutionForwardAlgorithm_v7", rc)
	}
	algo := algoPerf.algo

	// Workspace.
	var wsBytes C.size_t
	if rc := C.cudnnGetConvolutionForwardWorkspaceSize(
		h.handle, xDesc, wDesc, convDesc, yDesc, algo, &wsBytes,
	); rc != C.CUDNN_STATUS_SUCCESS {
		return cudnnErr("GetConvolutionForwardWorkspaceSize", rc)
	}
	var wsPtr uintptr
	if wsBytes > 0 {
		p, err := Malloc(uintptr(wsBytes))
		if err != nil {
			return fmt.Errorf("conv workspace malloc (%d bytes): %w", uint64(wsBytes), err)
		}
		wsPtr = p
		defer Free(wsPtr)
	}

	alpha := C.float(1.0)
	beta := C.float(0.0)
	if rc := C.cudnnConvolutionForward(
		h.handle,
		unsafe.Pointer(&alpha),
		xDesc, unsafe.Pointer(xPtr),
		wDesc, unsafe.Pointer(wPtr),
		convDesc, algo,
		unsafe.Pointer(wsPtr), wsBytes,
		unsafe.Pointer(&beta),
		yDesc, unsafe.Pointer(yPtr),
	); rc != C.CUDNN_STATUS_SUCCESS {
		return cudnnErr("ConvolutionForward", rc)
	}

	if biasPtr != 0 {
		// (1, Cout, 1, 1) per-channel add.
		if rc := C.cudnnCreateTensorDescriptor(&biasDesc); rc != C.CUDNN_STATUS_SUCCESS {
			return cudnnErr("CreateTensorDescriptor(bias)", rc)
		}
		defer C.cudnnDestroyTensorDescriptor(biasDesc)
		if rc := C.cudnnSetTensor4dDescriptor(
			biasDesc, C.CUDNN_TENSOR_NCHW, dataType,
			C.int(1), C.int(cout), C.int(1), C.int(1),
		); rc != C.CUDNN_STATUS_SUCCESS {
			return cudnnErr("SetTensor4dDescriptor(bias)", rc)
		}
		alpha2 := C.float(1.0)
		beta2 := C.float(1.0)
		if rc := C.cudnnAddTensor(
			h.handle,
			unsafe.Pointer(&alpha2),
			biasDesc, unsafe.Pointer(biasPtr),
			unsafe.Pointer(&beta2),
			yDesc, unsafe.Pointer(yPtr),
		); rc != C.CUDNN_STATUS_SUCCESS {
			return cudnnErr("AddTensor(bias)", rc)
		}
	}
	return nil
}

func cudnnErr(where string, rc C.cudnnStatus_t) error {
	return fmt.Errorf("cudnn %s: status %d (%s)", where, int(rc), C.GoString(C.cudnnGetErrorString(rc)))
}
