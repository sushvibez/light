//go:build !nocuda && cgo

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
	"os"
	"sync"
	"unsafe"
)

// convTF32 lets F32 convolutions use TF32 tensor cores (LIGHT_CONV_TF32=1).
var convTF32 = os.Getenv("LIGHT_CONV_TF32") == "1"

// convProfile prints each convolution's algorithm and workspace (LIGHT_CONV_PROFILE=1).
var convProfile = os.Getenv("LIGHT_CONV_PROFILE") == "1"

// convAlgo forces a forward algorithm (LIGHT_CONV_ALGO=0..7), for measurement.
var convAlgo = func() int {
	if v := os.Getenv("LIGHT_CONV_ALGO"); v != "" {
		return int(v[0] - '0')
	}
	return -1
}()

// DnnHandle wraps a cudnnHandle_t. One per device, cached for the process
// lifetime. cuDNN handles are heavyweight (tens of MB of allocations on
// first kernel call), so creating one per launch would dominate runtime.
type DnnHandle struct {
	handle C.cudnnHandle_t
	stream uintptr
	ord    int

	planMu sync.Mutex
	plans  map[convKey]*convPlan
}

// ScratchAlloc, when set (the storage package's caching pool), supplies
// convolution workspaces: a block for the device and a release that hands
// it back. Without it a workspace is a cudaMalloc and a cudaFree per call,
// and cudaFree waits for the device (0.7 ms on a 3 ms BF16 convolution).
// Releasing right after the launch is safe on one stream: whatever takes
// the block next is queued behind the convolution.
var ScratchAlloc func(ord int, bytes uintptr) (ptr uintptr, release func(), err error)

var dnnHandleMu sync.Mutex
var dnnHandleByOrd = map[int]*DnnHandle{}

// GetDnnHandle returns a cached cuDNN handle for the given device ordinal.
// Creates one on first use.
func GetDnnHandle(ordinal int) (*DnnHandle, error) {
	dnnHandleMu.Lock()
	defer dnnHandleMu.Unlock()
	if dh, ok := dnnHandleByOrd[ordinal]; ok {
		return dh, dh.useStream(current.Load())
	}
	if err := SetDevice(ordinal); err != nil {
		return nil, err
	}
	var h C.cudnnHandle_t
	if rc := C.cudnnCreate(&h); rc != C.CUDNN_STATUS_SUCCESS {
		return nil, fmt.Errorf("cudnnCreate: status %d (%s)", int(rc), C.GoString(C.cudnnGetErrorString(rc)))
	}
	dh := &DnnHandle{handle: h, ord: ordinal}
	dnnHandleByOrd[ordinal] = dh
	return dh, dh.useStream(current.Load())
}

// useStream points the handle at s (the package's current stream) when
// that changed since the last call, so convolutions order with the rest.
func (dh *DnnHandle) useStream(s uintptr) error {
	if dh.stream == s {
		return nil
	}
	if rc := C.cudnnSetStream(dh.handle, C.cudaStream_t(unsafe.Pointer(s))); rc != C.CUDNN_STATUS_SUCCESS {
		return fmt.Errorf("cudnnSetStream: status %d (%s)", int(rc), C.GoString(C.cudnnGetErrorString(rc)))
	}
	dh.stream = s
	return nil
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

// convKey identifies a convolution's descriptors and algorithm choice.
type convKey struct {
	dt               DnnDType
	n, cin, hIn, wIn int
	cout, kH, kW     int
	hOut, wOut       int
	sH, sW, pH, pW   int
}

// convPlan is what cuDNN needs besides the data: the four descriptors,
// the heuristic's algorithm and its workspace size. Building it costs
// 0.3 ms (BF16) to 0.6 ms (F32) of host time before the device starts,
// 10 % of a 3 ms convolution, so plans are kept per shape.
type convPlan struct {
	xDesc, yDesc C.cudnnTensorDescriptor_t
	biasDesc     C.cudnnTensorDescriptor_t // (1, Cout, 1, 1)
	wDesc        C.cudnnFilterDescriptor_t
	convDesc     C.cudnnConvolutionDescriptor_t
	algo         C.cudnnConvolutionFwdAlgo_t
	wsBytes      C.size_t
}

func (p *convPlan) destroy() {
	if p.xDesc != nil {
		C.cudnnDestroyTensorDescriptor(p.xDesc)
	}
	if p.yDesc != nil {
		C.cudnnDestroyTensorDescriptor(p.yDesc)
	}
	if p.biasDesc != nil {
		C.cudnnDestroyTensorDescriptor(p.biasDesc)
	}
	if p.wDesc != nil {
		C.cudnnDestroyFilterDescriptor(p.wDesc)
	}
	if p.convDesc != nil {
		C.cudnnDestroyConvolutionDescriptor(p.convDesc)
	}
}

// convPlanCap bounds the plans kept per handle (a VAE has a few dozen
// distinct convolution shapes per resolution).
const convPlanCap = 256

// convPlan returns the cached plan for k, building it on first use.
func (h *DnnHandle) convPlan(k convKey) (*convPlan, error) {
	h.planMu.Lock()
	defer h.planMu.Unlock()
	if p, ok := h.plans[k]; ok {
		return p, nil
	}
	p, err := h.buildConvPlan(k)
	if err != nil {
		p.destroy()
		return nil, err
	}
	if len(h.plans) >= convPlanCap {
		for _, old := range h.plans {
			old.destroy()
		}
		h.plans = nil
	}
	if h.plans == nil {
		h.plans = map[convKey]*convPlan{}
	}
	h.plans[k] = p
	return p, nil
}

func (h *DnnHandle) buildConvPlan(k convKey) (*convPlan, error) {
	dt, dataType := k.dt, k.dt.cudnn()
	n, cin, hIn, wIn, cout, kH, kW, hOut, wOut := k.n, k.cin, k.hIn, k.wIn, k.cout, k.kH, k.kW, k.hOut, k.wOut
	sH, sW, pH, pW := k.sH, k.sW, k.pH, k.pW
	p := &convPlan{}
	if rc := C.cudnnCreateTensorDescriptor(&p.xDesc); rc != C.CUDNN_STATUS_SUCCESS {
		return p, cudnnErr("CreateTensorDescriptor(x)", rc)
	}
	if rc := C.cudnnCreateTensorDescriptor(&p.yDesc); rc != C.CUDNN_STATUS_SUCCESS {
		return p, cudnnErr("CreateTensorDescriptor(y)", rc)
	}
	if rc := C.cudnnCreateFilterDescriptor(&p.wDesc); rc != C.CUDNN_STATUS_SUCCESS {
		return p, cudnnErr("CreateFilterDescriptor", rc)
	}
	if rc := C.cudnnCreateConvolutionDescriptor(&p.convDesc); rc != C.CUDNN_STATUS_SUCCESS {
		return p, cudnnErr("CreateConvolutionDescriptor", rc)
	}

	if rc := C.cudnnSetTensor4dDescriptor(
		p.xDesc, C.CUDNN_TENSOR_NCHW, dataType,
		C.int(n), C.int(cin), C.int(hIn), C.int(wIn),
	); rc != C.CUDNN_STATUS_SUCCESS {
		return p, cudnnErr("SetTensor4dDescriptor(x)", rc)
	}
	if rc := C.cudnnSetTensor4dDescriptor(
		p.yDesc, C.CUDNN_TENSOR_NCHW, dataType,
		C.int(n), C.int(cout), C.int(hOut), C.int(wOut),
	); rc != C.CUDNN_STATUS_SUCCESS {
		return p, cudnnErr("SetTensor4dDescriptor(y)", rc)
	}
	if rc := C.cudnnSetFilter4dDescriptor(
		p.wDesc, dataType, C.CUDNN_TENSOR_NCHW,
		C.int(cout), C.int(cin), C.int(kH), C.int(kW),
	); rc != C.CUDNN_STATUS_SUCCESS {
		return p, cudnnErr("SetFilter4dDescriptor", rc)
	}
	if rc := C.cudnnCreateTensorDescriptor(&p.biasDesc); rc != C.CUDNN_STATUS_SUCCESS {
		return p, cudnnErr("CreateTensorDescriptor(bias)", rc)
	}
	if rc := C.cudnnSetTensor4dDescriptor(
		p.biasDesc, C.CUDNN_TENSOR_NCHW, dataType,
		C.int(1), C.int(cout), C.int(1), C.int(1),
	); rc != C.CUDNN_STATUS_SUCCESS {
		return p, cudnnErr("SetTensor4dDescriptor(bias)", rc)
	}
	// Conv math is always F32 for stability; tensor cores kick in
	// automatically when filter/input dtype is BF16.
	if rc := C.cudnnSetConvolution2dDescriptor(
		p.convDesc,
		C.int(pH), C.int(pW), C.int(sH), C.int(sW),
		C.int(1), C.int(1), // dilation
		C.CUDNN_CROSS_CORRELATION,
		C.CUDNN_DATA_FLOAT,
	); rc != C.CUDNN_STATUS_SUCCESS {
		return p, cudnnErr("SetConvolution2dDescriptor", rc)
	}
	math := C.cudnnMathType_t(C.CUDNN_FMA_MATH)
	if dt == DnnBF16 || convTF32 {
		// Allow tensor-core math for BF16 conv. cuDNN may still pick a
		// non-tensor-core path when the shape doesn't fit; this is just a
		// permission flag.
		math = C.CUDNN_TENSOR_OP_MATH
	}
	// F32 convs compute in true F32: cuDNN's default math on Ampere and
	// later permits TF32 (10-bit mantissa inputs), which puts a diffusion
	// VAE ~1e-3 off its F32 reference. LIGHT_CONV_TF32=1 allows it.
	if rc := C.cudnnSetConvolutionMathType(p.convDesc, math); rc != C.CUDNN_STATUS_SUCCESS {
		return p, cudnnErr("SetConvolutionMathType", rc)
	}

	// Pick the fastest algorithm cuDNN's heuristic returns. Asking for the
	// top-1 is effectively free — the heuristic doesn't run kernels.
	var algoPerf C.cudnnConvolutionFwdAlgoPerf_t
	var returned C.int
	if rc := C.cudnnGetConvolutionForwardAlgorithm_v7(
		h.handle, p.xDesc, p.wDesc, p.convDesc, p.yDesc,
		C.int(1), &returned, &algoPerf,
	); rc != C.CUDNN_STATUS_SUCCESS || returned < 1 {
		return p, cudnnErr("GetConvolutionForwardAlgorithm_v7", rc)
	}
	algo := algoPerf.algo
	if convAlgo >= 0 {
		algo = C.cudnnConvolutionFwdAlgo_t(convAlgo)
	}

	// Workspace.
	if rc := C.cudnnGetConvolutionForwardWorkspaceSize(
		h.handle, p.xDesc, p.wDesc, p.convDesc, p.yDesc, algo, &p.wsBytes,
	); rc != C.CUDNN_STATUS_SUCCESS {
		return p, cudnnErr("GetConvolutionForwardWorkspaceSize", rc)
	}
	p.algo = algo
	if convProfile {
		fmt.Fprintf(os.Stderr, "cudnn conv dt=%d (%d,%d,%d,%d)->%d k%dx%d: algo %d, workspace %d MB\n",
			int(dt), n, cin, hIn, wIn, cout, kH, kW, int(algo), int(p.wsBytes)>>20)
	}
	return p, nil
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
// Winograd path for 3×3 convs. Descriptors, algorithm and workspace size
// are cached per shape (convPlan); the workspace itself comes from
// ScratchAlloc and goes back as soon as the convolution is queued.
func Conv2DForward(
	h *DnnHandle,
	dt DnnDType,
	xPtr, wPtr, biasPtr, yPtr uintptr,
	n, cin, hIn, wIn int,
	cout, kH, kW int,
	hOut, wOut int,
	sH, sW, pH, pW int,
) error {
	plan, err := h.convPlan(convKey{dt, n, cin, hIn, wIn, cout, kH, kW, hOut, wOut, sH, sW, pH, pW})
	if err != nil {
		return err
	}
	xDesc, yDesc, wDesc, convDesc := plan.xDesc, plan.yDesc, plan.wDesc, plan.convDesc
	algo, wsBytes := plan.algo, plan.wsBytes
	biasDesc := plan.biasDesc

	var wsPtr uintptr
	if wsBytes > 0 {
		var p uintptr
		var err error
		release := func() { Free(p) }
		if ScratchAlloc != nil {
			p, release, err = ScratchAlloc(h.ord, uintptr(wsBytes))
		} else {
			p, err = MallocRetry(uintptr(wsBytes))
		}
		if err != nil {
			// No room for the workspace even after freeing what can be freed:
			// implicit GEMM needs none (slower, but the convolution still runs).
			algo = C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
			wsBytes = 0
		} else {
			wsPtr = p
			defer release()
		}
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
