//go:build !nocuda && cgo

package cudart

/*
#include <cuda.h>
#include <cuda_runtime.h>
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// ensureContext makes sure the calling OS thread has a current CUDA context
// before a driver-API call. The runtime API binds its primary context to a
// thread lazily, on that thread's first runtime call; a goroutine that is
// scheduled onto a fresh thread and goes straight to cuModuleLoadDataEx /
// cuLaunchKernel therefore sees "invalid device context" (CUDA_ERROR_INVALID
// CONTEXT, 201). Observed intermittently when many test binaries start at
// once. Binding the current runtime device's primary context here is cheap
// (a no-op when a context is already current) and makes the driver calls
// thread-agnostic.
//
// The check and the driver call it guards must run on the same OS thread:
// every cgo call may land on a different thread, so a goroutine can pass
// ensureContext on one thread and reach cuLaunchKernel on another that has
// no context (seen on an A100 as an intermittent code 201 during a long
// decode). Callers therefore hold runtime.LockOSThread across both.
func ensureContext() error {
	var ctx C.CUcontext
	if rc := C.cuCtxGetCurrent(&ctx); rc != C.CUDA_SUCCESS {
		return driverError(rc, "cuCtxGetCurrent")
	}
	if ctx != nil {
		return nil
	}
	return BindPrimaryContext()
}

// BindPrimaryContext makes the current runtime device's primary context
// current on the calling OS thread, replacing whatever the thread had. The
// driver occasionally reports CUDA_ERROR_INVALID_CONTEXT for a module load
// on a thread whose current context looks valid (observed with CUDA 13.1
// when a goroutine migrates threads between runtime-API and driver-API
// calls); re-binding the primary context and retrying resolves it.
func BindPrimaryContext() error {
	var ordinal C.int
	if rc := C.cudaGetDevice(&ordinal); rc != C.cudaSuccess {
		return runtimeError(rc, "cudaGetDevice")
	}
	var dev C.CUdevice
	if rc := C.cuDeviceGet(&dev, ordinal); rc != C.CUDA_SUCCESS {
		return driverError(rc, "cuDeviceGet")
	}
	var ctx C.CUcontext
	if rc := C.cuDevicePrimaryCtxRetain(&ctx, dev); rc != C.CUDA_SUCCESS {
		return driverError(rc, "cuDevicePrimaryCtxRetain")
	}
	if rc := C.cuCtxSetCurrent(ctx); rc != C.CUDA_SUCCESS {
		return driverError(rc, "cuCtxSetCurrent")
	}
	return nil
}

// Module wraps a loaded CUmodule — a collection of PTX-compiled kernels.
type Module struct {
	handle C.CUmodule
}

// Function is a specific kernel entry point within a Module.
type Function struct {
	handle C.CUfunction
}

// ModuleLoadData loads PTX text (must be null-terminated; we append \0 here).
// Uses cuModuleLoadDataEx so we can pass JIT options if/when we need them.
func ModuleLoadData(ptxText []byte) (*Module, error) {
	if err := EnsureDriver(); err != nil {
		return nil, err
	}
	runtime.LockOSThread() // ensureContext and the load on one thread
	defer runtime.UnlockOSThread()
	if err := ensureContext(); err != nil {
		return nil, err
	}
	// The driver scans for a null terminator when the input is PTX text, so
	// append one unless already present.
	buf := ptxText
	if len(buf) == 0 || buf[len(buf)-1] != 0 {
		buf = make([]byte, len(ptxText)+1)
		copy(buf, ptxText)
	}
	var mod C.CUmodule
	rc := C.cuModuleLoadDataEx(&mod, unsafe.Pointer(&buf[0]), 0, nil, nil)
	if rc != C.CUDA_SUCCESS {
		return nil, driverError(rc, "cuModuleLoadDataEx")
	}
	return &Module{handle: mod}, nil
}

// Unload releases the module's device resources.
func (m *Module) Unload() error {
	if m == nil || m.handle == nil {
		return nil
	}
	if rc := C.cuModuleUnload(m.handle); rc != C.CUDA_SUCCESS {
		return driverError(rc, "cuModuleUnload")
	}
	m.handle = nil
	return nil
}

// GetFunction looks up a kernel entry point by name.
func (m *Module) GetFunction(name string) (*Function, error) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	var fn C.CUfunction
	rc := C.cuModuleGetFunction(&fn, m.handle, cname)
	if rc != C.CUDA_SUCCESS {
		return nil, driverError(rc, fmt.Sprintf("cuModuleGetFunction(%q)", name))
	}
	return &Function{handle: fn}, nil
}

// LaunchKernel invokes fn with the given grid/block geometry.
// `args` is a slice of pointers — each element is an unsafe.Pointer to the
// argument value (e.g. &ptr, &size). The caller owns the argument storage; it
// must remain reachable until cuLaunchKernel returns.
//
// CGo's pointer-safety rules reject passing a Go pointer whose target contains
// further Go pointers. `args` is exactly that, so we use runtime.Pinner to
// pin each inner pointer plus the slice backing.
func LaunchKernel(
	fn *Function,
	gridX, gridY, gridZ uint32,
	blockX, blockY, blockZ uint32,
	sharedBytes uint32,
	stream uintptr,
	args []unsafe.Pointer,
) error {
	tally(&counters.Launches)
	runtime.LockOSThread() // ensureContext and the launch on one thread
	defer runtime.UnlockOSThread()
	if err := ensureContext(); err != nil {
		return err
	}
	var pinner runtime.Pinner
	defer pinner.Unpin()
	for _, p := range args {
		if p != nil {
			pinner.Pin(p)
		}
	}
	var argsPtr unsafe.Pointer
	if len(args) > 0 {
		pinner.Pin(&args[0])
		argsPtr = unsafe.Pointer(&args[0])
	}
	launch := func() C.CUresult {
		return C.cuLaunchKernel(
			fn.handle,
			C.uint(gridX), C.uint(gridY), C.uint(gridZ),
			C.uint(blockX), C.uint(blockY), C.uint(blockZ),
			C.uint(sharedBytes),
			C.CUstream(unsafe.Pointer(stream)),
			(*unsafe.Pointer)(argsPtr),
			nil,
		)
	}
	rc := launch()
	if rc == C.CUDA_ERROR_INVALID_CONTEXT {
		// Same recovery as module loads: re-bind the primary context on
		// this (locked) thread and retry once.
		if err := BindPrimaryContext(); err != nil {
			return fmt.Errorf("%w (after %v)", err, driverError(rc, "cuLaunchKernel"))
		}
		rc = launch()
	}
	if rc != C.CUDA_SUCCESS {
		return driverError(rc, "cuLaunchKernel")
	}
	return nil
}

// CtxSynchronize blocks until all work on the current context is complete.
func CtxSynchronize() error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := ensureContext(); err != nil {
		return err
	}
	if rc := C.cuCtxSynchronize(); rc != C.CUDA_SUCCESS {
		return driverError(rc, "cuCtxSynchronize")
	}
	return nil
}
