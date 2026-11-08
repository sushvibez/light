//go:build !nocuda && cgo

package cudart

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart -L/usr/lib/x86_64-linux-gnu -lcuda

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// DeviceCount returns the number of CUDA-capable devices visible to the
// process.
func DeviceCount() (int, error) {
	var count C.int
	if rc := C.cudaGetDeviceCount(&count); rc != C.cudaSuccess {
		return 0, runtimeError(rc, "cudaGetDeviceCount")
	}
	return int(count), nil
}

// SetDevice switches the calling thread's CUDA context to the given ordinal.
// Every cuda-using goroutine should pin itself with runtime.LockOSThread()
// and call SetDevice at startup; that pairing is handled at the higher layer.
func SetDevice(ordinal int) error {
	if rc := C.cudaSetDevice(C.int(ordinal)); rc != C.cudaSuccess {
		return runtimeError(rc, "cudaSetDevice")
	}
	return nil
}

// Malloc allocates `bytes` of device memory and returns the opaque pointer.
func Malloc(bytes uintptr) (uintptr, error) {
	if bytes == 0 {
		return 0, nil
	}
	var ptr unsafe.Pointer
	if rc := C.cudaMalloc(&ptr, C.size_t(bytes)); rc != C.cudaSuccess {
		return 0, runtimeError(rc, "cudaMalloc")
	}
	return uintptr(ptr), nil
}

// Free releases a device pointer. Safe on 0.
func Free(ptr uintptr) error {
	if ptr == 0 {
		return nil
	}
	if rc := C.cudaFree(unsafe.Pointer(ptr)); rc != C.cudaSuccess {
		return runtimeError(rc, "cudaFree")
	}
	return nil
}

// MallocHost allocates page-locked host memory, the only kind a host→device
// copy can leave in flight without blocking the caller.
func MallocHost(bytes uintptr) (uintptr, error) {
	var ptr unsafe.Pointer
	if rc := C.cudaMallocHost(&ptr, C.size_t(bytes)); rc != C.cudaSuccess {
		return 0, runtimeError(rc, "cudaMallocHost")
	}
	return uintptr(ptr), nil
}

// MemcpyHtoDAsync queues a copy from pinned host memory on the default
// stream: it returns at once, ordered after the kernels already queued, and
// the source must stay untouched until the copy has run.
func MemcpyHtoDAsync(dst, src, bytes uintptr) error {
	if bytes == 0 {
		return nil
	}
	rc := C.cudaMemcpyAsync(unsafe.Pointer(dst), unsafe.Pointer(src), C.size_t(bytes), C.cudaMemcpyHostToDevice, nil)
	if rc != C.cudaSuccess {
		return runtimeError(rc, "cudaMemcpyAsync HtoD")
	}
	return nil
}

// MemcpyHtoD copies from host slice src to device pointer dst.
func MemcpyHtoD(dst uintptr, src []byte) error {
	if len(src) == 0 {
		return nil
	}
	rc := C.cudaMemcpy(
		unsafe.Pointer(dst),
		unsafe.Pointer(&src[0]),
		C.size_t(len(src)),
		C.cudaMemcpyHostToDevice,
	)
	if rc != C.cudaSuccess {
		return runtimeError(rc, "cudaMemcpy HtoD")
	}
	return nil
}

// MemcpyDtoH copies from device pointer src to host slice dst.
func MemcpyDtoH(dst []byte, src uintptr) error {
	if len(dst) == 0 {
		return nil
	}
	rc := C.cudaMemcpy(
		unsafe.Pointer(&dst[0]),
		unsafe.Pointer(src),
		C.size_t(len(dst)),
		C.cudaMemcpyDeviceToHost,
	)
	if rc != C.cudaSuccess {
		return runtimeError(rc, "cudaMemcpy DtoH")
	}
	return nil
}

// MemcpyDtoD copies between two device pointers (same or different devices
// with peer access enabled).
func MemcpyDtoD(dst, src uintptr, bytes uintptr) error {
	if bytes == 0 {
		return nil
	}
	rc := C.cudaMemcpy(
		unsafe.Pointer(dst),
		unsafe.Pointer(src),
		C.size_t(bytes),
		C.cudaMemcpyDeviceToDevice,
	)
	if rc != C.cudaSuccess {
		return runtimeError(rc, "cudaMemcpy DtoD")
	}
	return nil
}

// Memcpy2DDtoD copies `height` rows of `width` bytes between device
// buffers whose rows are `spitch` / `dpitch` bytes apart: one call for a
// whole strided block (concat along an axis, KV-cache appends).
func Memcpy2DDtoD(dst, dpitch, src, spitch, width, height uintptr) error {
	if width == 0 || height == 0 {
		return nil
	}
	rc := C.cudaMemcpy2D(
		unsafe.Pointer(dst), C.size_t(dpitch),
		unsafe.Pointer(src), C.size_t(spitch),
		C.size_t(width), C.size_t(height),
		C.cudaMemcpyDeviceToDevice,
	)
	if rc != C.cudaSuccess {
		return runtimeError(rc, "cudaMemcpy2D DtoD")
	}
	return nil
}

// Memset fills device memory with the given byte.
func Memset(ptr uintptr, value byte, bytes uintptr) error {
	if bytes == 0 {
		return nil
	}
	rc := C.cudaMemset(unsafe.Pointer(ptr), C.int(value), C.size_t(bytes))
	if rc != C.cudaSuccess {
		return runtimeError(rc, "cudaMemset")
	}
	return nil
}

// DeviceSynchronize blocks until all work on the current device's default
// stream has completed. Use after launches for correctness during testing.
func DeviceSynchronize() error {
	if rc := C.cudaDeviceSynchronize(); rc != C.cudaSuccess {
		return runtimeError(rc, "cudaDeviceSynchronize")
	}
	return nil
}

// MemInfo returns (free, total) device-memory bytes on the current device.
func MemInfo() (free, total uintptr, err error) {
	var f, t C.size_t
	if rc := C.cudaMemGetInfo(&f, &t); rc != C.cudaSuccess {
		return 0, 0, runtimeError(rc, "cudaMemGetInfo")
	}
	return uintptr(f), uintptr(t), nil
}

// DeviceProperties is a small subset of cudaDeviceProp we actually need.
type DeviceProperties struct {
	Name           string
	ComputeMajor   int
	ComputeMinor   int
	TotalGlobalMem uintptr
	MultiProcCount int
	// SharedMemPerBlock is the static shared-memory budget per block in
	// bytes (cudaDeviceProp.sharedMemPerBlock); kernels that tile through
	// shared memory size their tiles from it.
	SharedMemPerBlock uintptr
}

// GetDeviceProperties fetches a (subset) of cudaDeviceProp for ordinal.
func GetDeviceProperties(ordinal int) (DeviceProperties, error) {
	var prop C.struct_cudaDeviceProp
	if rc := C.cudaGetDeviceProperties(&prop, C.int(ordinal)); rc != C.cudaSuccess {
		return DeviceProperties{}, runtimeError(rc, "cudaGetDeviceProperties")
	}
	name := C.GoString(&prop.name[0])
	return DeviceProperties{
		Name:              name,
		ComputeMajor:      int(prop.major),
		ComputeMinor:      int(prop.minor),
		TotalGlobalMem:    uintptr(prop.totalGlobalMem),
		MultiProcCount:    int(prop.multiProcessorCount),
		SharedMemPerBlock: uintptr(prop.sharedMemPerBlock),
	}, nil
}

func runtimeError(rc C.cudaError_t, op string) error {
	return fmt.Errorf("cudart: %s: %s (code %d)", op, C.GoString(C.cudaGetErrorString(rc)), int(rc))
}

// One-time driver init. cuInit must be called before any driver-API function.
var driverInitOnce sync.Once
var driverInitErr error

// EnsureDriver initializes the CUDA driver API exactly once. Returns any
// error from cuInit (stored and returned on subsequent calls).
func EnsureDriver() error {
	driverInitOnce.Do(func() {
		rc := C.cuInit(0)
		if rc != C.CUDA_SUCCESS {
			driverInitErr = driverError(rc, "cuInit")
		}
	})
	return driverInitErr
}

func driverError(rc C.CUresult, op string) error {
	var str *C.char
	C.cuGetErrorString(rc, &str)
	name := "?"
	if str != nil {
		name = C.GoString(str)
	}
	return fmt.Errorf("cudart: %s: %s (code %d)", op, name, int(rc))
}

// DeviceName reports device ordinal's name (e.g. "NVIDIA L4").
func DeviceName(ordinal int) (string, error) {
	p, err := GetDeviceProperties(ordinal)
	return p.Name, err
}
