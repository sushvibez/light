//go:build !nocuda && cgo

package cudart

/*
#include <cuda_runtime.h>
*/
import "C"

import (
	"sync/atomic"
	"unsafe"
)

// current is the stream every launch, copy and memset goes to when the
// caller passes none: 0 is the legacy default stream. A decode graph
// capture needs all of a step's work on one non-default stream, so
// SetStream switches the whole package (and cuBLAS / cuBLASLt through
// their handles) over; it stays switched, which is harmless outside
// capture because the stream is created non-blocking and the synchronous
// host copies wait on it explicitly.
var current atomic.Uintptr

// SetStream makes s the stream for subsequent work (0: legacy default).
// The caller synchronises the device first so work queued on the old
// stream is complete.
func SetStream(s uintptr) { current.Store(s) }

// Stream returns the stream subsequent work goes to.
func Stream() uintptr { return current.Load() }

// StreamCreate creates a non-blocking stream: it does not synchronise with
// the legacy default stream, which is what graph capture requires.
func StreamCreate() (uintptr, error) {
	var s C.cudaStream_t
	if rc := C.cudaStreamCreateWithFlags(&s, C.cudaStreamNonBlocking); rc != C.cudaSuccess {
		return 0, runtimeError(rc, "cudaStreamCreateWithFlags")
	}
	return uintptr(unsafe.Pointer(s)), nil
}

// StreamDestroy releases a stream created by StreamCreate.
func StreamDestroy(s uintptr) error {
	if rc := C.cudaStreamDestroy(C.cudaStream_t(unsafe.Pointer(s))); rc != C.cudaSuccess {
		return runtimeError(rc, "cudaStreamDestroy")
	}
	return nil
}

// StreamSynchronize waits for the work queued on s.
func StreamSynchronize(s uintptr) error {
	tally(&counters.Syncs)
	if rc := C.cudaStreamSynchronize(C.cudaStream_t(unsafe.Pointer(s))); rc != C.cudaSuccess {
		return runtimeError(rc, "cudaStreamSynchronize")
	}
	return nil
}

// StreamBeginCapture starts recording the work queued on s into a graph
// instead of running it. Relaxed mode: allocations and frees from any
// thread (the storage pool, GC finalizers) stay legal during capture.
func StreamBeginCapture(s uintptr) error {
	if rc := C.cudaStreamBeginCapture(C.cudaStream_t(unsafe.Pointer(s)), C.cudaStreamCaptureModeRelaxed); rc != C.cudaSuccess {
		return runtimeError(rc, "cudaStreamBeginCapture")
	}
	return nil
}

// StreamEndCapture stops recording and returns the graph. An operation
// that could not be captured (a synchronous copy, say) invalidated the
// capture earlier; this reports it, and the stream is usable again.
func StreamEndCapture(s uintptr) (uintptr, error) {
	var g C.cudaGraph_t
	rc := C.cudaStreamEndCapture(C.cudaStream_t(unsafe.Pointer(s)), &g)
	if rc != C.cudaSuccess {
		C.cudaGetLastError() // clear the sticky capture error
		return 0, runtimeError(rc, "cudaStreamEndCapture")
	}
	return uintptr(unsafe.Pointer(g)), nil
}

// GraphInstantiate builds an executable from a captured graph.
func GraphInstantiate(g uintptr) (uintptr, error) {
	var e C.cudaGraphExec_t
	if rc := C.cudaGraphInstantiate(&e, C.cudaGraph_t(unsafe.Pointer(g)), 0); rc != C.cudaSuccess {
		return 0, runtimeError(rc, "cudaGraphInstantiate")
	}
	return uintptr(unsafe.Pointer(e)), nil
}

// GraphNodeCount reports how many nodes a captured graph holds.
func GraphNodeCount(g uintptr) (int, error) {
	var n C.size_t
	if rc := C.cudaGraphGetNodes(C.cudaGraph_t(unsafe.Pointer(g)), nil, &n); rc != C.cudaSuccess {
		return 0, runtimeError(rc, "cudaGraphGetNodes")
	}
	return int(n), nil
}

// GraphLaunch runs an instantiated graph on s (0: the current stream).
func GraphLaunch(exec, s uintptr) error {
	tally(&counters.GraphLaunches)
	if s == 0 {
		s = current.Load()
	}
	if rc := C.cudaGraphLaunch(C.cudaGraphExec_t(unsafe.Pointer(exec)), C.cudaStream_t(unsafe.Pointer(s))); rc != C.cudaSuccess {
		return runtimeError(rc, "cudaGraphLaunch")
	}
	return nil
}

// GraphExecDestroy frees an executable graph.
func GraphExecDestroy(exec uintptr) error {
	if rc := C.cudaGraphExecDestroy(C.cudaGraphExec_t(unsafe.Pointer(exec))); rc != C.cudaSuccess {
		return runtimeError(rc, "cudaGraphExecDestroy")
	}
	return nil
}

// GraphDestroy frees a captured graph (its executables stay valid).
func GraphDestroy(g uintptr) error {
	if rc := C.cudaGraphDestroy(C.cudaGraph_t(unsafe.Pointer(g))); rc != C.cudaSuccess {
		return runtimeError(rc, "cudaGraphDestroy")
	}
	return nil
}

// MemcpyHtoDAsyncOn queues a host→device copy on s from memory that must
// stay valid until the copy ran (pinned memory, a Go buffer the caller
// pins or keeps alive and does not rewrite). Captured into a graph it
// replays as a copy from that same host address.
func MemcpyHtoDAsyncOn(dst, src, bytes, s uintptr) error {
	tally(&counters.HtoD)
	if bytes == 0 {
		return nil
	}
	rc := C.cudaMemcpyAsync(unsafe.Pointer(dst), unsafe.Pointer(src), C.size_t(bytes), C.cudaMemcpyHostToDevice, C.cudaStream_t(unsafe.Pointer(s)))
	if rc != C.cudaSuccess {
		return runtimeError(rc, "cudaMemcpyAsync HtoD")
	}
	return nil
}
