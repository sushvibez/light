package storage

import (
	"runtime"
	"sync/atomic"

	"light/device"
	"light/dtype"
)

// Storage is a refcounted buffer on a specific device. A single Storage may be
// shared by multiple Tensors (views); the buffer is released when the refcount
// hits zero. The release callback is nil for CPU (Go GC reclaims the backing
// slice automatically) and typically a cudaFree wrapper for CUDA.
type Storage struct {
	device device.Device
	dtype  dtype.DType
	nelems int
	bytes  int

	// CPU-only: the Go-managed backing slice. Nil on CUDA. Holding this keeps
	// the underlying array alive as long as Storage is alive.
	cpu []byte

	// cudaPtr is set by the cuda-tagged constructor. Kept on the struct so
	// CUDA support can land without changing Storage's layout; unused in
	// CPU-only builds.
	cudaPtr uintptr //nolint:unused // used by cuda build

	// gpuHandle is an opaque backend handle for non-CUDA accelerators (set by
	// the webgpu-tagged constructor to a *wgpu.Buffer wrapper). Kept untyped
	// so the struct layout is identical across build tags.
	gpuHandle any //nolint:unused // used by webgpu build

	refcount atomic.Int32
	release  func()
}

// NewCPU allocates a new CPU-backed Storage. The backing byte slice is
// allocated to dtype.ByteSize(nelems) bytes (block-rounded for quantized).
func NewCPU(dt dtype.DType, nelems int) *Storage {
	n := dt.ByteSize(nelems)
	return &Storage{
		device: device.CPU(),
		dtype:  dt,
		nelems: nelems,
		bytes:  n,
		cpu:    make([]byte, n),
	}
}

// WrapCPU wraps an existing []byte as a Storage without copying. The caller
// is responsible for ensuring the slice outlives all users of the Storage.
// nelems is the logical element count.
func WrapCPU(dt dtype.DType, nelems int, buf []byte) *Storage {
	return &Storage{
		device: device.CPU(),
		dtype:  dt,
		nelems: nelems,
		bytes:  len(buf),
		cpu:    buf,
	}
}

// Device returns the device this storage lives on.
func (s *Storage) Device() device.Device { return s.device }

// DType returns the element type.
func (s *Storage) DType() dtype.DType { return s.dtype }

// Len returns the logical element count.
func (s *Storage) Len() int { return s.nelems }

// Bytes returns the raw byte count of the buffer (block-rounded for quantized).
func (s *Storage) Bytes() int { return s.bytes }

// CPUBytes returns the underlying byte slice. Panics if the storage is not
// CPU-backed. Callers must not keep references to the returned slice past the
// lifetime of the Storage unless they also hold a Track'd reference.
func (s *Storage) CPUBytes() []byte {
	if s.device.Kind() != device.KindCPU {
		panic("storage.CPUBytes: not a CPU storage")
	}
	return s.cpu
}

// RefCount returns the current refcount value (for tests/diagnostics).
func (s *Storage) RefCount() int32 { return s.refcount.Load() }

// Acquire increments the refcount. Callers should pair with Release.
func (s *Storage) Acquire() { s.refcount.Add(1) }

// Release decrements the refcount. When it reaches zero, the release callback
// (if any) is invoked. Must be paired with an earlier Acquire (directly or
// via Track).
func (s *Storage) Release() {
	if s.refcount.Add(-1) == 0 && s.release != nil {
		s.release()
	}
}

// Track attaches this Storage to obj such that obj's GC will Release it. It
// also Acquires once upfront, so Storage lifetime is bound to obj's.
//
// obj must be a typed pointer (e.g., *Tensor). runtime.AddCleanup requires
// a concrete pointer type, which is why Track is a free generic function and
// not a method on *Storage.
//
// Typical use: constructors of Tensor and its views.
func Track[T any](s *Storage, obj *T) {
	s.Acquire()
	runtime.AddCleanup(obj, storageReleaseCleanup, s)
}

// storageReleaseCleanup is the cleanup entry point. It must take a non-obj
// argument so that obj can be collected; that argument retains Storage until
// the cleanup runs.
func storageReleaseCleanup(s *Storage) {
	s.Release()
}
