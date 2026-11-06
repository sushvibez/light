//go:build !nocuda && cgo

package cuda

import (
	"sync"
	"unsafe"

	"light/internal/cudart"
)

// Strided kernels (contiguous copy, broadcast binary ops) take their dims and
// strides through a small device buffer. Allocating one per call costs a
// cudaMalloc and a cudaFree (an implicit device sync), and even a plain
// cudaMemcpy upload blocks until every queued kernel has run. Each device
// therefore gets a ring of fixed slots in pinned host memory mirrored on the
// device: a call fills the next host slot and queues an async copy, which the
// default stream runs after the kernels already queued and before the kernel
// that reads it. When the ring wraps the device is synchronised once, so a
// host slot is never rewritten before its copy has executed.
const (
	metaSlotBytes = 256  // ≤ 64 int32: rank ≤ 21 for three strides
	metaSlots     = 8192 // ring depth: one sync per 8192 strided ops
)

type metaRing struct {
	dev  uintptr
	host uintptr
	next int
}

var (
	metaMu    sync.Mutex
	metaRings = map[int]*metaRing{}
)

// uploadMeta copies meta into the next ring slot of device ord and returns
// its device address. The caller must have selected the device.
func uploadMeta(ord int, meta []int32) (uintptr, error) {
	n := 4 * len(meta)
	if n > metaSlotBytes {
		// Oversized (rank > 21): a dedicated allocation the caller never
		// frees; practically unreachable.
		p, err := cudart.Malloc(uintptr(n))
		if err != nil {
			return 0, err
		}
		return p, cudart.MemcpyHtoD(p, unsafe.Slice((*byte)(unsafe.Pointer(&meta[0])), n))
	}
	metaMu.Lock()
	defer metaMu.Unlock()
	r := metaRings[ord]
	if r == nil {
		dev, err := cudart.Malloc(uintptr(metaSlotBytes * metaSlots))
		if err != nil {
			return 0, err
		}
		host, err := cudart.MallocHost(uintptr(metaSlotBytes * metaSlots))
		if err != nil {
			return 0, err
		}
		r = &metaRing{dev: dev, host: host}
		metaRings[ord] = r
	}
	if r.next == 0 {
		// About to reuse slot 0: every copy queued in the previous lap must
		// have run. Only the first lap skips this.
		if err := cudart.DeviceSynchronize(); err != nil {
			return 0, err
		}
	}
	off := uintptr(r.next * metaSlotBytes)
	r.next = (r.next + 1) % metaSlots
	hostSlot := unsafe.Slice((*int32)(unsafe.Pointer(r.host+off)), len(meta))
	copy(hostSlot, meta)
	return r.dev + off, cudart.MemcpyHtoDAsync(r.dev+off, r.host+off, uintptr(n))
}
