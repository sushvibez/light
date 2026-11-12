//go:build !nocuda && cgo

package cuda

import (
	"strings"
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
	if p, ok := metaCached(ord, meta); ok {
		return p, nil
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

// Persistent meta: the same dims / strides recur every step (one decode
// token through the same model), so each distinct meta gets a fixed
// device slot it keeps for the life of the process, uploaded once from a
// pinned host copy that is never rewritten. That makes the upload
// capturable into a decode graph (a replay re-copies the same bytes) and
// removes the per-call HtoD from steady-state decode. metaCacheMax bounds
// the slots per device; beyond it the ring above serves.
const (
	metaSlabSlots = 1024
	metaCacheMax  = 64 * 1024
)

type metaCache struct {
	slots    map[string]uintptr
	devSlab  uintptr
	hostSlab uintptr
	used     int // slots used in the current slab
	total    int
}

var metaCaches = map[int]*metaCache{}

func metaCached(ord int, meta []int32) (uintptr, bool) {
	n := 4 * len(meta)
	if n > metaSlotBytes {
		return 0, false
	}
	key := string(unsafe.Slice((*byte)(unsafe.Pointer(&meta[0])), n))
	metaMu.Lock()
	defer metaMu.Unlock()
	c := metaCaches[ord]
	if c == nil {
		c = &metaCache{slots: map[string]uintptr{}}
		metaCaches[ord] = c
	}
	if p, ok := c.slots[key]; ok {
		return p, true
	}
	if c.total >= metaCacheMax {
		return 0, false
	}
	if c.used == metaSlabSlots || c.devSlab == 0 {
		dev, err := cudart.Malloc(uintptr(metaSlotBytes * metaSlabSlots))
		if err != nil {
			return 0, false
		}
		host, err := cudart.MallocHost(uintptr(metaSlotBytes * metaSlabSlots))
		if err != nil {
			return 0, false
		}
		c.devSlab, c.hostSlab, c.used = dev, host, 0
	}
	off := uintptr(c.used * metaSlotBytes)
	hostSlot := unsafe.Slice((*int32)(unsafe.Pointer(c.hostSlab+off)), len(meta))
	copy(hostSlot, meta)
	if err := cudart.MemcpyHtoDAsync(c.devSlab+off, c.hostSlab+off, uintptr(n)); err != nil {
		return 0, false
	}
	c.used++
	c.total++
	c.slots[strings.Clone(key)] = c.devSlab + off
	return c.devSlab + off, true
}
