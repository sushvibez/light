//go:build !nocuda && cgo

package storage

import (
	"runtime"
	"sync"
	"sync/atomic"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
)

// NewCUDAFromPointer installs a CUDA-backed Storage wrapping an existing
// device pointer. The release fn is called when refcount reaches zero (and
// is responsible for cudaFree).
func NewCUDAFromPointer(dev device.Device, dt dtype.DType, nelems int, bytes int, ptr uintptr, release func()) *Storage {
	return &Storage{
		device:  dev,
		dtype:   dt,
		nelems:  nelems,
		bytes:   bytes,
		cudaPtr: ptr,
		release: release,
	}
}

// NewCUDA allocates a device buffer for nelems elements of dtype dt on the
// given CUDA device. The release callback (invoked when refcount reaches
// zero) re-selects the right device and calls cudaFree.
func NewCUDA(dev device.Device, dt dtype.DType, nelems int) (*Storage, error) {
	return newCUDA(dev, dt, nelems, true)
}

// NewCUDAUninit allocates device memory without zeroing it, for op outputs
// that every kernel writes completely. On a decode step of a small model the
// cudaMemset of each fresh output was the single largest group of driver
// calls (454 of ~1100 per token on Llama-3.2-1B), each a tiny kernel of its
// own.
func NewCUDAUninit(dev device.Device, dt dtype.DType, nelems int) (*Storage, error) {
	return newCUDA(dev, dt, nelems, false)
}

func newCUDA(dev device.Device, dt dtype.DType, nelems int, zero bool) (*Storage, error) {
	if dev.Kind() != device.KindCUDA {
		panic("storage.NewCUDA: device is not CUDA")
	}
	bytes := dt.ByteSize(nelems)
	ord := dev.Ordinal()
	if err := cudart.SetDevice(ord); err != nil {
		return nil, err
	}
	ptr, class, err := cudaPool.alloc(ord, bytes)
	if err != nil {
		return nil, err
	}
	// Zero-initialize to match NewCPU semantics.
	if zero && bytes > 0 {
		if err := cudart.Memset(ptr, 0, uintptr(bytes)); err != nil {
			cudaPool.release(ord, ptr, class)
			return nil, err
		}
	}
	return NewCUDAFromPointer(dev, dt, nelems, bytes, ptr, func() {
		// Re-select the device because the cleanup may fire on a different
		// goroutine. Ignore errors — best-effort release.
		_ = cudart.SetDevice(ord)
		cudaPool.release(ord, ptr, class)
	}), nil
}

// cudaPool is a caching device allocator. cudaMalloc is slow and cudaFree
// synchronises the whole device, and inference allocates thousands of
// short-lived buffers per step, so freed blocks are kept per size class and
// reused; the cache is capped and the oldest blocks are freed beyond it.
// Reuse is safe with asynchronous kernels because everything runs on one
// stream in order: whatever still reads a block was queued before the
// kernel that will overwrite it.
var cudaPool = &cudaCachingPool{free: map[poolKey][]uintptr{}}

type poolKey struct {
	ord   int
	class int
}

type cudaCachingPool struct {
	mu     sync.Mutex
	free   map[poolKey][]uintptr
	cached int // bytes held in free lists
	order  []poolKey
	fresh  atomic.Int64 // bytes of cudaMalloc since the last forced GC
}

// PoolCapBytes is the most device memory the allocator keeps cached.
var PoolCapBytes = 2 << 30

// FreshBytesPerGC is how much device memory may be cudaMalloc'd between
// forced garbage collections (0 disables them); see cudaCachingPool.alloc.
var FreshBytesPerGC int64 = 1 << 30

// sizeClass rounds a byte count to its class: 512-byte steps below 1 MB,
// 1 MB steps above, so nearly-equal activations share blocks.
func sizeClass(bytes int) int {
	if bytes < 1<<20 {
		return (bytes + 511) &^ 511
	}
	return (bytes + (1<<20 - 1)) &^ (1<<20 - 1)
}

func (p *cudaCachingPool) alloc(ord, bytes int) (uintptr, int, error) {
	if bytes == 0 {
		return 0, 0, nil
	}
	class := sizeClass(bytes)
	key := poolKey{ord, class}
	p.mu.Lock()
	if l := p.free[key]; len(l) > 0 {
		ptr := l[len(l)-1]
		p.free[key] = l[:len(l)-1]
		p.cached -= class
		p.mu.Unlock()
		return ptr, class, nil
	}
	p.mu.Unlock()
	// Fresh device memory. Buffers come back to the pool only through GC
	// cleanups, and a long prefill allocates gigabytes of short-lived
	// activations without growing the Go heap enough to trigger one, so
	// every FreshBytesPerGC of misses forces a collection: it bounds the
	// peak at roughly the live set plus that budget (a 543-token Ministral
	// prefill held 5 GB of dead activations otherwise).
	if FreshBytesPerGC > 0 && p.fresh.Add(int64(class)) >= FreshBytesPerGC {
		p.fresh.Store(0)
		runtime.GC()
	}
	ptr, err := cudart.Malloc(uintptr(class))
	if err != nil {
		// Out of memory: drop the cache and retry once.
		p.Drain()
		ptr, err = cudart.Malloc(uintptr(class))
	}
	return ptr, class, err
}

func (p *cudaCachingPool) release(ord int, ptr uintptr, class int) {
	if ptr == 0 {
		return
	}
	p.mu.Lock()
	key := poolKey{ord, class}
	p.free[key] = append(p.free[key], ptr)
	p.order = append(p.order, key)
	p.cached += class
	var evict []uintptr
	for p.cached > PoolCapBytes && len(p.order) > 0 {
		k := p.order[0]
		p.order = p.order[1:]
		l := p.free[k]
		if len(l) == 0 {
			continue
		}
		evict = append(evict, l[0])
		p.free[k] = l[1:]
		p.cached -= k.class
	}
	p.mu.Unlock()
	for _, e := range evict {
		_ = cudart.Free(e)
	}
}

// Drain returns every cached block to the driver (used when a model is
// unloaded or an allocation fails).
func (p *cudaCachingPool) Drain() {
	p.mu.Lock()
	var all []uintptr
	for k, l := range p.free {
		all = append(all, l...)
		delete(p.free, k)
	}
	p.order = nil
	p.cached = 0
	p.mu.Unlock()
	for _, e := range all {
		_ = cudart.Free(e)
	}
}

// DrainCUDAPool frees the allocator's cached device blocks.
func DrainCUDAPool() { cudaPool.Drain() }

// CUDAPtr returns the device pointer stored alongside this Storage. Panics
// if the storage is not CUDA-backed.
func (s *Storage) CUDAPtr() uintptr {
	if s.device.Kind() != device.KindCUDA {
		panic("storage.CUDAPtr: not a CUDA storage")
	}
	return s.cudaPtr
}
