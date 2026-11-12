//go:build !nocuda && cgo

package storage

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"

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
	st := NewCUDAFromPointer(dev, dt, nelems, bytes, ptr, func() {
		// Re-select the device because the cleanup may fire on a different
		// goroutine. Ignore errors — best-effort release.
		_ = cudart.SetDevice(ord)
		cudaPool.release(ord, ptr, class)
	})
	if obs := allocObserver.Load(); obs != nil {
		(*obs)(st)
	}
	return st, nil
}

// allocObserver, when set, sees every CUDA storage as it is allocated. A
// decode-graph capture installs one to retain the step's transients: the
// captured graph refers to their addresses, so they must outlive it.
var allocObserver atomic.Pointer[func(*Storage)]

// SetCUDAAllocObserver installs fn (nil removes it) and returns the
// previous observer.
func SetCUDAAllocObserver(fn func(*Storage)) func(*Storage) {
	var prev *func(*Storage)
	if fn == nil {
		prev = allocObserver.Swap(nil)
	} else {
		prev = allocObserver.Swap(&fn)
	}
	if prev == nil {
		return nil
	}
	return *prev
}

func init() {
	cudart.OOMHook = func() {
		for range 2 {
			runtime.GC()
			time.Sleep(30 * time.Millisecond) // the storage cleanups run on their own goroutine
		}
		cudaPool.Drain()
	}
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
	handed atomic.Int64 // bytes handed out (cache hits too) since the last forced GC
}

// PoolCapBytes is the most device memory the allocator keeps cached; 0
// (the default) means a quarter of the device's memory, at least
// poolCapMin. Past the cap every release is a cudaFree, which waits for the
// device: a fixed 2 GB cap evicted ~860 blocks per Z-Image 1024² step on
// the L4 (each step cycles tens of GB of activations through the pool) and
// left the GPU idle about a third of the time. A cudaMalloc that fails
// drains the cache before retrying, so cached blocks never starve a load.
var PoolCapBytes = 0

const poolCapMin = 2 << 30

var autoPoolCap = sync.OnceValue(func() int {
	_, total, err := cudart.MemInfo()
	if err != nil || int(total)/4 < poolCapMin {
		return poolCapMin
	}
	return int(total) / 4
})

// poolCap is the cache cap in force.
func poolCap() int {
	if PoolCapBytes > 0 {
		return PoolCapBytes
	}
	return autoPoolCap()
}

// FreshBytesPerGC is how much device memory may be cudaMalloc'd between
// forced garbage collections (0 disables them, including the one after
// every half pool cap handed out); see cudaCachingPool.alloc.
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
	// Dead tensors come back only through GC cleanups. Collecting after every
	// half cap handed out returns them while the cache still has room for
	// them: waiting for misses instead (FreshBytesPerGC alone) lets a
	// step drain the whole cache, malloc more, then get it all back at once,
	// past the cap, as cudaFree evictions every cycle.
	if FreshBytesPerGC > 0 && p.handed.Add(int64(class)) >= int64(poolCap()/2) {
		p.handed.Store(0)
		p.fresh.Store(0)
		poolStats.forcedGCs.Add(1)
		runtime.GC()
	}
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
		p.handed.Store(0)
		poolStats.forcedGCs.Add(1)
		runtime.GC()
	}
	ptr, err := cudart.Malloc(uintptr(class))
	// Out of memory: drop the cache and retry; then collect, since the device
	// blocks of dead tensors are freed only by GC cleanups (which run on
	// their own goroutine, hence the short waits) and the GC cannot see
	// device memory pressure.
	for attempt := 0; err != nil && attempt < 4; attempt++ {
		poolStats.oomRetries.Add(1)
		if attempt > 0 {
			runtime.GC()
			time.Sleep(time.Duration(10<<attempt) * time.Millisecond)
		}
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
	limit := poolCap()
	for p.cached > limit && len(p.order) > 0 {
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
	poolStats.evictions.Add(int64(len(evict)))
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

// poolStats counts the caching allocator's slow paths (CUDAPoolStats).
var poolStats struct{ forcedGCs, oomRetries, evictions atomic.Int64 }

// CUDAPoolStats reports how often the caching allocator forced a GC after
// FreshBytesPerGC of misses, retried a failed cudaMalloc (each retry drains
// the cache; later ones also collect and wait), and evicted a cached block
// past the cap (a cudaFree each), since the process started.
func CUDAPoolStats() (forcedGCs, oomRetries, evictions int64) {
	return poolStats.forcedGCs.Load(), poolStats.oomRetries.Load(), poolStats.evictions.Load()
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
