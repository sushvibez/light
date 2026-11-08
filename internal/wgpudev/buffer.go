//go:build !nowebgpu

package wgpudev

import (
	"fmt"
	"sync"

	"light/internal/gpu"
)

// Buffer is a pooled storage buffer. Size is the caller-visible byte length;
// the underlying wgpu buffer may be larger (size-class rounding).
type Buffer struct {
	Buf   *gpu.Buffer
	Size  uint64 // requested bytes (rounded up to a 4-byte multiple)
	class uint64 // allocated bytes (pool size class)
	dev   *Device
}

// Raw returns the wgpu buffer for binding.
func (b *Buffer) Raw() *gpu.Buffer { return b.Buf }

// Alloc returns a storage buffer of at least n bytes. When zero is set the
// contents are guaranteed to be zero (fresh buffers are zero by spec; pool
// hits are cleared on the queue).
func (d *Device) Alloc(n uint64, zero bool) (*Buffer, error) {
	size := (n + 3) &^ 3
	if size == 0 {
		size = 4
	}
	if err := d.CheckSize(size); err != nil {
		return nil, err
	}
	return d.pool.get(size, zero)
}

// Free returns the buffer to the pool (or releases it when the pool bucket
// is full). The buffer must not be used afterwards.
func (d *Device) Free(b *Buffer) {
	if b == nil || b.Buf == nil {
		return
	}
	d.pool.put(b)
}

// Pool bookkeeping — size-class free lists.
//
// Buffers up to 1 MiB round to the next power of two (min 256 B); larger
// ones round to a 1 MiB multiple, so a 1.4 GB weight does not reserve 2 GB.
// The pool is bounded per bucket and in total so VRAM cannot grow without
// limit; evicted buffers are Release()d (never Destroy()ed: wgpu refcounts
// keep a buffer alive until every queued command that references it has
// executed, so a GC-triggered free of an in-flight intermediate is safe).
type pool struct {
	mu         sync.Mutex
	dev        *Device
	buckets    map[uint64][]*Buffer
	pooledByte uint64
	liveByte   uint64 // bytes of device buffers created and not yet released
}

const (
	maxPooledPerBucket = 16
	maxPooledBytes     = 1 << 30 // 1 GiB of idle buffers at most
	smallClassCap      = 1 << 20
	largeClassStep     = 1 << 20
)

func newPool(d *Device) *pool {
	return &pool{dev: d, buckets: make(map[uint64][]*Buffer)}
}

func sizeClass(n uint64) uint64 {
	if n <= 256 {
		return 256
	}
	if n <= smallClassCap {
		v := n - 1
		v |= v >> 1
		v |= v >> 2
		v |= v >> 4
		v |= v >> 8
		v |= v >> 16
		v |= v >> 32
		return v + 1
	}
	return (n + largeClassStep - 1) / largeClassStep * largeClassStep
}

func (p *pool) get(size uint64, zero bool) (*Buffer, error) {
	cls := sizeClass(size)
	if cls > p.dev.MaxBufferSize && p.dev.MaxBufferSize > 0 {
		cls = size // never round past the device cap
	}
	p.mu.Lock()
	list := p.buckets[cls]
	if n := len(list); n > 0 {
		b := list[n-1]
		p.buckets[cls] = list[:n-1]
		p.pooledByte -= cls
		p.mu.Unlock()
		b.Size = size
		if zero {
			if err := p.dev.Clear(b.Buf, 0, size); err != nil {
				return nil, err
			}
		}
		return b, nil
	}
	p.mu.Unlock()
	raw := p.dev.Device.CreateBuffer(gpu.BufferStorage|gpu.BufferCopySrc|gpu.BufferCopyDst, int(cls))
	if !raw.Valid() {
		return nil, fmt.Errorf("wgpudev: create buffer (%d bytes) failed", cls)
	}
	p.mu.Lock()
	p.liveByte += cls
	p.mu.Unlock()
	return &Buffer{Buf: raw, Size: size, class: cls, dev: p.dev}, nil
}

func (p *pool) put(b *Buffer) {
	p.mu.Lock()
	list := p.buckets[b.class]
	if len(list) < maxPooledPerBucket && p.pooledByte+b.class <= maxPooledBytes {
		p.buckets[b.class] = append(list, b)
		p.pooledByte += b.class
		p.mu.Unlock()
		return
	}
	p.liveByte -= b.class
	p.mu.Unlock()
	b.Buf.Release()
	b.Buf = nil
}

// PoolStats reports idle pooled buffers per size class (for tests).
func (d *Device) PoolStats() map[uint64]int {
	d.pool.mu.Lock()
	defer d.pool.mu.Unlock()
	out := make(map[uint64]int)
	for cls, l := range d.pool.buckets {
		if len(l) > 0 {
			out[cls] = len(l)
		}
	}
	return out
}

// PoolClear releases every idle pooled buffer.
func (d *Device) PoolClear() {
	d.pool.mu.Lock()
	buckets := d.pool.buckets
	d.pool.buckets = make(map[uint64][]*Buffer)
	d.pool.liveByte -= d.pool.pooledByte
	d.pool.pooledByte = 0
	d.pool.mu.Unlock()
	for _, l := range buckets {
		for _, b := range l {
			b.Buf.Release()
			b.Buf = nil
		}
	}
}

// AllocatedBytes reports the device memory held by storage buffers (in use
// or pooled for reuse).
func (d *Device) AllocatedBytes() uint64 {
	d.pool.mu.Lock()
	defer d.pool.mu.Unlock()
	return d.pool.liveByte
}
