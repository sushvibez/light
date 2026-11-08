//go:build !js

package gpu

import (
	"fmt"
	"runtime"
	"unsafe"

	"light/internal/ffi"
)

// Light's additions to the vendored gfx compute surface: bind groups over
// buffer sub-ranges, buffer clears, copies and readback at offsets, and
// adapter details. Everything else in this package is gfx's code as is.

var addr_wgpuCommandEncoderClearBuffer uintptr

// bindLight resolves the symbols only these additions use.
func bindLight(lib *ffi.Lib) {
	addr_wgpuCommandEncoderClearBuffer = lib.MustSym("wgpuCommandEncoderClearBuffer")
}

// BufferBinding binds bytes [Offset, Offset+Size) of Buffer at a binding
// index (Size 0: to the end of the buffer).
type BufferBinding struct {
	Binding      int
	Buffer       *Buffer
	Offset, Size uint64
}

// CreateBufferBindGroup creates a bind group of storage / uniform buffer
// ranges for layout.
func (d *Device) CreateBufferBindGroup(layout *BindGroupLayout, entries []BufferBinding) *BindGroup {
	raw := make([]bindGroupEntry, len(entries))
	for i, e := range entries {
		size := e.Size
		if size == 0 {
			size = uint64(e.Buffer.size) - e.Offset
		}
		raw[i] = bindGroupEntry{binding: uint32(e.Binding), buffer: e.Buffer.h, offset: e.Offset, size: size}
	}
	desc := bindGroupDescriptor{layout: layout.h, entryCount: uintptr(len(raw))}
	if len(raw) > 0 {
		desc.entries = uintptr(unsafe.Pointer(&raw[0]))
	}
	h := wgpuDeviceCreateBindGroup(d.h, &desc)
	runtime.KeepAlive(raw)
	return &BindGroup{h}
}

// Valid reports whether the buffer handle is non-nil (creation can fail).
func (b *Buffer) Valid() bool { return b != nil && b.h != 0 }

// CopyBufferToBuffer records a copy of size bytes (4-byte multiples).
func (e *CommandEncoder) CopyBufferToBuffer(src *Buffer, srcOff uint64, dst *Buffer, dstOff, size uint64) {
	wgpuCommandEncoderCopyBufferToBuffer(e.h, src.h, srcOff, dst.h, dstOff, size)
}

// ClearBuffer records a zero fill of size bytes of b at off (4-byte multiples).
func (e *CommandEncoder) ClearBuffer(b *Buffer, off, size uint64) {
	ffi.Call(addr_wgpuCommandEncoderClearBuffer, e.h, b.h, uintptr(off), uintptr(size), 0, 0)
}

// Poll processes completed GPU work; with wait it blocks until the queue is
// idle.
func (d *Device) Poll(wait bool) {
	var w uint32
	if wait {
		w = 1
	}
	wgpuDevicePoll(d.h, w, nil)
}

// ReadBufferAt copies size bytes of b from byte offset off back to the host,
// blocking until the GPU is done (b needs BufferCopySrc; off and size are
// 4-byte multiples).
func (d *Device) ReadBufferAt(q *Queue, b *Buffer, off uint64, size int) ([]byte, error) {
	staging := d.CreateBuffer(BufferMapRead|BufferCopyDst, size)
	defer staging.Release()
	enc := d.CreateCommandEncoder()
	enc.CopyBufferToBuffer(b, off, staging, 0, uint64(size))
	q.Submit(enc.Finish())

	mapResult.Lock()
	defer mapResult.Unlock()
	mapResult.fired = false
	issueBufferMapAsync(staging.h, mapModeRead, 0, uintptr(size),
		bufferMapCallbackInfo{mode: callbackModeAllowSpontaneous, callback: mapCallback()})
	wgpuDevicePoll(d.h, 1, nil)
	if !mapResult.fired || mapResult.status != 1 {
		return nil, fmt.Errorf("gpu: buffer map failed (fired=%v status=%d)", mapResult.fired, mapResult.status)
	}
	ptr := wgpuBufferGetMappedRange(staging.h, 0, uintptr(size))
	if ptr == nil {
		return nil, fmt.Errorf("gpu: GetMappedRange returned nil")
	}
	out := make([]byte, size)
	copy(out, unsafe.Slice((*byte)(ptr), size))
	wgpuBufferUnmap(staging.h)
	return out, nil
}

// Info reports the adapter's device name and backend (Vulkan, D3D12, ...).
func (a *Adapter) Info() (name, backend string) {
	var ai adapterInfo
	wgpuAdapterGetInfo(a.h, &ai)
	return goStringView(ai.device.data, ai.device.length), backendNames[ai.backendType]
}
