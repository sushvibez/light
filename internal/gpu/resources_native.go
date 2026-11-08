//go:build !js

package gpu

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"light/internal/ffi"
)

// Buffers, vertex input, blending, offscreen textures and CPU readback —
// the gpu surface area phase 1's 2D renderer and headless screenshots need.

// ---- raw struct mirrors (verified against webgpu.h v0.19) ----

type bufferDescriptor struct {
	nextInChain      uintptr
	label            stringView
	usage            uint64
	size             uint64
	mappedAtCreation uint32
	_                uint32
}

type extent3D struct {
	width              uint32
	height             uint32
	depthOrArrayLayers uint32
}

type textureDescriptor struct {
	nextInChain     uintptr
	label           stringView
	usage           uint64
	dimension       uint32
	size            extent3D
	format          uint32
	mipLevelCount   uint32
	sampleCount     uint32
	viewFormatCount uintptr
	viewFormats     uintptr
}

type origin3D struct {
	x, y, z uint32
}

// imageCopyTexture mirrors WGPUTexelCopyTextureInfo (no nextInChain in 29.x).
type imageCopyTexture struct {
	texture  uintptr
	mipLevel uint32
	origin   origin3D
	aspect   uint32
	_        uint32
}

// textureDataLayout mirrors WGPUTexelCopyBufferLayout (no nextInChain).
type textureDataLayout struct {
	offset       uint64
	bytesPerRow  uint32
	rowsPerImage uint32
}

// imageCopyBuffer mirrors WGPUTexelCopyBufferInfo (no nextInChain).
type imageCopyBuffer struct {
	layout textureDataLayout
	buffer uintptr
}

// bufferMapCallbackInfo mirrors WGPUBufferMapCallbackInfo (40 bytes).
type bufferMapCallbackInfo struct {
	nextInChain uintptr
	mode        uint32
	_           uint32
	callback    uintptr
	userdata1   uintptr
	userdata2   uintptr
}

type vertexAttribute struct {
	nextInChain    uintptr // added in 29.x
	format         uint32
	_              uint32
	offset         uint64
	shaderLocation uint32
	_              uint32
}

type vertexBufferLayout struct {
	nextInChain    uintptr // added in 29.x; stepMode now precedes arrayStride
	stepMode       uint32
	_              uint32
	arrayStride    uint64
	attributeCount uintptr
	attributes     uintptr
}

type blendComponent struct {
	operation uint32
	srcFactor uint32
	dstFactor uint32
}

type blendState struct {
	color blendComponent
	alpha blendComponent
}

// ---- public API ----

// BufferUsage is a WGPUBufferUsage flag set (uint64 in 29.x).
type BufferUsage uint64

const (
	BufferMapRead BufferUsage = 0x01
	BufferCopySrc BufferUsage = 0x04
	BufferCopyDst BufferUsage = 0x08
	BufferIndex   BufferUsage = 0x10
	BufferVertex  BufferUsage = 0x20
	BufferUniform BufferUsage = 0x40
	BufferStorage BufferUsage = 0x80
)

// VertexFormat names a vertex attribute layout (29.x enum values).
type VertexFormat uint32

const (
	// Uint16x4 is the skinning joint-index format (four bone indices per
	// vertex, read as vec4<u32> in WGSL).
	VertexUint16x4  VertexFormat = 0x0F
	VertexFloat32x2 VertexFormat = 0x1D
	VertexFloat32x3 VertexFormat = 0x1E
	VertexFloat32x4 VertexFormat = 0x1F
)

// FormatRGBA8Unorm is the offscreen/readback-friendly texture format.
const FormatRGBA8Unorm TextureFormat = 0x16

// FormatBGRA8Unorm is the format most native surfaces prefer; a readback
// of it is blue-first (see ReadTexture callers that write PNGs).
const FormatBGRA8Unorm TextureFormat = 0x1B

// FormatR8Unorm is one byte a texel: coverage masks, and nothing else
// so far. A glyph sheet holds an alpha channel and three bytes of white
// that every shader throws away.
const FormatR8Unorm TextureFormat = 0x01

// FormatRGBA16Float is a half-float colour target: an HDR accumulation
// buffer (additive point clouds, bloom) that a later pass tone-maps.
const FormatRGBA16Float TextureFormat = 0x28 // WGPUTextureFormat_RGBA16Float

const (
	dimension2D       = 0x2
	mapModeRead       = 0x1
	aspectAll         = 0x1
	blendOpAdd        = 0x1
	blendFactorOne    = 0x2
	blendSrcAlpha     = 0x5
	blendOneMinusSrcA = 0x6
	usageCopySrc      = 0x1

	// Dual-source blend factors (29.x), for subpixel/LCD text: the fragment's
	// second output (src1) is a per-channel coverage mask.
	blendSrc1              = 0x0E
	blendOneMinusSrc1      = 0x0F
	blendSrc1Alpha         = 0x10
	blendOneMinusSrc1Alpha = 0x11
)

// BlendFactor is a WGPUBlendFactor: what a colour-target blend multiplies
// the source or destination by.
type BlendFactor uint32

const (
	BlendZero             BlendFactor = 0x1
	BlendOne              BlendFactor = 0x2
	BlendSrc              BlendFactor = 0x3
	BlendOneMinusSrc      BlendFactor = 0x4
	BlendSrcAlpha         BlendFactor = 0x5
	BlendOneMinusSrcAlpha BlendFactor = 0x6
	BlendDst              BlendFactor = 0x7
	BlendOneMinusDst      BlendFactor = 0x8
	BlendDstAlpha         BlendFactor = 0x9
	BlendOneMinusDstAlpha BlendFactor = 0xA
)

// BlendOp is a WGPUBlendOperation: how the two weighted terms combine.
type BlendOp uint32

const (
	BlendAdd         BlendOp = 0x1
	BlendSubtract    BlendOp = 0x2
	BlendRevSubtract BlendOp = 0x3
	BlendMin         BlendOp = 0x4
	BlendMax         BlendOp = 0x5
)

// BlendComponent is one channel group's blend equation.
type BlendComponent struct {
	Op       BlendOp
	Src, Dst BlendFactor
}

// BlendDesc is a colour target's full blend state. Pass it on a pipeline
// to get an equation other than the standard alpha blend — the separable
// blend modes (multiply, screen, darken, lighten) are exactly this.
type BlendDesc struct {
	Color, Alpha BlendComponent
}

// Buffer is GPU-resident memory.
type Buffer struct {
	h    uintptr
	size int
}

func (b *Buffer) Release() { wgpuBufferRelease(b.h) }

// Size returns the buffer's capacity in bytes.
func (b *Buffer) Size() int { return b.size }

// CreateBuffer allocates a GPU buffer.
func (d *Device) CreateBuffer(usage BufferUsage, size int) *Buffer {
	if limit := d.limits.MaxBufferSize; limit > 0 && int64(size) > limit {
		// Say so here, once, with the call site — that is the whole
		// difference between a five-minute fix and an afternoon.
		buf := make([]byte, 4096)
		n := runtime.Stack(buf, false)
		fmt.Fprintf(os.Stderr,
			"gpu: buffer of %d bytes (%.0f MB) exceeds the device's %d MB maximum; "+
				"this will be an invalid buffer and the next draw that binds it will abort\n%s\n",
			size, float64(size)/(1<<20), limit>>20, buf[:n])
	}
	desc := bufferDescriptor{usage: uint64(usage), size: uint64(size)}
	return &Buffer{h: wgpuDeviceCreateBuffer(d.h, &desc), size: size}
}

// WriteBuffer schedules a copy of data into b at offset.
func (q *Queue) WriteBuffer(b *Buffer, offset int, data []byte) {
	if len(data) == 0 {
		return
	}
	noteBufferUpload(len(data))
	defer uploadTimer(false)()
	// wgpu-native serves writes up to ~1 MB from its staging belt (≈0.4 ms
	// per MB measured); anything larger gets a dedicated host-visible
	// allocation per call and costs 25–200 ms. Split big writes so a 20 MB
	// skinned-mesh upload stays on the fast path.
	for off := 0; off < len(data); off += writeChunk {
		end := min(off+writeChunk, len(data))
		t0 := time.Now()
		wgpuQueueWriteBuffer(q.h, b.h, uint64(offset+off), unsafe.Pointer(&data[off]), uintptr(end-off))
		q.noteStaging(end - off)
		if traceUploads {
			if d := time.Since(t0); d > 2*time.Millisecond {
				fmt.Fprintf(os.Stderr, "gpu: slow WriteBuffer buf=%x size=%d KB off=%d took %v\n", b.h, (end-off)/1024, offset+off, d)
			}
		}
	}
	runtime.KeepAlive(data)
}

// writeChunk is the largest single staging write we issue (buffer or
// texture band).
const writeChunk = 1 << 20

// stagingFlush is how much staging data may accumulate between submits
// before the queue flushes itself with an empty command buffer, which lets
// wgpu recycle its staging chunks instead of growing the belt.
const stagingFlush = 16 << 20

// noteStaging accounts a staging write and flushes the queue when the
// pending volume is large (load-time texture bursts).
func (q *Queue) noteStaging(n int) {
	if q.pending == nil {
		return
	}
	*q.pending += n
	if *q.pending >= stagingFlush {
		q.Flush()
	}
}

// Flush submits an empty command buffer so pending writes are handed to the
// GPU now. Submit resets the pending-staging count too.
func (q *Queue) Flush() {
	if q.pending != nil {
		*q.pending = 0
	}
	wgpuQueueSubmit(q.h, 0, nil)
}

// traceUploads (GFX_UPLOAD_TRACE=1) logs individual slow buffer writes.
var traceUploads = os.Getenv("GFX_UPLOAD_TRACE") != ""

// SetVertexBuffer binds b to a vertex buffer slot for subsequent draws.
func (r *RenderPass) SetVertexBuffer(slot int, b *Buffer) {
	callVBuf.Add(1)
	wgpuRenderPassEncoderSetVertexBuffer(r.h, uint32(slot), b.h, 0, uint64(b.size))
}

// SetScissor restricts subsequent draws to a pixel rectangle, which must
// lie within the render target.
func (r *RenderPass) SetScissor(x, y, w, h int) {
	callScissor.Add(1)
	wgpuRenderPassEncoderSetScissorRect(r.h, uint32(x), uint32(y), uint32(w), uint32(h))
}

// SetViewport maps clip space onto a pixel rectangle of the render
// target, so one pass can draw several views into one image — the split
// screen. Pair it with SetScissor: the viewport transforms, and only the
// scissor actually clips.
func (r *RenderPass) SetViewport(x, y, w, h float32) {
	setViewport(r.h, x, y, w, h, 0, 1)
}

// CreateRenderTexture allocates a 2D texture usable as a render target and
// as a readback source (headless rendering, DESIGN.md §8.1).
func (d *Device) CreateRenderTexture(width, height int, format TextureFormat) *Texture {
	desc := textureDescriptor{
		usage:         usageRenderAttachment | usageCopySrc,
		dimension:     dimension2D,
		size:          extent3D{uint32(width), uint32(height), 1},
		format:        uint32(format),
		mipLevelCount: 1,
		sampleCount:   1,
	}
	return &Texture{h: wgpuDeviceCreateTexture(d.h, &desc), bpt: BytesPerTexel(format)}
}

// CreateMSAATexture allocates a multisampled color target for anti-aliased
// rendering; resolve it into a 1-sample texture in the render pass. It is a
// transient attachment (not readable/copyable).
func (d *Device) CreateMSAATexture(width, height int, format TextureFormat, samples int) *Texture {
	if samples < 1 {
		samples = 1
	}
	desc := textureDescriptor{
		usage:         usageRenderAttachment,
		dimension:     dimension2D,
		size:          extent3D{uint32(width), uint32(height), 1},
		format:        uint32(format),
		mipLevelCount: 1,
		sampleCount:   uint32(samples),
	}
	return &Texture{h: wgpuDeviceCreateTexture(d.h, &desc), bpt: BytesPerTexel(format)}
}

// mapCallback delivers wgpuBufferMapAsync's status; requests are serialized
// by mapResult's lock.
var mapResult struct {
	sync.Mutex
	status uint32
	fired  bool
}

// mapCallback receives (status, WGPUStringView message, ud1, ud2); the
// StringView is two scalar args. Map success status is 1 in the 29.x ABI.
var mapCallback = sync.OnceValue(func() uintptr {
	return ffi.Callback(func(status, _, _, _, _ uintptr) uintptr {
		mapResult.status = uint32(status)
		mapResult.fired = true
		return 0
	})
})

// ReadTexture copies a rendered texture back to the CPU and returns tightly
// packed rows of its own texels: width*height*BytesPerTexel bytes (4 for
// RGBA8/BGRA8, 1 for R8, 8 half floats for RGBA16Float). It blocks until the GPU is done —
// a tool for tests, screenshots and goldens, not for per-frame use.
func (d *Device) ReadTexture(q *Queue, t *Texture, width, height int) ([]byte, error) {
	const rowAlign = 256 // WebGPU requires 256-byte-aligned rows for copies
	bpt := max(1, t.bpt)
	row := width * bpt
	paddedRow := (row + rowAlign - 1) / rowAlign * rowAlign
	size := paddedRow * height

	staging := d.CreateBuffer(BufferMapRead|BufferCopyDst, size)
	defer staging.Release()

	enc := d.CreateCommandEncoder()
	src := imageCopyTexture{texture: t.h, aspect: aspectAll}
	dst := imageCopyBuffer{
		layout: textureDataLayout{bytesPerRow: uint32(paddedRow), rowsPerImage: uint32(height)},
		buffer: staging.h,
	}
	ext := extent3D{uint32(width), uint32(height), 1}
	wgpuCommandEncoderCopyTextureToBuffer(enc.h, &src, &dst, &ext)
	q.Submit(enc.Finish())

	mapResult.Lock()
	defer mapResult.Unlock()
	mapResult.fired = false
	issueBufferMapAsync(staging.h, mapModeRead, 0, uintptr(size),
		bufferMapCallbackInfo{mode: callbackModeAllowSpontaneous, callback: mapCallback()})
	wgpuDevicePoll(d.h, 1, nil)                    // wait=true: blocks until queue is empty
	if !mapResult.fired || mapResult.status != 1 { // 1 = MapAsyncStatus_Success
		return nil, fmt.Errorf("gpu: buffer map failed (fired=%v status=%d)", mapResult.fired, mapResult.status)
	}
	ptr := wgpuBufferGetMappedRange(staging.h, 0, uintptr(size))
	if ptr == nil {
		return nil, fmt.Errorf("gpu: GetMappedRange returned nil")
	}
	mapped := unsafe.Slice((*byte)(ptr), size)
	out := make([]byte, row*height)
	for y := range height {
		copy(out[y*row:(y+1)*row], mapped[y*paddedRow:y*paddedRow+row])
	}
	wgpuBufferUnmap(staging.h)
	return out, nil
}

// Tick lets the device process completed work without blocking: it retires
// finished submissions and recycles their staging memory (the buffers
// WriteBuffer allocates per call). Call once per frame in loops that upload
// a lot; without it wgpu can only reclaim staging inside Submit, and a frame
// with dozens of large WriteBuffers keeps allocating fresh host-visible
// memory.
func (d *Device) Tick() { wgpuDevicePoll(d.h, 0, nil) }

// WaitIdle blocks until every submitted command buffer has finished.
func (d *Device) WaitIdle() { wgpuDevicePoll(d.h, 1, nil) }
