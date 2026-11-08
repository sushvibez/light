//go:build !js

package gpu

import (
	"runtime"
	"unsafe"
)

// Samplers, bind groups and texture uploads — the gpu surface area for
// textured 2D (glyph atlases, images).

// ---- raw struct mirrors (verified against webgpu.h v0.19) ----

type samplerDescriptor struct {
	nextInChain   uintptr
	label         stringView
	addressModeU  uint32
	addressModeV  uint32
	addressModeW  uint32
	magFilter     uint32
	minFilter     uint32
	mipmapFilter  uint32
	lodMinClamp   float32
	lodMaxClamp   float32
	compare       uint32
	maxAnisotropy uint16
	_             uint16
}

type bindGroupDescriptor struct {
	nextInChain uintptr
	label       stringView
	layout      uintptr
	entryCount  uintptr
	entries     uintptr
}

type bindGroupEntry struct {
	nextInChain uintptr
	binding     uint32
	_           uint32
	buffer      uintptr
	offset      uint64
	size        uint64
	sampler     uintptr
	textureView uintptr
}

const (
	addressClampToEdge = 0x1
	addressRepeat      = 0x2
	filterLinear       = 0x2
)

// TextureUsage is a WGPUTextureUsage flag set (uint64 in 29.x).
type TextureUsage uint64

const (
	TextureCopySrc          TextureUsage = 0x01
	TextureCopyDst          TextureUsage = 0x02
	TextureBinding          TextureUsage = 0x04
	TextureRenderAttachment TextureUsage = 0x10
)

// ---- public API ----

// CreateTexture2D allocates a 2D texture with explicit usage flags.
func (d *Device) CreateTexture2D(width, height int, format TextureFormat, usage TextureUsage) *Texture {
	desc := textureDescriptor{
		usage:         uint64(usage),
		dimension:     dimension2D,
		size:          extent3D{uint32(width), uint32(height), 1},
		format:        uint32(format),
		mipLevelCount: 1,
		sampleCount:   1,
	}
	return &Texture{h: wgpuDeviceCreateTexture(d.h, &desc), bpt: BytesPerTexel(format)}
}

// WriteTexture uploads tightly packed RGBA8 pixels (width*height*4 bytes)
// into t.
func (q *Queue) WriteTexture(t *Texture, width, height int, data []byte) {
	q.WriteTextureRows(t, width, 0, height, data)
}

// WriteTextureRows uploads a horizontal BAND of a texture: rows [y, y+height)
// at full width, from data holding exactly those rows. A glyph atlas grows a
// few rows at a time, and re-uploading the whole 1024x1024 sheet for each
// new glyph cost more than everything else on a text-heavy animated page put
// together (640ms a frame on the shapes page).
func (q *Queue) WriteTextureRows(t *Texture, width, y, height int, data []byte) {
	if len(data) == 0 || height <= 0 {
		return
	}
	noteTextureUpload(len(data))
	defer uploadTimer(true)()
	// Upload in bands of at most writeChunk bytes. wgpu-native v29 gives a
	// staging write above ~1 MB a dedicated allocation, and a burst of them
	// in one submission (a level's textures at load) leaves the device's
	// staging path ~100× slower for every later write, permanently — see
	// TestUploadStaysFastAfterTextureBurst. Banding avoids the dedicated
	// path altogether; noteStaging flushes the queue every few MB besides.
	rowBytes := width * max(1, t.bpt)
	rowsPer := max(1, writeChunk/max(1, rowBytes))
	for r := 0; r < height; r += rowsPer {
		n := min(rowsPer, height-r)
		dst := imageCopyTexture{texture: t.h, aspect: aspectAll, origin: origin3D{0, uint32(y + r), 0}}
		layout := textureDataLayout{bytesPerRow: uint32(rowBytes), rowsPerImage: uint32(n)}
		ext := extent3D{uint32(width), uint32(n), 1}
		wgpuQueueWriteTexture(q.h, &dst, unsafe.Pointer(&data[r*rowBytes]), uintptr(n*rowBytes), &layout, &ext)
		q.noteStaging(n * rowBytes)
	}
	runtime.KeepAlive(data)
}

// Sampler describes texture filtering and addressing.
type Sampler struct{ h uintptr }

func (s *Sampler) Release() { wgpuSamplerRelease(s.h) }

// CreateSampler returns a clamp-to-edge sampler, linear- or
// nearest-filtered.
func (d *Device) CreateSampler(linear bool) *Sampler {
	desc := samplerDescriptor{
		addressModeU:  addressClampToEdge,
		addressModeV:  addressClampToEdge,
		addressModeW:  addressClampToEdge,
		lodMaxClamp:   32,
		maxAnisotropy: 1,
	}
	if linear {
		desc.magFilter, desc.minFilter = filterLinear, filterLinear
	}
	return &Sampler{wgpuDeviceCreateSampler(d.h, &desc)}
}

// CreateSamplerRepeat is CreateSampler with wrapping (tiling) addressing —
// what world-space tiled UVs (and glTF's default wrap mode) want.
func (d *Device) CreateSamplerRepeat(linear bool) *Sampler {
	desc := samplerDescriptor{
		addressModeU:  addressRepeat,
		addressModeV:  addressRepeat,
		addressModeW:  addressRepeat,
		lodMaxClamp:   32,
		maxAnisotropy: 1,
	}
	if linear {
		desc.magFilter, desc.minFilter = filterLinear, filterLinear
	}
	return &Sampler{wgpuDeviceCreateSampler(d.h, &desc)}
}

// BindGroupLayout is a pipeline-derived layout ("auto" layout group).
type BindGroupLayout struct{ h uintptr }

func (l *BindGroupLayout) Release() { wgpuBindGroupLayoutRelease(l.h) }

// BindGroupLayout returns the pipeline's automatically derived layout for a
// group index.
func (p *RenderPipeline) BindGroupLayout(index int) *BindGroupLayout {
	return &BindGroupLayout{wgpuRenderPipelineGetBindGroupLayout(p.h, uint32(index))}
}

// BindGroupEntry binds one resource; set exactly one of Sampler,
// TextureView or Buffer.
type BindGroupEntry struct {
	Binding     int
	Sampler     *Sampler
	TextureView *TextureView
	Buffer      *Buffer
}

// BindGroup is a bound resource set.
type BindGroup struct{ h uintptr }

func (g *BindGroup) Release() { wgpuBindGroupRelease(g.h) }

func (d *Device) CreateBindGroup(layout *BindGroupLayout, entries ...BindGroupEntry) *BindGroup {
	raw := make([]bindGroupEntry, len(entries))
	for i, e := range entries {
		raw[i].binding = uint32(e.Binding)
		switch {
		case e.Sampler != nil:
			raw[i].sampler = e.Sampler.h
		case e.TextureView != nil:
			raw[i].textureView = e.TextureView.h
		case e.Buffer != nil:
			raw[i].buffer = e.Buffer.h
			raw[i].size = uint64(e.Buffer.size)
		}
	}
	desc := bindGroupDescriptor{
		layout:     layout.h,
		entryCount: uintptr(len(raw)),
	}
	if len(raw) > 0 {
		desc.entries = uintptr(unsafe.Pointer(&raw[0]))
	}
	h := wgpuDeviceCreateBindGroup(d.h, &desc)
	runtime.KeepAlive(raw)
	return &BindGroup{h}
}

// SetBindGroup binds g at a group index for subsequent draws.
func (r *RenderPass) SetBindGroup(index int, g *BindGroup) {
	callBind.Add(1)
	wgpuRenderPassEncoderSetBindGroup(r.h, uint32(index), g.h, 0, nil)
}
