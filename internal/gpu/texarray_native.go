//go:build !js

package gpu

import (
	"runtime"
	"unsafe"
)

// textureViewDescriptor mirrors WGPUTextureViewDescriptor (29.x).
type textureViewDescriptor struct {
	nextInChain     uintptr
	label           stringView
	format          uint32
	dimension       uint32
	baseMipLevel    uint32
	mipLevelCount   uint32
	baseArrayLayer  uint32
	arrayLayerCount uint32
	aspect          uint32
	_               uint32
	usage           uint64
}

const viewDimension2DArray = 0x3

// CreateTexture2DArray allocates a 2D texture with `layers` array layers
// (one mip level): a texture atlas whose slots are whole images, bound
// once and indexed per draw.
func (d *Device) CreateTexture2DArray(width, height, layers int, format TextureFormat, usage TextureUsage) *Texture {
	desc := textureDescriptor{
		usage:         uint64(usage),
		dimension:     dimension2D,
		size:          extent3D{uint32(width), uint32(height), uint32(layers)},
		format:        uint32(format),
		mipLevelCount: 1,
		sampleCount:   1,
	}
	return &Texture{h: wgpuDeviceCreateTexture(d.h, &desc), bpt: BytesPerTexel(format)}
}

// CreateViewArray returns a texture_2d_array view over all layers.
func (t *Texture) CreateViewArray(format TextureFormat, layers int) *TextureView {
	desc := textureViewDescriptor{
		format:          uint32(format),
		dimension:       viewDimension2DArray,
		mipLevelCount:   1,
		arrayLayerCount: uint32(layers),
		aspect:          aspectAll,
	}
	return &TextureView{wgpuTextureCreateView(t.h, unsafe.Pointer(&desc))}
}

// WriteTextureLayer uploads tightly packed RGBA8 pixels into one array
// layer of t.
func (q *Queue) WriteTextureLayer(t *Texture, layer, width, height int, data []byte) {
	q.WriteTextureRegion(t, layer, 0, 0, width, height, data)
}

// WriteTextureRegion uploads a width×height RGBA8 block at (x, y) of one
// array layer (several small images share a layer).
func (q *Queue) WriteTextureRegion(t *Texture, layer, x, y, width, height int, data []byte) {
	if len(data) == 0 {
		return
	}
	dst := imageCopyTexture{texture: t.h, aspect: aspectAll, origin: origin3D{uint32(x), uint32(y), uint32(layer)}}
	layout := textureDataLayout{bytesPerRow: uint32(width * 4), rowsPerImage: uint32(height)}
	ext := extent3D{uint32(width), uint32(height), 1}
	wgpuQueueWriteTexture(q.h, &dst, unsafe.Pointer(&data[0]), uintptr(len(data)), &layout, &ext)
	runtime.KeepAlive(data)
}
