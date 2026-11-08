//go:build !js

package gpu

import (
	"runtime"
	"unsafe"
)

// Mipmapped textures and anisotropic sampling — what a 3D material system
// needs to stop minified albedo/normal maps from aliasing into noise
// (DESIGN.md §9). The mip chain itself is built on the CPU by the caller
// (render.NewImage3DMipmapped) and uploaded level by level, which keeps the
// wasm backend byte-identical to the native one.

// MipLevels returns the number of mip levels in a full chain for a
// width x height texture (at least 1).
func MipLevels(width, height int) int {
	n := 1
	for width > 1 || height > 1 {
		width, height = max(width/2, 1), max(height/2, 1)
		n++
	}
	return n
}

// CreateTexture2DMips allocates a 2D texture with a mip chain of mips
// levels (use MipLevels for a full chain). Levels are uploaded with
// WriteTextureMip.
func (d *Device) CreateTexture2DMips(width, height int, format TextureFormat, usage TextureUsage, mips int) *Texture {
	if mips < 1 {
		mips = 1
	}
	desc := textureDescriptor{
		usage:         uint64(usage),
		dimension:     dimension2D,
		size:          extent3D{uint32(width), uint32(height), 1},
		format:        uint32(format),
		mipLevelCount: uint32(mips),
		sampleCount:   1,
	}
	return &Texture{h: wgpuDeviceCreateTexture(d.h, &desc), bpt: BytesPerTexel(format)}
}

// WriteTextureMip uploads tightly packed RGBA8 pixels into one mip level.
func (q *Queue) WriteTextureMip(t *Texture, mip, width, height int, data []byte) {
	if len(data) == 0 || width <= 0 || height <= 0 {
		return
	}
	noteTextureUpload(len(data))
	defer uploadTimer(true)()
	dst := imageCopyTexture{texture: t.h, mipLevel: uint32(mip), aspect: aspectAll}
	layout := textureDataLayout{bytesPerRow: uint32(width * 4), rowsPerImage: uint32(height)}
	ext := extent3D{uint32(width), uint32(height), 1}
	wgpuQueueWriteTexture(q.h, &dst, unsafe.Pointer(&data[0]), uintptr(len(data)), &layout, &ext)
	runtime.KeepAlive(data)
}

// CreateSamplerMip returns a trilinear sampler (linear mag/min plus linear
// mipmap filtering) with the requested maximum anisotropy, clamped to the
// 1..16 range WebGPU allows. repeat selects tiling vs clamped addressing.
func (d *Device) CreateSamplerMip(repeat bool, maxAnisotropy int) *Sampler {
	mode := uint32(addressClampToEdge)
	if repeat {
		mode = addressRepeat
	}
	if maxAnisotropy < 1 {
		maxAnisotropy = 1
	}
	if maxAnisotropy > 16 {
		maxAnisotropy = 16
	}
	desc := samplerDescriptor{
		addressModeU:  mode,
		addressModeV:  mode,
		addressModeW:  mode,
		magFilter:     filterLinear,
		minFilter:     filterLinear,
		mipmapFilter:  filterLinear,
		lodMaxClamp:   32,
		maxAnisotropy: uint16(maxAnisotropy),
	}
	return &Sampler{wgpuDeviceCreateSampler(d.h, &desc)}
}
