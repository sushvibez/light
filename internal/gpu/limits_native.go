//go:build !js

package gpu

import "unsafe"

// wgpuLimits mirrors WGPULimits (webgpu.h v29): u32 fields with the three
// u64 sizes in their places; the layout is what wgpuAdapterGetLimits
// fills and the device descriptor's requiredLimits points at.
type wgpuLimits struct {
	nextInChain                               uintptr
	maxTextureDimension1D                     uint32
	maxTextureDimension2D                     uint32
	maxTextureDimension3D                     uint32
	maxTextureArrayLayers                     uint32
	maxBindGroups                             uint32
	maxBindGroupsPlusVertexBuffers            uint32
	maxBindingsPerBindGroup                   uint32
	maxDynamicUniformBuffersPerPipelineLayout uint32
	maxDynamicStorageBuffersPerPipelineLayout uint32
	maxSampledTexturesPerShaderStage          uint32
	maxSamplersPerShaderStage                 uint32
	maxStorageBuffersPerShaderStage           uint32
	maxStorageTexturesPerShaderStage          uint32
	maxUniformBuffersPerShaderStage           uint32
	maxUniformBufferBindingSize               uint64
	maxStorageBufferBindingSize               uint64
	minUniformBufferOffsetAlignment           uint32
	minStorageBufferOffsetAlignment           uint32
	maxVertexBuffers                          uint32
	_                                         uint32
	maxBufferSize                             uint64
	maxVertexAttributes                       uint32
	maxVertexBufferArrayStride                uint32
	maxInterStageShaderVariables              uint32
	maxColorAttachments                       uint32
	maxColorAttachmentBytesPerSample          uint32
	maxComputeWorkgroupStorageSize            uint32
	maxComputeInvocationsPerWorkgroup         uint32
	maxComputeWorkgroupSizeX                  uint32
	maxComputeWorkgroupSizeY                  uint32
	maxComputeWorkgroupSizeZ                  uint32
	maxComputeWorkgroupsPerDimension          uint32
	maxImmediateSize                          uint32
}

const (
	limitU32Undefined = ^uint32(0)
	limitU64Undefined = ^uint64(0)
)

// Layout assertions against webgpu.h: a wrong offset fails to compile
// (a constant index out of range), in either direction.
var (
	_ = [1]struct{}{}[unsafe.Offsetof(wgpuLimits{}.maxUniformBufferBindingSize)-64]
	_ = [1]struct{}{}[unsafe.Offsetof(wgpuLimits{}.maxStorageBufferBindingSize)-72]
	_ = [1]struct{}{}[unsafe.Offsetof(wgpuLimits{}.maxBufferSize)-96]
	_ = [1]struct{}{}[unsafe.Offsetof(wgpuLimits{}.maxImmediateSize)-148]
	_ = [1]struct{}{}[unsafe.Sizeof(wgpuLimits{})-152]
)

// undefinedLimits is a WGPULimits asking for the defaults everywhere.
func undefinedLimits() wgpuLimits {
	var l wgpuLimits
	p := (*[152 / 4]uint32)(unsafe.Pointer(&l))
	for i := 2; i < len(p); i++ { // leave nextInChain nil
		p[i] = limitU32Undefined
	}
	return l
}

// requestedLimits is the limits block a device is requested with: the
// adapter's own buffer-size limits (so a model, a terrain store or a point
// cloud can use the memory the GPU actually has instead of WebGPU's 256 MB
// / 128 MB defaults), everything else left at the default.
func (a *Adapter) requestedLimits() (wgpuLimits, Limits) {
	req := undefinedLimits()
	var have wgpuLimits
	if wgpuAdapterGetLimits(a.h, &have) == 1 {
		req.maxBufferSize = have.maxBufferSize
		req.maxStorageBufferBindingSize = have.maxStorageBufferBindingSize
		return req, Limits{MaxBufferSize: int64(have.maxBufferSize), MaxStorageBindingSize: int64(have.maxStorageBufferBindingSize)}
	}
	return req, Limits{MaxBufferSize: MaxBufferSize, MaxStorageBindingSize: MaxStorageBindingSize}
}
