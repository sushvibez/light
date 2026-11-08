//go:build !js

package gpu

import (
	"runtime"
	"unsafe"
)

// Depth testing and indexed drawing — the gpu surface 3D rendering needs.

// FormatDepth32Float is the depth texture format 3D passes use.
const FormatDepth32Float TextureFormat = 0x30

const (
	compareLess    = 0x2
	cullBack       = 0x3
	indexUint32    = 0x2
	compareAlways  = 0x8
	stencilOpKeep  = 0x1
	storeOpDiscard = 0x2
)

type stencilFaceState struct {
	compare     uint32
	failOp      uint32
	depthFailOp uint32
	passOp      uint32
}

type depthStencilState struct {
	nextInChain         uintptr
	format              uint32
	depthWriteEnabled   uint32
	depthCompare        uint32
	stencilFront        stencilFaceState
	stencilBack         stencilFaceState
	stencilReadMask     uint32
	stencilWriteMask    uint32
	depthBias           int32
	depthBiasSlopeScale float32
	depthBiasClamp      float32
}

type renderPassDepthStencilAttachment struct {
	nextInChain       uintptr // added in 29.x
	view              uintptr
	depthLoadOp       uint32
	depthStoreOp      uint32
	depthClearValue   float32
	depthReadOnly     uint32
	stencilLoadOp     uint32
	stencilStoreOp    uint32
	stencilClearValue uint32
	stencilReadOnly   uint32
}

// CreateDepthTexture allocates a Depth32Float render attachment.
func (d *Device) CreateDepthTexture(width, height int) *Texture {
	return d.CreateTexture2D(width, height, FormatDepth32Float, TextureRenderAttachment)
}

// CreateShadowMap allocates a depth texture that shaders can sample
// (shadow mapping).
func (d *Device) CreateShadowMap(size int) *Texture {
	return d.CreateTexture2D(size, size, FormatDepth32Float, TextureRenderAttachment|TextureBinding)
}

// CreateComparisonSampler returns a linear-filtered less-equal comparison
// sampler for shadow lookups (hardware 2x2 PCF).
func (d *Device) CreateComparisonSampler() *Sampler {
	desc := samplerDescriptor{
		addressModeU:  addressClampToEdge,
		addressModeV:  addressClampToEdge,
		addressModeW:  addressClampToEdge,
		magFilter:     filterLinear,
		minFilter:     filterLinear,
		compare:       compareLessEqual,
		lodMaxClamp:   32,
		maxAnisotropy: 1,
	}
	return &Sampler{wgpuDeviceCreateSampler(d.h, &desc)}
}

const compareLessEqual = 0x4

// BeginShadowPass starts a depth-only pass writing into a shadow map
// (cleared to 1, stored for later sampling).
func (e *CommandEncoder) BeginShadowPass(depth *TextureView) *RenderPass {
	depthAtt := renderPassDepthStencilAttachment{
		view:            depth.h,
		depthLoadOp:     loadOpClear,
		depthStoreOp:    storeOpStore,
		depthClearValue: 1,
	}
	desc := renderPassDescriptor{
		depthStencilAttachment: uintptr(unsafe.Pointer(&depthAtt)),
	}
	h := wgpuCommandEncoderBeginRenderPass(e.h, &desc)
	runtime.KeepAlive(&depthAtt)
	return &RenderPass{h}
}

// BeginRenderPassDepth starts a pass with a color attachment (cleared to
// clear) and a depth attachment (cleared to 1.0).
func (e *CommandEncoder) BeginRenderPassDepth(view, depth *TextureView, clear Color) *RenderPass {
	return beginDepthPass(e, view, depth, clear, true)
}

// BeginRenderPassDepthKeep starts a pass that keeps whatever colour is
// already in the target and still clears the depth buffer. It is what a
// split screen needs: each pane is its own pass with its own camera, so
// the depth has to start again, while the panes already drawn must
// survive — the scissor is what stops the new one touching them.
func (e *CommandEncoder) BeginRenderPassDepthKeep(view, depth *TextureView) *RenderPass {
	return beginDepthPass(e, view, depth, Color{}, false)
}

func beginDepthPass(e *CommandEncoder, view, depth *TextureView, clear Color, clearColor bool) *RenderPass {
	load := uint32(loadOpLoad)
	if clearColor {
		load = loadOpClear
	}
	att := renderPassColorAttachment{
		view:       view.h,
		depthSlice: depthSliceUndefined,
		loadOp:     load,
		storeOp:    storeOpStore,
		clearValue: wgpuColor{clear.R, clear.G, clear.B, clear.A},
	}
	depthAtt := renderPassDepthStencilAttachment{
		view:            depth.h,
		depthLoadOp:     loadOpClear,
		depthStoreOp:    storeOpDiscard, // nothing reads depth after the pass yet
		depthClearValue: 1,
	}
	desc := renderPassDescriptor{
		colorAttachmentCount:   1,
		colorAttachments:       uintptr(unsafe.Pointer(&att)),
		depthStencilAttachment: uintptr(unsafe.Pointer(&depthAtt)),
	}
	h := wgpuCommandEncoderBeginRenderPass(e.h, &desc)
	runtime.KeepAlive(&att)
	runtime.KeepAlive(&depthAtt)
	return &RenderPass{h}
}

// SetIndexBuffer binds a uint32 index buffer.
func (r *RenderPass) SetIndexBuffer(b *Buffer) {
	wgpuRenderPassEncoderSetIndexBuffer(r.h, b.h, indexUint32, 0, uint64(b.size))
}

// DrawIndexed draws using the bound index buffer.
func (r *RenderPass) DrawIndexed(indexCount, instanceCount int) {
	wgpuRenderPassEncoderDrawIndexed(r.h, uint32(indexCount), uint32(instanceCount), 0, 0, 0)
}

// DrawIndexedInstanced draws with an explicit first instance, so shaders
// can index per-draw data with instance_index.
func (r *RenderPass) DrawIndexedInstanced(indexCount, instanceCount, firstInstance int) {
	wgpuRenderPassEncoderDrawIndexed(r.h, uint32(indexCount), uint32(instanceCount), 0, 0, uint32(firstInstance))
}
