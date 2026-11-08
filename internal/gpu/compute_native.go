//go:build !js

package gpu

import (
	"fmt"
	"runtime"
	"unsafe"
)

// Compute pipelines and buffer readback — the gpu surface phase 2's ECS
// columns and particle systems run on.

type programmableStageDescriptor struct {
	nextInChain   uintptr
	module        uintptr
	entryPoint    stringView
	constantCount uintptr
	constants     uintptr
}

type computePipelineDescriptor struct {
	nextInChain uintptr
	label       stringView
	layout      uintptr
	compute     programmableStageDescriptor
}

// ComputePipeline is a compiled compute shader entry point.
type ComputePipeline struct{ h uintptr }

func (p *ComputePipeline) Release() { wgpuComputePipelineRelease(p.h) }

// BindGroupLayout returns the pipeline's auto-derived layout for a group.
func (p *ComputePipeline) BindGroupLayout(index int) *BindGroupLayout {
	return &BindGroupLayout{wgpuComputePipelineGetBindGroupLayout(p.h, uint32(index))}
}

// CreateComputePipeline compiles a compute pipeline from a WGSL module's
// entry point, with an auto-derived layout.
func (d *Device) CreateComputePipeline(m *ShaderModule, entry string) (*ComputePipeline, error) {
	e := []byte(entry)
	desc := computePipelineDescriptor{
		compute: programmableStageDescriptor{
			module:     m.h,
			entryPoint: strView(e),
		},
	}
	h := wgpuDeviceCreateComputePipeline(d.h, &desc)
	runtime.KeepAlive(e)
	if h == 0 {
		return nil, fmt.Errorf("gpu: compute pipeline creation failed")
	}
	return &ComputePipeline{h}, nil
}

// ComputePass records compute work.
type ComputePass struct{ h uintptr }

// BeginComputePass starts recording compute commands.
func (e *CommandEncoder) BeginComputePass() *ComputePass {
	return &ComputePass{wgpuCommandEncoderBeginComputePass(e.h, nil)}
}

func (c *ComputePass) SetPipeline(p *ComputePipeline) { wgpuComputePassEncoderSetPipeline(c.h, p.h) }

func (c *ComputePass) SetBindGroup(index int, g *BindGroup) {
	wgpuComputePassEncoderSetBindGroup(c.h, uint32(index), g.h, 0, nil)
}

// Dispatch launches x*y*z workgroups.
func (c *ComputePass) Dispatch(x, y, z int) {
	wgpuComputePassEncoderDispatch(c.h, uint32(x), uint32(y), uint32(z))
}

// End closes and releases the pass encoder.
func (c *ComputePass) End() {
	wgpuComputePassEncoderEnd(c.h)
	wgpuComputePassEncoderRelease(c.h)
}

// ReadBuffer copies size bytes of b back to the CPU, blocking until the
// GPU is done — for tests and tools, not per-frame use. b needs
// BufferCopySrc usage.
func (d *Device) ReadBuffer(q *Queue, b *Buffer, size int) ([]byte, error) {
	staging := d.CreateBuffer(BufferMapRead|BufferCopyDst, size)
	defer staging.Release()

	enc := d.CreateCommandEncoder()
	wgpuCommandEncoderCopyBufferToBuffer(enc.h, b.h, 0, staging.h, 0, uint64(size))
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
