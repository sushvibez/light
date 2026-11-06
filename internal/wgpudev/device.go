//go:build webgpu

package wgpudev

import (
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/cogentcore/webgpu/wgpu"
)

// Device is the singleton WebGPU context.
type Device struct {
	Instance *wgpu.Instance
	Adapter  *wgpu.Adapter
	Device   *wgpu.Device
	Queue    *wgpu.Queue

	// AdapterName is the adapter's reported name (e.g. "NVIDIA GeForce ...").
	AdapterName string
	// BackendName is the wgpu backend in use (Vulkan, Metal, DX12, ...).
	BackendName string

	// MaxBufferSize is the granted per-buffer byte cap; allocations above it
	// fail with a clear error instead of an opaque wgpu validation error.
	MaxBufferSize uint64
	// MaxStorageBufferBindingSize is the granted cap on a single storage
	// binding — a buffer larger than this can be allocated but not bound.
	MaxStorageBufferBindingSize uint64
	// MaxWorkgroupsPerDim is the granted cap on workgroups per dispatch axis.
	MaxWorkgroupsPerDim uint32

	mu        sync.Mutex // guards everything below plus submit ordering
	shaders   map[string]*wgpu.ShaderModule
	pipelines map[string]*pipelineEntry
	pending   int
	pool      *pool
}

type pipelineEntry struct {
	pipeline *wgpu.ComputePipeline
	layout   *wgpu.BindGroupLayout
}

// maxPendingSubmits bounds the number of queued, unpolled submissions before
// we block on the device. wgpu keeps every resource referenced by a queued
// command alive, so a long run of unpolled dispatches would also hold on to
// freed buffers.
const maxPendingSubmits = 64

// requestedBufferCap is the per-buffer limit we ask the device for. Measured
// on an RTX 5060 Ti with wgpu-native 0.23: 2 GiB is granted and both
// allocates and binds; 4 GiB-1 is refused; >= 4 GiB allocates but cannot be
// bound as a storage buffer. The adapter advertises a u64::MAX sentinel for
// MaxBufferSize which the device request rejects, so an explicit value is
// required to get past the 256 MiB WebGPU default.
const requestedBufferCap = 2 << 30

var (
	global   *Device
	initOnce sync.Once
	initErr  error
)

// Get returns the singleton device, opening it on first use.
func Get() (*Device, error) {
	initOnce.Do(func() {
		global, initErr = open()
	})
	return global, initErr
}

func open() (*Device, error) {
	// A nil descriptor on purpose: cogentcore/webgpu v0.23.0 mallocs the
	// WGPUInstanceExtras chain without zeroing it, so a non-nil descriptor
	// hands wgpu-native garbage in `flags`/`gles3MinorVersion` and the
	// process intermittently aborts with "Unknown gles3 minor version".
	instance := wgpu.CreateInstance(nil)
	if instance == nil {
		return nil, fmt.Errorf("wgpudev: failed to create WebGPU instance")
	}
	adapter, err := instance.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: wgpu.PowerPreferenceHighPerformance,
	})
	if err != nil {
		instance.Release()
		return nil, fmt.Errorf("wgpudev: no WebGPU adapter: %w", err)
	}
	info := adapter.GetInfo()
	adapterLimits := adapter.GetLimits().Limits

	// Start from the WebGPU defaults and raise only what we need. Passing
	// the raw adapter limits fails validation (several are u64::MAX
	// sentinels), which silently leaves the 256 MiB default in place.
	req := wgpu.DefaultLimits()
	want := uint64(requestedBufferCap)
	if adapterLimits.MaxStorageBufferBindingSize > 0 && adapterLimits.MaxStorageBufferBindingSize < want {
		want = adapterLimits.MaxStorageBufferBindingSize
	}
	req.MaxBufferSize = want
	req.MaxStorageBufferBindingSize = want
	if adapterLimits.MaxComputeWorkgroupsPerDimension > 0 && adapterLimits.MaxComputeWorkgroupsPerDimension != wgpu.LimitU32Undefined {
		req.MaxComputeWorkgroupsPerDimension = adapterLimits.MaxComputeWorkgroupsPerDimension
	}

	device, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{
		Label:          "light",
		RequiredLimits: &wgpu.RequiredLimits{Limits: req},
	})
	if err != nil {
		// Retry with the plain defaults so a conservative adapter still works.
		device, err = adapter.RequestDevice(&wgpu.DeviceDescriptor{Label: "light"})
		if err != nil {
			adapter.Release()
			instance.Release()
			return nil, fmt.Errorf("wgpudev: request device: %w", err)
		}
	}
	granted := device.GetLimits().Limits
	d := &Device{
		Instance:                    instance,
		Adapter:                     adapter,
		Device:                      device,
		Queue:                       device.GetQueue(),
		AdapterName:                 info.Name,
		BackendName:                 info.BackendType.String(),
		MaxBufferSize:               granted.MaxBufferSize,
		MaxStorageBufferBindingSize: granted.MaxStorageBufferBindingSize,
		MaxWorkgroupsPerDim:         granted.MaxComputeWorkgroupsPerDimension,
		shaders:                     make(map[string]*wgpu.ShaderModule),
		pipelines:                   make(map[string]*pipelineEntry),
	}
	if d.MaxWorkgroupsPerDim == 0 || d.MaxWorkgroupsPerDim == wgpu.LimitU32Undefined {
		d.MaxWorkgroupsPerDim = 65535
	}
	d.pool = newPool(d)
	return d, nil
}

// CheckSize returns a descriptive error when a single buffer of n bytes
// cannot be allocated and bound on this device.
func (d *Device) CheckSize(n uint64) error {
	cap := d.MaxBufferSize
	if d.MaxStorageBufferBindingSize < cap {
		cap = d.MaxStorageBufferBindingSize
	}
	if cap > 0 && n > cap {
		return fmt.Errorf("wgpudev: buffer of %d bytes (%d MiB) exceeds the granted WebGPU per-buffer limit of %d MiB on %q; split the tensor or use the CUDA backend",
			n, n>>20, cap>>20, d.AdapterName)
	}
	return nil
}

// submitLocked queues a command buffer and lazily flushes. Caller holds d.mu.
func (d *Device) submitLocked(cmd *wgpu.CommandBuffer) {
	d.Queue.Submit(cmd)
	cmd.Release()
	d.pending++
	if d.pending >= maxPendingSubmits {
		d.flushLocked()
	}
}

// flushLocked blocks until all queued work has completed. Caller holds d.mu.
func (d *Device) flushLocked() {
	if d.pending == 0 {
		return
	}
	if !d.Device.Poll(true, nil) {
		for !d.Device.Poll(false, nil) {
			runtime.Gosched()
		}
	}
	d.pending = 0
}

// Flush blocks until every queued dispatch has completed.
func (d *Device) Flush() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.flushLocked()
}

// Synchronize is Flush plus a final poll, for benchmarks that want to time a
// fully-drained queue.
func (d *Device) Synchronize() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.flushLocked()
	d.Device.Poll(true, nil)
}

// shaderLocked returns the cached module for name, compiling code on first use.
func (d *Device) shaderLocked(name, code string) (*wgpu.ShaderModule, error) {
	if sm, ok := d.shaders[name]; ok {
		return sm, nil
	}
	sm, err := d.Device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          name,
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: code},
	})
	if err != nil {
		return nil, fmt.Errorf("wgpudev: compile shader %q: %w", name, err)
	}
	d.shaders[name] = sm
	return sm, nil
}

// pipelineLocked returns the cached pipeline for name, building the bind
// group layout from the binding kinds on first use.
func (d *Device) pipelineLocked(name, code string, kinds []BindKind) (*pipelineEntry, error) {
	if pe, ok := d.pipelines[name]; ok {
		return pe, nil
	}
	sm, err := d.shaderLocked(name, code)
	if err != nil {
		return nil, err
	}
	entries := make([]wgpu.BindGroupLayoutEntry, len(kinds))
	for i, k := range kinds {
		var bt wgpu.BufferBindingType
		switch k {
		case ReadStorage:
			bt = wgpu.BufferBindingTypeReadOnlyStorage
		case ReadWriteStorage:
			bt = wgpu.BufferBindingTypeStorage
		case Uniform:
			bt = wgpu.BufferBindingTypeUniform
		}
		entries[i] = wgpu.BindGroupLayoutEntry{
			Binding:    uint32(i),
			Visibility: wgpu.ShaderStageCompute,
			Buffer:     wgpu.BufferBindingLayout{Type: bt},
		}
	}
	bgl, err := d.Device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label:   name + "_bgl",
		Entries: entries,
	})
	if err != nil {
		return nil, fmt.Errorf("wgpudev: bind group layout %q: %w", name, err)
	}
	pl, err := d.Device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            name + "_layout",
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgl},
	})
	if err != nil {
		bgl.Release()
		return nil, fmt.Errorf("wgpudev: pipeline layout %q: %w", name, err)
	}
	defer pl.Release()
	cp, err := d.Device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   name,
		Layout:  pl,
		Compute: wgpu.ProgrammableStageDescriptor{Module: sm, EntryPoint: "main"},
	})
	if err != nil {
		bgl.Release()
		return nil, fmt.Errorf("wgpudev: pipeline %q: %w", name, err)
	}
	pe := &pipelineEntry{pipeline: cp, layout: bgl}
	d.pipelines[name] = pe
	return pe, nil
}

// BindKind classifies a shader binding slot.
type BindKind uint8

const (
	ReadStorage BindKind = iota
	ReadWriteStorage
	Uniform
)

// Binding is one bind-group slot for Run. Offset/Size are in bytes; a zero
// Size binds the whole buffer from Offset.
type Binding struct {
	Kind   BindKind
	Buf    *wgpu.Buffer
	Offset uint64
	Size   uint64
}

// Run dispatches the compute shader `code` (cached under `name`, entry point
// "main") with the given bindings and workgroup grid. The bind group is
// released after submission; wgpu keeps it alive until the dispatch runs.
func (d *Device) Run(name, code string, binds []Binding, wx, wy, wz uint32) error {
	if wx == 0 || wy == 0 || wz == 0 {
		return nil
	}
	if wx > d.MaxWorkgroupsPerDim || wy > d.MaxWorkgroupsPerDim || wz > d.MaxWorkgroupsPerDim {
		return fmt.Errorf("wgpudev: %s: dispatch (%d,%d,%d) exceeds max workgroups per dimension %d", name, wx, wy, wz, d.MaxWorkgroupsPerDim)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	kinds := make([]BindKind, len(binds))
	for i, b := range binds {
		kinds[i] = b.Kind
	}
	pe, err := d.pipelineLocked(name, code, kinds)
	if err != nil {
		return err
	}
	entries := make([]wgpu.BindGroupEntry, len(binds))
	for i, b := range binds {
		size := b.Size
		if size == 0 {
			size = b.Buf.GetSize() - b.Offset
		}
		entries[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b.Buf, Offset: b.Offset, Size: size}
	}
	bg, err := d.Device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   name + "_bg",
		Layout:  pe.layout,
		Entries: entries,
	})
	if err != nil {
		return fmt.Errorf("wgpudev: %s: bind group: %w", name, err)
	}
	defer bg.Release()
	enc, err := d.Device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{Label: name})
	if err != nil {
		return fmt.Errorf("wgpudev: %s: command encoder: %w", name, err)
	}
	defer enc.Release()
	pass := enc.BeginComputePass(&wgpu.ComputePassDescriptor{Label: name})
	pass.SetPipeline(pe.pipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(wx, wy, wz)
	if err := pass.End(); err != nil {
		pass.Release()
		return fmt.Errorf("wgpudev: %s: end pass: %w", name, err)
	}
	pass.Release()
	cmd, err := enc.Finish(&wgpu.CommandBufferDescriptor{Label: name})
	if err != nil {
		return fmt.Errorf("wgpudev: %s: finish: %w", name, err)
	}
	d.submitLocked(cmd)
	return nil
}

// Grid1D splits n work items into a (x, y) workgroup grid for a shader with
// workgroup_size(wg) that computes its flat index as
// `gid.x + gid.y * (maxPerDim * wg)`. y stays 1 until x overflows the
// per-dimension cap.
func (d *Device) Grid1D(n, wg uint64) (uint32, uint32) {
	if n == 0 {
		return 0, 0
	}
	groups := (n + wg - 1) / wg
	maxX := uint64(d.MaxWorkgroupsPerDim)
	if groups <= maxX {
		return uint32(groups), 1
	}
	return uint32(maxX), uint32((groups + maxX - 1) / maxX)
}

// Rows splits r rows into an (x, y) grid of one workgroup per row, for
// row-per-workgroup shaders that compute `row = wid.x + wid.y * maxPerDim`.
func (d *Device) Rows(r uint64) (uint32, uint32) {
	if r == 0 {
		return 0, 0
	}
	maxX := uint64(d.MaxWorkgroupsPerDim)
	if r <= maxX {
		return uint32(r), 1
	}
	return uint32(maxX), uint32((r + maxX - 1) / maxX)
}

// Uniform creates a uniform buffer holding the given u32 words (padded to a
// 16-byte multiple). The caller releases it after Run returns.
func (d *Device) Uniform(words ...uint32) (*wgpu.Buffer, error) {
	n := len(words)
	for n%4 != 0 {
		n++
	}
	data := make([]byte, 4*n)
	for i, w := range words {
		data[4*i] = byte(w)
		data[4*i+1] = byte(w >> 8)
		data[4*i+2] = byte(w >> 16)
		data[4*i+3] = byte(w >> 24)
	}
	buf, err := d.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "uniform",
		Contents: data,
		Usage:    wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("wgpudev: uniform: %w", err)
	}
	return buf, nil
}

// StorageInit creates a read-only storage buffer pre-filled with data (used
// for per-dispatch metadata such as shape/stride tables). The caller
// releases it after Run returns.
func (d *Device) StorageInit(data []byte) (*wgpu.Buffer, error) {
	n := len(data)
	for n%4 != 0 || n == 0 {
		n++
	}
	padded := data
	if n != len(data) {
		padded = make([]byte, n)
		copy(padded, data)
	}
	buf, err := d.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "meta",
		Contents: padded,
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("wgpudev: meta buffer: %w", err)
	}
	return buf, nil
}

// Write uploads data into buf at byte offset off (queue-ordered after any
// prior dispatch). The length is padded up to a 4-byte multiple.
func (d *Device) Write(buf *wgpu.Buffer, off uint64, data []byte) error {
	if len(data) == 0 {
		return nil
	}
	padded := data
	if len(data)%4 != 0 {
		padded = make([]byte, (len(data)+3)&^3)
		copy(padded, data)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if err := d.Queue.WriteBuffer(buf, off, padded); err != nil {
		return fmt.Errorf("wgpudev: write buffer: %w", err)
	}
	return nil
}

// Read copies n bytes from buf (at byte offset off) back to the host via a
// MAP_READ staging buffer. It flushes every queued dispatch first.
func (d *Device) Read(buf *wgpu.Buffer, off uint64, n int) ([]byte, error) {
	if n == 0 {
		return nil, nil
	}
	size := uint64((n + 3) &^ 3)
	d.mu.Lock()
	defer d.mu.Unlock()
	staging, err := d.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "staging_readback",
		Size:  size,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("wgpudev: staging buffer: %w", err)
	}
	defer staging.Release()
	enc, err := d.Device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{Label: "readback"})
	if err != nil {
		return nil, err
	}
	defer enc.Release()
	if err := enc.CopyBufferToBuffer(buf, off, staging, 0, size); err != nil {
		return nil, fmt.Errorf("wgpudev: readback copy: %w", err)
	}
	cmd, err := enc.Finish(&wgpu.CommandBufferDescriptor{})
	if err != nil {
		return nil, err
	}
	d.submitLocked(cmd)
	d.flushLocked()

	done := make(chan wgpu.BufferMapAsyncStatus, 1)
	if err := staging.MapAsync(wgpu.MapModeRead, 0, size, func(s wgpu.BufferMapAsyncStatus) { done <- s }); err != nil {
		return nil, fmt.Errorf("wgpudev: map async: %w", err)
	}
	deadline := time.Now().Add(60 * time.Second)
	for {
		d.Device.Poll(false, nil)
		select {
		case s := <-done:
			if s != wgpu.BufferMapAsyncStatusSuccess {
				return nil, fmt.Errorf("wgpudev: map failed: %v", s)
			}
			out := make([]byte, n)
			copy(out, staging.GetMappedRange(0, uint(size)))
			if err := staging.Unmap(); err != nil {
				return nil, err
			}
			return out, nil
		default:
			if time.Now().After(deadline) {
				return nil, fmt.Errorf("wgpudev: map async timed out")
			}
			runtime.Gosched()
		}
	}
}

// Copy enqueues a device-to-device copy of n bytes. All offsets and n must
// be 4-byte multiples.
func (d *Device) Copy(src *wgpu.Buffer, srcOff uint64, dst *wgpu.Buffer, dstOff uint64, n uint64) error {
	if n == 0 {
		return nil
	}
	return d.CopyMany([]CopyRegion{{Src: src, SrcOff: srcOff, Dst: dst, DstOff: dstOff, Size: n}})
}

// CopyRegion is one device-to-device copy for CopyMany.
type CopyRegion struct {
	Src    *wgpu.Buffer
	SrcOff uint64
	Dst    *wgpu.Buffer
	DstOff uint64
	Size   uint64
}

// CopyMany enqueues several device-to-device copies in one command buffer.
func (d *Device) CopyMany(regions []CopyRegion) error {
	if len(regions) == 0 {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	enc, err := d.Device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{Label: "copy"})
	if err != nil {
		return err
	}
	defer enc.Release()
	for _, r := range regions {
		if r.Size == 0 {
			continue
		}
		if err := enc.CopyBufferToBuffer(r.Src, r.SrcOff, r.Dst, r.DstOff, r.Size); err != nil {
			return fmt.Errorf("wgpudev: copy: %w", err)
		}
	}
	cmd, err := enc.Finish(&wgpu.CommandBufferDescriptor{})
	if err != nil {
		return err
	}
	d.submitLocked(cmd)
	return nil
}

// Clear enqueues a zero-fill of n bytes of buf at byte offset off.
func (d *Device) Clear(buf *wgpu.Buffer, off, n uint64) error {
	if n == 0 {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.clearLocked(buf, off, n)
}

func (d *Device) clearLocked(buf *wgpu.Buffer, off, n uint64) error {
	enc, err := d.Device.CreateCommandEncoder(&wgpu.CommandEncoderDescriptor{Label: "clear"})
	if err != nil {
		return err
	}
	defer enc.Release()
	if err := enc.ClearBuffer(buf, off, n); err != nil {
		return fmt.Errorf("wgpudev: clear: %w", err)
	}
	cmd, err := enc.Finish(&wgpu.CommandBufferDescriptor{})
	if err != nil {
		return err
	}
	d.submitLocked(cmd)
	return nil
}
