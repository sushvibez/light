//go:build !nowebgpu

package wgpudev

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"sync"

	"light/internal/gpu"
)

// Device is the singleton WebGPU context.
type Device struct {
	Instance *gpu.Instance
	Adapter  *gpu.Adapter
	Device   *gpu.Device
	Queue    *gpu.Queue

	// AdapterName is the adapter's reported name (e.g. "Intel(R) Arc(TM) ...").
	AdapterName string
	// BackendName is the wgpu backend in use (Vulkan, Metal, D3D12, ...).
	BackendName string
	// Library is the wgpu-native shared library that was loaded.
	Library string

	// MaxBufferSize is the granted per-buffer byte cap; allocations above it
	// fail with a clear error instead of an invalid buffer.
	MaxBufferSize uint64
	// MaxStorageBufferBindingSize is the granted cap on a single storage
	// binding — a buffer larger than this can be allocated but not bound.
	MaxStorageBufferBindingSize uint64
	// MaxWorkgroupsPerDim is the cap on workgroups per dispatch axis
	// (WebGPU's default, 65535).
	MaxWorkgroupsPerDim uint32

	mu        sync.Mutex // guards everything below plus submit ordering
	shaders   map[string]*gpu.ShaderModule
	pipelines map[string]*pipelineEntry
	pending   int
	pool      *pool

	// The open batch: dispatches, copies and clears recorded into one
	// command encoder (and a compute pass while dispatching) and submitted
	// together. Per-dispatch parameters live in ring slots whose bytes are
	// staged on the host and uploaded once, right before the submit.
	enc     *gpu.CommandEncoder
	pass    *gpu.ComputePass
	batchN  int
	gen     uint64 // batch generation; Params of an older batch are re-staged
	bgs     []*gpu.BindGroup
	ring    *gpu.Buffer
	stage   []byte
	Submits uint64 // command buffers submitted (for tests and profiling)
}

// Batching limits: dispatches per command buffer, and the parameter ring
// (256-byte slots: WebGPU's default uniform / storage offset alignment).
const (
	slotAlign = 256
	ringBytes = 1 << 20
)

// maxBatch is the number of dispatches per command buffer
// ($LIGHT_WGPU_BATCH overrides, for experiments).
var maxBatch = func() int {
	if n, err := strconv.Atoi(os.Getenv("LIGHT_WGPU_BATCH")); err == nil && n > 0 {
		return n
	}
	return 256
}()

type pipelineEntry struct {
	pipeline *gpu.ComputePipeline
	layout   *gpu.BindGroupLayout
}

// maxPendingSubmits bounds the number of queued, unpolled submissions before
// we block on the device. wgpu keeps every resource referenced by a queued
// command alive, so a long run of unpolled dispatches would also hold on to
// freed buffers.
const maxPendingSubmits = 64

var (
	global   *Device
	initOnce sync.Once
	initErr  error
)

// Get returns the singleton device, opening it on first use: wgpu-native is
// loaded at run time (no cgo), from the first of LibraryCandidates that
// loads.
func Get() (*Device, error) {
	initOnce.Do(func() {
		global, initErr = open()
	})
	return global, initErr
}

// libName is wgpu-native's file name on this platform.
func libName() string { return gpu.DefaultLibNames()[0] }

// LibraryCandidates lists where wgpu-native is looked for, in order:
// $LIGHT_WGPU_LIB (a file path), next to the executable,
// ~/.cache/light/wgpu/, then the bare name (the system search path).
func LibraryCandidates() []string {
	var out []string
	if p := os.Getenv("LIGHT_WGPU_LIB"); p != "" {
		out = append(out, p)
	}
	if exe, err := os.Executable(); err == nil {
		out = append(out, filepath.Join(filepath.Dir(exe), libName()))
	}
	if home, err := os.UserHomeDir(); err == nil {
		out = append(out, filepath.Join(home, ".cache", "light", "wgpu", libName()))
	}
	return append(out, libName())
}

func open() (*Device, error) {
	cands := LibraryCandidates()
	var lib string
	for _, c := range cands {
		if c == libName() {
			lib = c // the system search path: let the loader decide
			break
		}
		if _, err := os.Stat(c); err == nil {
			lib = c
			break
		}
	}
	instance, err := gpu.NewInstance(lib)
	if err != nil {
		return nil, fmt.Errorf("wgpudev: cannot load %s (looked in %v; scripts/fetch-wgpu.sh fetches it): %w", libName(), cands, err)
	}
	adapter, err := instance.RequestAdapter(nil)
	if err != nil {
		instance.Release()
		return nil, fmt.Errorf("wgpudev: no WebGPU adapter: %w", err)
	}
	device, err := adapter.RequestDevice()
	if err != nil {
		adapter.Release()
		instance.Release()
		return nil, fmt.Errorf("wgpudev: request device: %w", err)
	}
	name, backend := adapter.Info()
	lim := device.Limits()
	d := &Device{
		Instance:                    instance,
		Adapter:                     adapter,
		Device:                      device,
		Queue:                       device.Queue(),
		AdapterName:                 name,
		BackendName:                 backend,
		Library:                     lib,
		MaxBufferSize:               uint64(lim.MaxBufferSize),
		MaxStorageBufferBindingSize: uint64(lim.MaxStorageBindingSize),
		MaxWorkgroupsPerDim:         65535,
		shaders:                     make(map[string]*gpu.ShaderModule),
		pipelines:                   make(map[string]*pipelineEntry),
	}
	d.pool = newPool(d)
	d.ring = device.CreateBuffer(gpu.BufferUniform|gpu.BufferStorage|gpu.BufferCopyDst, ringBytes)
	if !d.ring.Valid() {
		return nil, fmt.Errorf("wgpudev: create parameter ring failed")
	}
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

// encoderLocked returns the open batch's encoder, starting a batch if
// none is open. Caller holds d.mu.
func (d *Device) encoderLocked() *gpu.CommandEncoder {
	if d.enc == nil {
		d.enc = d.Device.CreateCommandEncoder()
	}
	return d.enc
}

// endPassLocked closes the open compute pass (copies and clears are
// recorded outside passes). Caller holds d.mu.
func (d *Device) endPassLocked() {
	if d.pass != nil {
		d.pass.End()
		d.pass = nil
	}
}

// submitLocked uploads the batch's staged parameters and submits it,
// polling the device every maxPendingSubmits batches. Caller holds d.mu.
func (d *Device) submitLocked() {
	if d.enc == nil {
		return
	}
	d.endPassLocked()
	if len(d.stage) > 0 {
		d.Queue.WriteBuffer(d.ring, 0, d.stage) // ordered before this submit, after the previous one
		d.stage = d.stage[:0]
	}
	d.Queue.Submit(d.enc.Finish()) // releases the command buffer
	d.enc, d.batchN = nil, 0
	d.gen++
	d.Submits++
	for _, bg := range d.bgs {
		bg.Release()
	}
	d.bgs = d.bgs[:0]
	d.pending++
	if d.pending >= maxPendingSubmits {
		d.flushLocked()
	}
}

// flushLocked submits the open batch and blocks until all queued work has
// completed. Caller holds d.mu.
func (d *Device) flushLocked() {
	if d.enc != nil {
		d.submitLocked()
	}
	if d.pending == 0 {
		return
	}
	d.Device.Poll(true)
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
	d.Device.Poll(true)
}

// Submit sends the open batch to the GPU without waiting for it.
func (d *Device) Submit() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.submitLocked()
}

// shaderLocked returns the cached module for name, compiling code on first use.
func (d *Device) shaderLocked(name, code string) (*gpu.ShaderModule, error) {
	if sm, ok := d.shaders[name]; ok {
		return sm, nil
	}
	before := gpu.ErrorCount()
	sm, err := d.Device.CreateShaderModuleWGSL(name, code)
	if err == nil && gpu.ErrorCount() != before {
		err = fmt.Errorf("validation failed (details on stderr)")
	}
	if err != nil {
		return nil, fmt.Errorf("wgpudev: compile shader %q: %w", name, err)
	}
	d.shaders[name] = sm
	return sm, nil
}

// pipelineLocked returns the cached pipeline for name. The bind group layout
// is derived from the shader (every declared binding must be used, as
// Light's shaders do).
func (d *Device) pipelineLocked(name, code string) (*pipelineEntry, error) {
	if pe, ok := d.pipelines[name]; ok {
		return pe, nil
	}
	sm, err := d.shaderLocked(name, code)
	if err != nil {
		return nil, err
	}
	before := gpu.ErrorCount()
	cp, err := d.Device.CreateComputePipeline(sm, "main")
	if err == nil && gpu.ErrorCount() != before {
		err = fmt.Errorf("validation failed (details on stderr)")
	}
	if err != nil {
		return nil, fmt.Errorf("wgpudev: pipeline %q: %w", name, err)
	}
	pe := &pipelineEntry{pipeline: cp, layout: cp.BindGroupLayout(0)}
	d.pipelines[name] = pe
	return pe, nil
}

// BindKind classifies a shader binding slot. With layouts derived from the
// shader it documents intent only; the WGSL declaration decides.
type BindKind uint8

const (
	ReadStorage BindKind = iota
	ReadWriteStorage
	Uniform
)

// Binding is one bind-group slot for Run. Offset/Size are in bytes; a zero
// Size binds the whole buffer from Offset. Param, when set, binds a
// per-dispatch parameter slot instead of Buf.
type Binding struct {
	Kind   BindKind
	Buf    *gpu.Buffer
	Offset uint64
	Size   uint64
	Param  *Param
}

// Param is a per-dispatch parameter block (uniform words or a metadata
// table) in a slot of the batch's parameter ring. It keeps its bytes, so a
// batch submitted between creating it and the Run that binds it only costs
// a re-stage into the new batch.
type Param struct {
	data []byte
	off  uint64
	gen  uint64
}

// Release is a no-op (slots are recycled per batch); it keeps the
// create / defer-Release shape of real buffers at call sites.
func (p *Param) Release() {}

// stageLocked places data in a ring slot of the open batch, submitting the
// batch first when the ring is full. Caller holds d.mu.
func (d *Device) stageLocked(data []byte) (off uint64) {
	n := (len(data) + slotAlign - 1) / slotAlign * slotAlign
	if n == 0 {
		n = slotAlign
	}
	if len(d.stage)+n > ringBytes {
		d.submitLocked()
	}
	off = uint64(len(d.stage))
	d.stage = append(d.stage, data...)
	d.stage = append(d.stage, make([]byte, n-len(data))...)
	return off
}

func (d *Device) newParam(data []byte) *Param {
	if len(data)%16 != 0 { // uniform blocks are vec4 multiples
		data = append(data, make([]byte, 16-len(data)%16)...)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return &Param{data: data, off: d.stageLocked(data), gen: d.gen}
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
	pe, err := d.pipelineLocked(name, code)
	if err != nil {
		return err
	}
	// Re-stage parameters created in an already submitted batch. Staging
	// can itself submit (ring full), so repeat until all are current.
	for again := true; again; {
		again = false
		gen := d.gen
		for _, b := range binds {
			if p := b.Param; p != nil && p.gen != gen {
				p.off, p.gen = d.stageLocked(p.data), d.gen
			}
		}
		for _, b := range binds {
			if b.Param != nil && b.Param.gen != d.gen {
				again = true
			}
		}
	}
	entries := make([]gpu.BufferBinding, len(binds))
	for i, b := range binds {
		if b.Param != nil {
			entries[i] = gpu.BufferBinding{Binding: i, Buffer: d.ring, Offset: b.Param.off, Size: uint64(len(b.Param.data))}
			continue
		}
		entries[i] = gpu.BufferBinding{Binding: i, Buffer: b.Buf, Offset: b.Offset, Size: b.Size}
	}
	bg := d.Device.CreateBufferBindGroup(pe.layout, entries)
	d.bgs = append(d.bgs, bg) // released after the batch is submitted
	if d.pass == nil {
		d.pass = d.encoderLocked().BeginComputePass()
	}
	d.pass.SetPipeline(pe.pipeline)
	d.pass.SetBindGroup(0, bg)
	d.pass.Dispatch(int(wx), int(wy), int(wz))
	if d.batchN++; d.batchN >= maxBatch {
		d.submitLocked()
	}
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

// newInit creates a buffer with the given usage holding data (padded to a
// 16-byte multiple).
func (d *Device) newInit(usage gpu.BufferUsage, data []byte) (*gpu.Buffer, error) {
	n := (len(data) + 15) &^ 15
	if n == 0 {
		n = 16
	}
	buf := d.Device.CreateBuffer(usage|gpu.BufferCopyDst, n)
	if !buf.Valid() {
		return nil, fmt.Errorf("wgpudev: create buffer (%d bytes) failed", n)
	}
	padded := data
	if len(data) != n {
		padded = make([]byte, n)
		copy(padded, data)
	}
	d.mu.Lock()
	d.Queue.WriteBuffer(buf, 0, padded)
	d.mu.Unlock()
	return buf, nil
}

// Uniform returns a per-dispatch uniform block holding the given u32
// words, in the open batch's parameter ring.
func (d *Device) Uniform(words ...uint32) (*Param, error) {
	data := make([]byte, 4*len(words))
	for i, w := range words {
		data[4*i] = byte(w)
		data[4*i+1] = byte(w >> 8)
		data[4*i+2] = byte(w >> 16)
		data[4*i+3] = byte(w >> 24)
	}
	return d.newParam(data), nil
}

// StorageTemp is Uniform for a read-only storage table (per-dispatch
// shape / stride metadata).
func (d *Device) StorageTemp(data []byte) (*Param, error) {
	return d.newParam(append([]byte(nil), data...)), nil
}

// StorageInit creates a persistent read-only storage buffer pre-filled with
// data (cached tables); the caller owns and releases it.
func (d *Device) StorageInit(data []byte) (*gpu.Buffer, error) {
	return d.newInit(gpu.BufferStorage, data)
}

// Write uploads data into buf at byte offset off (queue-ordered after any
// prior dispatch). The length is padded up to a 4-byte multiple.
func (d *Device) Write(buf *gpu.Buffer, off uint64, data []byte) error {
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
	d.submitLocked() // recorded work comes before this write on the queue
	d.Queue.WriteBuffer(buf, int(off), padded)
	return nil
}

// Read copies n bytes from buf (at byte offset off) back to the host via a
// MAP_READ staging buffer. It flushes every queued dispatch first.
func (d *Device) Read(buf *gpu.Buffer, off uint64, n int) ([]byte, error) {
	if n == 0 {
		return nil, nil
	}
	size := (n + 3) &^ 3
	d.mu.Lock()
	defer d.mu.Unlock()
	d.flushLocked()
	out, err := d.Device.ReadBufferAt(d.Queue, buf, off, size)
	if err != nil {
		return nil, fmt.Errorf("wgpudev: read: %w", err)
	}
	return out[:n], nil
}

// Copy enqueues a device-to-device copy of n bytes. All offsets and n must
// be 4-byte multiples.
func (d *Device) Copy(src *gpu.Buffer, srcOff uint64, dst *gpu.Buffer, dstOff uint64, n uint64) error {
	if n == 0 {
		return nil
	}
	return d.CopyMany([]CopyRegion{{Src: src, SrcOff: srcOff, Dst: dst, DstOff: dstOff, Size: n}})
}

// CopyRegion is one device-to-device copy for CopyMany.
type CopyRegion struct {
	Src    *gpu.Buffer
	SrcOff uint64
	Dst    *gpu.Buffer
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
	d.endPassLocked()
	enc := d.encoderLocked()
	for _, r := range regions {
		if r.Size > 0 {
			enc.CopyBufferToBuffer(r.Src, r.SrcOff, r.Dst, r.DstOff, r.Size)
		}
	}
	return nil
}

// Clear enqueues a zero-fill of n bytes of buf at byte offset off.
func (d *Device) Clear(buf *gpu.Buffer, off, n uint64) error {
	if n == 0 {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.endPassLocked()
	d.encoderLocked().ClearBuffer(buf, off, n)
	return nil
}
