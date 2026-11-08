//go:build !js

package gpu

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"light/internal/ffi"
)

// Public WebGPU types for the native backend. The API is a thin, faithful
// mirror of WebGPU (DESIGN.md §5); convenience belongs in the render module.
// Phase-0 scope: everything a surface-cleared, bufferless triangle needs.

// ---- raw symbols (wgpu-native v0.19.4.1) ----

// ---- raw struct mirrors (64-bit layout, verified against webgpu.h) ----

type chainedStruct struct {
	next  uintptr
	sType uint32
	_     uint32
}

const (
	sTypeShaderSourceSPIRV                = 0x1
	sTypeShaderSourceWGSL                 = 0x2
	sTypeSurfaceSourceWindowsHWND         = 0x5
	sTypeSurfaceSourceXlibWindow          = 0x6
	sTypeSurfaceSourceWaylandSurface      = 0x7
	sTypeSurfaceSourceAndroidNativeWindow = 0x8
	sTypeInstanceExtras                   = 0x00030006 // wgpu-native extension
)

// WGPUInstanceBackend bits (wgpu.h) — which backends the instance enumerates.
const (
	backendAll    = 0
	backendVulkan = 1 << 0
	backendGL     = 1 << 1
	backendMetal  = 1 << 2
	backendDX12   = 1 << 3
)

// instanceDescriptor mirrors WGPUInstanceDescriptor (v29).
type instanceDescriptor struct {
	nextInChain          uintptr
	requiredFeatureCount uintptr
	requiredFeatures     uintptr
	requiredLimits       uintptr
}

// nativeDisplayHandle mirrors WGPUNativeDisplayHandle: a {type, union} where
// the largest union member is WGPUXlibDisplayHandle {void*, int}.
type nativeDisplayHandle struct {
	typ uint32
	_   uint32
	ptr uintptr
	scr int32
	_   uint32
}

// instanceExtras mirrors WGPUInstanceExtras (wgpu.h v29).
type instanceExtras struct {
	chain                   chainedStruct
	backends                uint64 // WGPUInstanceBackend (WGPUFlags = uint64)
	flags                   uint64
	dx12ShaderCompiler      uint32
	gles3MinorVersion       uint32
	glFenceBehaviour        uint32
	_                       uint32
	dxcPath                 stringView
	dxcMaxShaderModel       uint32
	dx12PresentationSystem  uint32
	budgetForDeviceCreation uintptr
	budgetForDeviceLoss     uintptr
	displayHandle           nativeDisplayHandle
}

// backendMask picks which native backends wgpu may enumerate. wgpu orders
// adapters Vulkan-first, but on Windows the Vulkan swapchain path is the
// slow/buggy one (on hybrid-graphics laptops wgpuSurfaceGetCurrentTexture
// was observed blocking ~500ms per frame); DX12 presents through DXGI/DWM
// properly, so it is the Windows default — like every browser's WebGPU.
// GFX_WGPU_BACKEND=vulkan|dx12|gl|metal|all overrides for experiments.
func backendMask() uint64 {
	switch os.Getenv("GFX_WGPU_BACKEND") {
	case "vulkan":
		return backendVulkan
	case "dx12":
		return backendDX12
	case "gl":
		return backendGL
	case "metal":
		return backendMetal
	case "all":
		return backendAll
	}
	if runtime.GOOS == "windows" {
		return backendDX12
	}
	return backendAll
}

// presentOverride, when non-empty, wins over GFX_WGPU_PRESENT for
// subsequent Surface.Configure calls — see SetPresentOverride.
var presentOverride string

// SetPresentOverride forces a present mode ("immediate", "mailbox",
// "fifo") for surfaces configured after the call; "" restores the
// GFX_WGPU_PRESENT/FIFO default. The shell switches to immediate during
// interactive window resizes: a vsync'd present inside the OS modal
// resize loop lags the drag by up to a vblank, and the compositor shows
// the previous frame stretched to the new size in the meantime.
func SetPresentOverride(mode string) { presentOverride = mode }

// presentMode picks the WGPUPresentMode for surface configuration. FIFO
// (vsync) is the default; GFX_WGPU_PRESENT=immediate|mailbox|fifo overrides
// for latency experiments and machines whose vsync path misbehaves.
func presentMode() uint32 {
	mode := presentOverride
	if mode == "" {
		mode = os.Getenv("GFX_WGPU_PRESENT")
	}
	switch mode {
	case "immediate":
		return 0x3 // WGPUPresentMode_Immediate
	case "mailbox":
		return 0x4 // WGPUPresentMode_Mailbox
	}
	return presentModeFifo
}

// powerPreference picks the WGPUPowerPreference for adapter requests.
// Default is high-performance so hybrid-graphics machines don't land on the
// software fallback; GFX_WGPU_POWER=low|high overrides (low = the iGPU, which
// on hybrid laptops is the adapter actually driving the display).
func powerPreference() uint32 {
	switch os.Getenv("GFX_WGPU_POWER") {
	case "low":
		return 0x1 // WGPUPowerPreference_LowPower
	case "high":
		return powerPreferenceHighPerformance
	}
	return powerPreferenceHighPerformance
}

// dx12SwapchainKind picks the DX12 presentation system. Default is a DXGI
// swapchain on a DirectComposition visual (what GPUI/browsers use): DWM
// composes the visual atomically with window geometry, so interactive
// resizes don't show the previous frame stretched to the new size — the
// visible "text shimmer" of the plain-HWND swapchain. Ignored on non-DX12
// backends; GFX_WGPU_DX12_SWAPCHAIN=hwnd|visual overrides for debugging.
func dx12SwapchainKind() uint32 {
	switch os.Getenv("GFX_WGPU_DX12_SWAPCHAIN") {
	case "hwnd":
		return 1 // WGPUDx12SwapchainKind_DxgiFromHwnd
	case "visual":
		return 2 // WGPUDx12SwapchainKind_DxgiFromVisual
	}
	return 2
}

// createInstance makes the WGPUInstance, chaining InstanceExtras when a
// specific backend set is wanted (backendAll needs no extras).
func createInstance() uintptr {
	mask := backendMask()
	if mask == backendAll {
		return wgpuCreateInstance(nil)
	}
	extras := instanceExtras{
		chain:                  chainedStruct{sType: sTypeInstanceExtras},
		backends:               mask,
		dx12PresentationSystem: dx12SwapchainKind(),
	}
	desc := instanceDescriptor{nextInChain: uintptr(unsafe.Pointer(&extras))}
	h := wgpuCreateInstance(unsafe.Pointer(&desc))
	runtime.KeepAlive(&extras)
	return h
}

type surfaceDescriptor struct {
	nextInChain uintptr
	label       stringView
}

type surfaceDescriptorFromXlibWindow struct {
	chain   chainedStruct
	display uintptr
	window  uint64
}

type surfaceDescriptorFromWindowsHWND struct {
	chain     chainedStruct
	hinstance uintptr
	hwnd      uintptr
}

type surfaceDescriptorFromAndroidNativeWindow struct {
	chain  chainedStruct
	window uintptr
}

// requestAdapterOptions mirrors WGPURequestAdapterOptions (29.x): featureLevel
// and forceFallbackAdapter come before backendType, and compatibleSurface is
// last.
type requestAdapterOptions struct {
	nextInChain          uintptr
	featureLevel         uint32
	powerPreference      uint32
	forceFallbackAdapter uint32
	backendType          uint32
	compatibleSurface    uintptr
}

type surfaceConfiguration struct {
	nextInChain     uintptr
	device          uintptr
	format          uint32
	_               uint32
	usage           uint64
	width           uint32
	height          uint32
	viewFormatCount uintptr
	viewFormats     uintptr
	alphaMode       uint32
	presentMode     uint32
}

type surfaceCapabilities struct {
	nextInChain      uintptr
	usages           uint64
	formatCount      uintptr
	formats          uintptr
	presentModeCount uintptr
	presentModes     uintptr
	alphaModeCount   uintptr
	alphaModes       uintptr
}

type surfaceTexture struct {
	nextInChain uintptr
	texture     uintptr
	status      uint32
	_           uint32
}

type shaderSourceWGSL struct {
	chain chainedStruct
	code  stringView
}

// shaderSourceSPIRV is WGPUShaderSourceSPIRV: a word count and a pointer to
// the SPIR-V words. wgpu runs them through naga's SPIR-V frontend, so this
// is validated like any other shader — it is not a passthrough.
type shaderSourceSPIRV struct {
	chain    chainedStruct
	codeSize uint32
	_        uint32 // padding to the pointer's alignment
	code     uintptr
}

type shaderModuleDescriptor struct {
	nextInChain uintptr
	label       stringView
}

type vertexState struct {
	nextInChain   uintptr
	module        uintptr
	entryPoint    stringView
	constantCount uintptr
	constants     uintptr
	bufferCount   uintptr
	buffers       uintptr
}

type primitiveState struct {
	nextInChain      uintptr
	topology         uint32
	stripIndexFormat uint32
	frontFace        uint32
	cullMode         uint32
	unclippedDepth   uint32 // added in 29.x
	_                uint32
}

type multisampleState struct {
	nextInChain            uintptr
	count                  uint32
	mask                   uint32
	alphaToCoverageEnabled uint32
	_                      uint32
}

type fragmentState struct {
	nextInChain   uintptr
	module        uintptr
	entryPoint    stringView
	constantCount uintptr
	constants     uintptr
	targetCount   uintptr
	targets       uintptr
}

type colorTargetState struct {
	nextInChain uintptr
	format      uint32
	_           uint32
	blend       uintptr
	writeMask   uint64
}

type renderPipelineDescriptor struct {
	nextInChain  uintptr
	label        stringView
	layout       uintptr
	vertex       vertexState
	primitive    primitiveState
	depthStencil uintptr
	multisample  multisampleState
	fragment     uintptr
}

type wgpuColor struct {
	r, g, b, a float64
}

type renderPassColorAttachment struct {
	nextInChain   uintptr
	view          uintptr
	depthSlice    uint32
	_             uint32
	resolveTarget uintptr
	loadOp        uint32
	storeOp       uint32
	clearValue    wgpuColor
}

type renderPassDescriptor struct {
	nextInChain            uintptr
	label                  stringView
	colorAttachmentCount   uintptr
	colorAttachments       uintptr
	depthStencilAttachment uintptr
	occlusionQuerySet      uintptr
	timestampWrites        uintptr
}

// ---- public API ----

// TextureFormat is a WGPUTextureFormat value.
type TextureFormat uint32

// Color is a render-pass clear color (linear, 0..1).
type Color struct{ R, G, B, A float64 }

const (
	loadOpLoad            = 0x1
	loadOpClear           = 0x2
	storeOpStore          = 0x1
	topologyLineList      = 0x2
	topologyTriangleList  = 0x4
	usageRenderAttachment = 0x10
	alphaModeAuto         = 0x0
	presentModeFifo       = 0x1
	colorWriteMaskAll     = 0xF
	featureLevelCore      = 0x2
	depthSliceUndefined   = 0xFFFFFFFF // WGPU_DEPTH_SLICE_UNDEFINED (non-3D views)

	// WGPUPowerPreference: prefer the discrete GPU. Without this, hybrid-graphics
	// laptops can hand back the software fallback (WARP / Microsoft Basic Render
	// Driver) instead of real hardware.
	powerPreferenceHighPerformance = 0x2
)

// Instance is the WebGPU entry point.
type Instance struct{ h uintptr }

// NewInstance loads wgpu-native (if not already loaded) and creates an
// instance.
func NewInstance(libPaths ...string) (*Instance, error) {
	if err := Load(libPaths...); err != nil {
		return nil, err
	}
	h := createInstance()
	if h == 0 {
		return nil, fmt.Errorf("gpu: wgpuCreateInstance returned nil")
	}
	return &Instance{h}, nil
}

func (i *Instance) Release() { wgpuInstanceRelease(i.h) }

// CreateSurfaceXlib wraps an X11 window in a WebGPU surface.
func (i *Instance) CreateSurfaceXlib(display uintptr, window uint64) (*Surface, error) {
	platform := surfaceDescriptorFromXlibWindow{
		chain:   chainedStruct{sType: sTypeSurfaceSourceXlibWindow},
		display: display,
		window:  window,
	}
	desc := surfaceDescriptor{nextInChain: uintptr(unsafe.Pointer(&platform))}
	h := wgpuInstanceCreateSurface(i.h, unsafe.Pointer(&desc))
	runtime.KeepAlive(&platform)
	if h == 0 {
		return nil, fmt.Errorf("gpu: wgpuInstanceCreateSurface returned nil")
	}
	return &Surface{h: h}, nil
}

// CreateSurfaceWindows wraps a Win32 window in a WebGPU surface.
func (i *Instance) CreateSurfaceWindows(hinstance, hwnd uintptr) (*Surface, error) {
	platform := surfaceDescriptorFromWindowsHWND{
		chain:     chainedStruct{sType: sTypeSurfaceSourceWindowsHWND},
		hinstance: hinstance,
		hwnd:      hwnd,
	}
	desc := surfaceDescriptor{nextInChain: uintptr(unsafe.Pointer(&platform))}
	h := wgpuInstanceCreateSurface(i.h, unsafe.Pointer(&desc))
	runtime.KeepAlive(&platform)
	if h == 0 {
		return nil, fmt.Errorf("gpu: wgpuInstanceCreateSurface returned nil")
	}
	return &Surface{h: h}, nil
}

// CreateSurfaceAndroid wraps an ANativeWindow* in a WebGPU surface.
func (i *Instance) CreateSurfaceAndroid(window unsafe.Pointer) (*Surface, error) {
	platform := surfaceDescriptorFromAndroidNativeWindow{
		chain:  chainedStruct{sType: sTypeSurfaceSourceAndroidNativeWindow},
		window: uintptr(window),
	}
	desc := surfaceDescriptor{nextInChain: uintptr(unsafe.Pointer(&platform))}
	h := wgpuInstanceCreateSurface(i.h, unsafe.Pointer(&desc))
	runtime.KeepAlive(&platform)
	if h == 0 {
		return nil, fmt.Errorf("gpu: wgpuInstanceCreateSurface returned nil")
	}
	return &Surface{h: h}, nil
}

// RequestAdapter picks an adapter, optionally constrained to be compatible
// with a surface.
func (i *Instance) RequestAdapter(compatible *Surface) (*Adapter, error) {
	opts := &requestAdapterOptions{
		featureLevel:    featureLevelCore,
		powerPreference: powerPreference(),
	}
	if compatible != nil {
		opts.compatibleSurface = compatible.h
	}
	h, err := await(i.h, func(cb reqCallbackInfo) {
		issueRequestAdapter(i.h, uintptr(unsafe.Pointer(opts)), cb)
	}, "request adapter")
	runtime.KeepAlive(opts)
	if err != nil {
		return nil, err
	}
	return &Adapter{h: h, inst: i.h}, nil
}

// Adapter is a physical GPU + backend pair.
type Adapter struct {
	h    uintptr
	inst uintptr
}

func (a *Adapter) Release() { wgpuAdapterRelease(a.h) }

// Name returns the adapter's human-readable name and backend.
func (a *Adapter) Name() string {
	var ai adapterInfo
	wgpuAdapterGetInfo(a.h, &ai)
	name := goStringView(ai.device.data, ai.device.length)
	backend := backendNames[ai.backendType]
	return fmt.Sprintf("%s (%s)", name, backend)
}

// deviceDescriptor mirrors WGPUDeviceDescriptor (29.x). queueDescriptor and
// the two callback-info structs are inline by value.
type deviceDescriptor struct {
	nextInChain          uintptr
	label                stringView
	requiredFeatureCount uintptr
	requiredFeatures     uintptr
	requiredLimits       uintptr
	queueNextInChain     uintptr    // WGPUQueueDescriptor.nextInChain
	queueLabel           stringView // WGPUQueueDescriptor.label
	deviceLostNext       uintptr    // WGPUDeviceLostCallbackInfo
	deviceLostMode       uint32
	_                    uint32
	deviceLostCallback   uintptr
	deviceLostUD1        uintptr
	deviceLostUD2        uintptr
	uncapturedNext       uintptr // WGPUUncapturedErrorCallbackInfo
	uncapturedCallback   uintptr
	uncapturedUD1        uintptr
	uncapturedUD2        uintptr
}

const featureDualSourceBlending = 0x11 // WGPUFeatureName_DualSourceBlending

// RequestDevice creates the logical device, requesting DualSourceBlending
// (for subpixel text) where the adapter offers it — a software or mobile
// GPU often does not, and a request it cannot meet fails outright — and
// wiring uncaptured GPU errors to stderr via the device descriptor's
// callback info (the standalone setter was removed).
func (a *Adapter) RequestDevice() (*Device, error) {
	var feats []uint32
	dual := wgpuAdapterHasFeature(a.h, featureDualSourceBlending) != 0
	if dual {
		feats = append(feats, featureDualSourceBlending)
	}
	limits, got := a.requestedLimits()
	desc := deviceDescriptor{
		requiredFeatureCount: uintptr(len(feats)),
		requiredLimits:       uintptr(unsafe.Pointer(&limits)),
		uncapturedCallback:   errorCallback(),
	}
	if len(feats) > 0 {
		desc.requiredFeatures = uintptr(unsafe.Pointer(&feats[0]))
	}
	h, err := await(a.inst, func(cb reqCallbackInfo) {
		issueRequestDevice(a.h, uintptr(unsafe.Pointer(&desc)), cb)
	}, "request device")
	runtime.KeepAlive(&desc)
	runtime.KeepAlive(feats)
	runtime.KeepAlive(&limits)
	if err != nil {
		return nil, err
	}
	return &Device{h: h, limits: got, dualSource: dual}, nil
}

// DualSourceBlending reports whether the device was created with the
// dual-source blending feature (subpixel text needs it).
func (d *Device) DualSourceBlending() bool { return d.dualSource }

// Limits reports the buffer-size limits the device was created with.
func (d *Device) Limits() Limits { return d.limits }

// errorCallback receives (device, errorType, WGPUStringView message, ud1,
// ud2); the by-value StringView's ABI differs per OS (see svMessage).
var errorCallback = sync.OnceValue(func() uintptr {
	return ffi.Callback(func(_, errType, msgA, msgB, _, _ uintptr) uintptr {
		atomic.AddUint64(&errorCount, 1)
		fmt.Fprintf(os.Stderr, "gpu: uncaptured error (type %d): %s\n", errType, svMessage(msgA, msgB))
		return 0
	})
})

// errorCount counts the uncaptured device errors seen so far. wgpu reports
// a validation failure through this callback and still hands back a
// non-nil object, so a caller that can RECOVER — one building an optional
// pipeline from shaders it did not write — reads the count around the
// calls and treats a change as "that did not work" instead of using a
// broken object and taking the panic that follows.
var errorCount uint64

// ErrorCount returns the number of uncaptured device errors so far.
func ErrorCount() uint64 { return atomic.LoadUint64(&errorCount) }

// Device is the logical GPU connection; all resources come from it.
type Device struct {
	limits     Limits
	dualSource bool
	h          uintptr
	// staging counts bytes written through the queue since its last
	// submit (shared by every Queue wrapper of this device).
	staging *int
}

func (d *Device) Release() {
	FlushCallTrace() // a short run still reports its GFX_CALLS line
	wgpuDeviceRelease(d.h)
}

func (d *Device) Queue() *Queue {
	if d.staging == nil {
		d.staging = new(int)
	}
	return &Queue{h: wgpuDeviceGetQueue(d.h), pending: d.staging}
}

// CreateShaderModuleWGSL compiles a WGSL source string.
func (d *Device) CreateShaderModuleWGSL(label, src string) (*ShaderModule, error) {
	code := []byte(src)
	wgsl := shaderSourceWGSL{
		chain: chainedStruct{sType: sTypeShaderSourceWGSL},
		code:  strView(code),
	}
	desc := shaderModuleDescriptor{
		nextInChain: uintptr(unsafe.Pointer(&wgsl)),
	}
	h := wgpuDeviceCreateShaderModule(d.h, &desc)
	runtime.KeepAlive(code)
	runtime.KeepAlive(&wgsl)
	if h == 0 {
		return nil, fmt.Errorf("gpu: shader module %q failed to compile", label)
	}
	return &ShaderModule{h}, nil
}

// CreateShaderModuleSPIRV compiles a SPIR-V binary (little-endian words, as
// a .spv or the blob inside a Qt .qsb carries). WebGPU proper takes only
// WGSL, so this exists for shaders that ship compiled: a QML ShaderEffect's
// .qsb, whose GLSL was compiled by Qt's own tools.
func (d *Device) CreateShaderModuleSPIRV(label string, code []byte) (*ShaderModule, error) {
	if len(code) < 4 || len(code)%4 != 0 {
		return nil, fmt.Errorf("gpu: shader module %q: %d bytes is not SPIR-V", label, len(code))
	}
	words := make([]uint32, len(code)/4)
	for i := range words {
		words[i] = uint32(code[4*i]) | uint32(code[4*i+1])<<8 |
			uint32(code[4*i+2])<<16 | uint32(code[4*i+3])<<24
	}
	if words[0] != 0x07230203 {
		return nil, fmt.Errorf("gpu: shader module %q: bad SPIR-V magic %#x", label, words[0])
	}
	spirv := shaderSourceSPIRV{
		chain:    chainedStruct{sType: sTypeShaderSourceSPIRV},
		codeSize: uint32(len(words)),
		code:     uintptr(unsafe.Pointer(&words[0])),
	}
	desc := shaderModuleDescriptor{
		nextInChain: uintptr(unsafe.Pointer(&spirv)),
	}
	h := wgpuDeviceCreateShaderModule(d.h, &desc)
	runtime.KeepAlive(words)
	runtime.KeepAlive(&spirv)
	if h == 0 {
		return nil, fmt.Errorf("gpu: shader module %q failed to compile", label)
	}
	return &ShaderModule{h}, nil
}

// VertexAttr describes one vertex attribute within a buffer layout.
type VertexAttr struct {
	Format   VertexFormat
	Offset   int
	Location int
}

// VertexLayout describes one vertex buffer's stride and attributes.
type VertexLayout struct {
	Stride int
	Attrs  []VertexAttr
	// StepMode advances the buffer per vertex (the default) or per
	// instance, for a layout whose records describe a whole primitive
	// rather than a corner of one.
	StepMode VertexStepMode
}

// RenderPipelineDesc is the pipeline shape the 2D renderer needs: triangle
// list, optional vertex buffers, optional standard alpha blending, no
// depth, single color target.
type RenderPipelineDesc struct {
	Shader *ShaderModule
	// FragmentShader is the module the FRAGMENT stage comes from when the
	// two stages were compiled separately (a Qt .qsb is one file per
	// stage). nil means both stages live in Shader, as a WGSL module does.
	FragmentShader *ShaderModule
	VertexEntry    string
	FragmentEntry  string
	Format         TextureFormat
	VertexLayouts  []VertexLayout
	AlphaBlend     bool
	DualSrcBlend   bool       // dual-source (subpixel text): src1 is a per-channel mask
	Blend          *BlendDesc // an explicit blend equation; wins over AlphaBlend
	DepthTest      bool       // Depth32Float attachment, less-compare, write
	CullBack       bool
	Samples        int // MSAA sample count (0/1 = none); must match the pass target
	// Lines draws the index list as line segments (pairs) instead of
	// triangles: 1px lines for grids, axes and edge overlays.
	Lines bool
	// NoColorTarget runs the fragment stage with zero colour attachments:
	// a depth-only pass (shadow map) whose fragment shader only discards
	// (cut-out leaves, glyphs). FragmentEntry must then return nothing.
	NoColorTarget bool
}

func (d *Device) CreateRenderPipeline(rp *RenderPipelineDesc) (*RenderPipeline, error) {
	if rp.DualSrcBlend && !d.dualSource {
		// wgpu would log a validation error and hand back an invalid
		// pipeline that aborts at its first draw; an error here lets the
		// caller fall back (grayscale text).
		return nil, fmt.Errorf("gpu: dual-source blending is not available on this device")
	}
	vsEntry, fsEntry := []byte(rp.VertexEntry), []byte(rp.FragmentEntry)
	target := colorTargetState{
		format:    uint32(rp.Format),
		writeMask: colorWriteMaskAll,
	}
	var blend *blendState
	switch {
	case rp.Blend != nil:
		blend = &blendState{
			color: blendComponent{
				operation: uint32(rp.Blend.Color.Op),
				srcFactor: uint32(rp.Blend.Color.Src),
				dstFactor: uint32(rp.Blend.Color.Dst),
			},
			alpha: blendComponent{
				operation: uint32(rp.Blend.Alpha.Op),
				srcFactor: uint32(rp.Blend.Alpha.Src),
				dstFactor: uint32(rp.Blend.Alpha.Dst),
			},
		}
		target.blend = uintptr(unsafe.Pointer(blend))
	case rp.DualSrcBlend:
		// Subpixel text: result = src0*src1 + dst*(1-src1) per channel.
		blend = &blendState{
			color: blendComponent{operation: blendOpAdd, srcFactor: blendSrc1, dstFactor: blendOneMinusSrc1},
			alpha: blendComponent{operation: blendOpAdd, srcFactor: blendSrc1Alpha, dstFactor: blendOneMinusSrc1Alpha},
		}
		target.blend = uintptr(unsafe.Pointer(blend))
	case rp.AlphaBlend:
		blend = &blendState{
			color: blendComponent{operation: blendOpAdd, srcFactor: blendSrcAlpha, dstFactor: blendOneMinusSrcA},
			alpha: blendComponent{operation: blendOpAdd, srcFactor: blendFactorOne, dstFactor: blendOneMinusSrcA},
		}
		target.blend = uintptr(unsafe.Pointer(blend))
	}
	// Empty FragmentEntry = depth-only pipeline (shadow passes).
	var frag fragmentState
	if rp.FragmentEntry != "" {
		fragMod := rp.Shader
		if rp.FragmentShader != nil {
			fragMod = rp.FragmentShader
		}
		frag = fragmentState{
			module:      fragMod.h,
			entryPoint:  strView(fsEntry),
			targetCount: 1,
			targets:     uintptr(unsafe.Pointer(&target)),
		}
		if rp.NoColorTarget {
			frag.targetCount = 0
			frag.targets = 0
		}
	}

	// Lower the vertex layouts to their C mirrors; attrs holds every
	// buffer's attribute array so the pointers stay alive through the call.
	var layouts []vertexBufferLayout
	var attrs [][]vertexAttribute
	for _, vl := range rp.VertexLayouts {
		la := make([]vertexAttribute, len(vl.Attrs))
		for i, a := range vl.Attrs {
			la[i] = vertexAttribute{
				format:         uint32(a.Format),
				offset:         uint64(a.Offset),
				shaderLocation: uint32(a.Location),
			}
		}
		attrs = append(attrs, la)
		l := vertexBufferLayout{
			stepMode:       uint32(vl.StepMode),
			arrayStride:    uint64(vl.Stride),
			attributeCount: uintptr(len(la)),
		}
		if len(la) > 0 {
			l.attributes = uintptr(unsafe.Pointer(&la[0]))
		}
		layouts = append(layouts, l)
	}

	desc := renderPipelineDescriptor{
		vertex: vertexState{
			module:     rp.Shader.h,
			entryPoint: strView(vsEntry),
		},
		primitive:   primitiveState{topology: topologyTriangleList},
		multisample: multisampleState{count: msaaCount(rp.Samples), mask: ^uint32(0)},
	}
	if rp.FragmentEntry != "" {
		desc.fragment = uintptr(unsafe.Pointer(&frag))
	}
	if rp.CullBack {
		desc.primitive.cullMode = cullBack
	}
	if rp.Lines {
		desc.primitive.topology = topologyLineList
	}
	var depth *depthStencilState
	if rp.DepthTest {
		depth = &depthStencilState{
			format:            uint32(FormatDepth32Float),
			depthWriteEnabled: 1,
			depthCompare:      compareLess,
			stencilFront:      stencilFaceState{compareAlways, stencilOpKeep, stencilOpKeep, stencilOpKeep},
			stencilBack:       stencilFaceState{compareAlways, stencilOpKeep, stencilOpKeep, stencilOpKeep},
			stencilReadMask:   0xFFFFFFFF,
			stencilWriteMask:  0xFFFFFFFF,
		}
		desc.depthStencil = uintptr(unsafe.Pointer(depth))
	}
	if len(layouts) > 0 {
		desc.vertex.bufferCount = uintptr(len(layouts))
		desc.vertex.buffers = uintptr(unsafe.Pointer(&layouts[0]))
	}
	h := wgpuDeviceCreateRenderPipeline(d.h, &desc)
	runtime.KeepAlive(vsEntry)
	runtime.KeepAlive(fsEntry)
	runtime.KeepAlive(&target)
	runtime.KeepAlive(blend)
	runtime.KeepAlive(depth)
	runtime.KeepAlive(&frag)
	runtime.KeepAlive(layouts)
	runtime.KeepAlive(attrs)
	if h == 0 {
		return nil, fmt.Errorf("gpu: render pipeline creation failed")
	}
	return &RenderPipeline{h}, nil
}

func (d *Device) CreateCommandEncoder() *CommandEncoder {
	callEncoders.Add(1)
	return &CommandEncoder{wgpuDeviceCreateCommandEncoder(d.h, nil)}
}

type ShaderModule struct{ h uintptr }

func (m *ShaderModule) Release() { wgpuShaderModuleRelease(m.h) }

type RenderPipeline struct{ h uintptr }

func (p *RenderPipeline) Release() { wgpuRenderPipelineRelease(p.h) }

// Surface is a presentable window surface.
type Surface struct {
	h          uintptr
	configured bool
	alphaMode  uint32 // WGPUCompositeAlphaMode; 0 = Auto (opaque)
}

// SetTransparent selects a compositor alpha mode that honors the
// framebuffer's alpha channel — premultiplied when the surface supports it,
// else inherit (the X11 ARGB path). Call before Configure. Windows without
// a per-pixel-alpha swapchain keep opaque compositing.
func (s *Surface) SetTransparent(a *Adapter) {
	var caps surfaceCapabilities
	wgpuSurfaceGetCapabilities(s.h, a.h, &caps)
	const premultiplied, inherit = 2, 4
	if caps.alphaModeCount == 0 || caps.alphaModes == 0 {
		return
	}
	p := caps.alphaModes
	base := *(*unsafe.Pointer)(unsafe.Pointer(&p))
	modes := unsafe.Slice((*uint32)(base), caps.alphaModeCount)
	for _, want := range [...]uint32{premultiplied, inherit} {
		for _, m := range modes {
			if m == want {
				s.alphaMode = m
				return
			}
		}
	}
}

func (s *Surface) Release() {
	if s.configured {
		wgpuSurfaceUnconfigure(s.h)
	}
	wgpuSurfaceRelease(s.h)
}

// PreferredFormat picks a non-sRGB surface format (BGRA8Unorm, then
// RGBA8Unorm, then any non-sRGB, then formats[0]). The renderer works in
// sRGB-encoded color values throughout; an *-Srgb swap chain would re-encode
// them on write and brighten every mid-tone.
func (s *Surface) PreferredFormat(a *Adapter) TextureFormat {
	var caps surfaceCapabilities
	wgpuSurfaceGetCapabilities(s.h, a.h, &caps)
	if caps.formatCount == 0 || caps.formats == 0 {
		return TextureFormat(0)
	}
	// caps.formats is C-owned memory (a uintptr), not a Go pointer; the
	// indirection through &p is the idiom vet accepts for that case.
	p := caps.formats
	base := *(*unsafe.Pointer)(unsafe.Pointer(&p))
	formats := unsafe.Slice((*uint32)(base), caps.formatCount)
	const (
		fmtRGBA8Srgb = 0x17 // WGPUTextureFormat_RGBA8UnormSrgb
		fmtBGRA8     = 0x1B // WGPUTextureFormat_BGRA8Unorm
		fmtBGRA8Srgb = 0x1C // WGPUTextureFormat_BGRA8UnormSrgb
	)
	for _, want := range [...]uint32{fmtBGRA8, uint32(FormatRGBA8Unorm)} {
		for _, f := range formats {
			if f == want {
				return TextureFormat(f)
			}
		}
	}
	for _, f := range formats {
		if f != fmtRGBA8Srgb && f != fmtBGRA8Srgb {
			return TextureFormat(f)
		}
	}
	return TextureFormat(formats[0])
}

// Configure sizes the surface's swap chain. Call again after resize.
//
// wgpu rejects a zero-area surface, which a window reports while minimized or
// mid-resize; skip those so the frame is simply dropped (CurrentTexture then
// reports the swap chain unavailable) instead of panicking.
func (s *Surface) Configure(d *Device, width, height int, format TextureFormat) {
	if width <= 0 || height <= 0 {
		return
	}
	cfg := surfaceConfiguration{
		device:      d.h,
		format:      uint32(format),
		usage:       usageRenderAttachment,
		alphaMode:   s.alphaMode, // Auto unless SetTransparent picked one
		width:       uint32(width),
		height:      uint32(height),
		presentMode: presentMode(),
	}
	wgpuSurfaceConfigure(s.h, &cfg)
	s.configured = true
}

// CurrentTexture acquires this frame's texture. ok=false means the swap
// chain is outdated (resize) and the caller should reconfigure and retry.
func (s *Surface) CurrentTexture() (tex *Texture, ok bool) {
	var st surfaceTexture
	wgpuSurfaceGetCurrentTexture(s.h, &st)
	if debugLog {
		fmt.Fprintf(os.Stderr, "wgpu: acquire status=%d tex=%#x\n", st.status, st.texture)
	}
	// 1 = SuccessOptimal, 2 = SuccessSuboptimal; anything else is outdated/lost.
	if st.status != 1 && st.status != 2 {
		if st.texture != 0 {
			wgpuTextureRelease(st.texture)
		}
		return nil, false
	}
	return &Texture{h: st.texture, bpt: 4}, true
}

// Present shows the frame acquired by CurrentTexture. Order matters: call
// Present BEFORE releasing that texture — releasing first discards the
// acquired frame and the present fails (nothing reaches the screen; on DX12
// every later acquire then blocks out wgpu's 1s frame timeout).
func (s *Surface) Present() {
	st := wgpuSurfacePresent(s.h)
	if st != 1 { // WGPUStatus_Success
		presentFailWarn.Do(func() {
			fmt.Fprintf(os.Stderr, "gpu: wgpuSurfacePresent failed (status %d) — "+
				"frames are not reaching the screen; was the surface texture "+
				"released before Present? (set GFX_WGPU_LOG=warn for details)\n", st)
		})
	} else if debugLog {
		fmt.Fprintf(os.Stderr, "wgpu: present status=%d\n", st)
	}
}

var presentFailWarn sync.Once

type Texture struct {
	h uintptr
	// bpt is the texture's bytes per texel, so an upload knows its row
	// stride. It was width*4 everywhere until glyph sheets went to one
	// byte a texel.
	bpt int
}

func (t *Texture) Release() { wgpuTextureRelease(t.h) }

func (t *Texture) CreateView() *TextureView {
	return &TextureView{wgpuTextureCreateView(t.h, nil)}
}

type TextureView struct{ h uintptr }

func (v *TextureView) Release() { wgpuTextureViewRelease(v.h) }

type CommandEncoder struct{ h uintptr }

// BeginRenderPass starts a single-target pass that clears view to clear and
// stores the result.
func (e *CommandEncoder) BeginRenderPass(view *TextureView, clear Color) *RenderPass {
	callPasses.Add(1)
	att := renderPassColorAttachment{
		view:       view.h,
		depthSlice: depthSliceUndefined,
		loadOp:     loadOpClear,
		storeOp:    storeOpStore,
		clearValue: wgpuColor{clear.R, clear.G, clear.B, clear.A},
	}
	desc := renderPassDescriptor{
		colorAttachmentCount: 1,
		colorAttachments:     uintptr(unsafe.Pointer(&att)),
	}
	h := wgpuCommandEncoderBeginRenderPass(e.h, &desc)
	runtime.KeepAlive(&att)
	return &RenderPass{h}
}

// BeginRenderPassLoad starts a single-target pass that LOADS the existing
// contents of view instead of clearing — for compositing a second pass (e.g.
// UI) over an earlier one (e.g. a 3D scene) in the same frame.
func (e *CommandEncoder) BeginRenderPassLoad(view *TextureView) *RenderPass {
	callPasses.Add(1)
	att := renderPassColorAttachment{
		view:       view.h,
		depthSlice: depthSliceUndefined,
		loadOp:     loadOpLoad,
		storeOp:    storeOpStore,
	}
	desc := renderPassDescriptor{
		colorAttachmentCount: 1,
		colorAttachments:     uintptr(unsafe.Pointer(&att)),
	}
	h := wgpuCommandEncoderBeginRenderPass(e.h, &desc)
	runtime.KeepAlive(&att)
	return &RenderPass{h}
}

// BeginRenderPassResolve renders into a multisampled color view and resolves
// the result into resolve (a single-sample texture: the screenshot or
// swapchain frame). Use with pipelines built at the same sample count.
func (e *CommandEncoder) BeginRenderPassResolve(msaa, resolve *TextureView, clear Color) *RenderPass {
	callPasses.Add(1)
	att := renderPassColorAttachment{
		view:          msaa.h,
		resolveTarget: resolve.h,
		depthSlice:    depthSliceUndefined,
		loadOp:        loadOpClear,
		storeOp:       storeOpStore,
		clearValue:    wgpuColor{clear.R, clear.G, clear.B, clear.A},
	}
	desc := renderPassDescriptor{
		colorAttachmentCount: 1,
		colorAttachments:     uintptr(unsafe.Pointer(&att)),
	}
	h := wgpuCommandEncoderBeginRenderPass(e.h, &desc)
	runtime.KeepAlive(&att)
	return &RenderPass{h}
}

// BeginRenderPassResolveLoad is BeginRenderPassResolve that LOADS the
// multisampled attachment instead of clearing it — the persistent MSAA
// target keeps last frame, the pass repaints only the damaged part of it,
// and the resolve still writes the whole frame into resolve.
func (e *CommandEncoder) BeginRenderPassResolveLoad(msaa, resolve *TextureView) *RenderPass {
	callPasses.Add(1)
	att := renderPassColorAttachment{
		view:          msaa.h,
		resolveTarget: resolve.h,
		depthSlice:    depthSliceUndefined,
		loadOp:        loadOpLoad,
		storeOp:       storeOpStore,
	}
	desc := renderPassDescriptor{
		colorAttachmentCount: 1,
		colorAttachments:     uintptr(unsafe.Pointer(&att)),
	}
	h := wgpuCommandEncoderBeginRenderPass(e.h, &desc)
	runtime.KeepAlive(&att)
	return &RenderPass{h}
}

// msaaCount clamps a requested sample count to a valid pipeline value.
func msaaCount(n int) uint32 {
	if n >= 4 {
		return 4
	}
	return 1
}

// Finish ends recording and releases the encoder.
func (e *CommandEncoder) Finish() *CommandBuffer {
	h := wgpuCommandEncoderFinish(e.h, nil)
	wgpuCommandEncoderRelease(e.h)
	return &CommandBuffer{h}
}

type RenderPass struct{ h uintptr }

func (r *RenderPass) SetPipeline(p *RenderPipeline) {
	callPipe.Add(1)
	wgpuRenderPassEncoderSetPipeline(r.h, p.h)
}

func (r *RenderPass) Draw(vertexCount, instanceCount, firstVertex, firstInstance int) {
	callDraw.Add(1)
	wgpuRenderPassEncoderDraw(r.h, uint32(vertexCount), uint32(instanceCount), uint32(firstVertex), uint32(firstInstance))
}

// End closes and releases the pass encoder.
func (r *RenderPass) End() {
	wgpuRenderPassEncoderEnd(r.h)
	wgpuRenderPassEncoderRelease(r.h)
}

type CommandBuffer struct{ h uintptr }

type Queue struct {
	h       uintptr
	pending *int // bytes staged since the last submit (may be nil)
}

func (q *Queue) Release() { wgpuQueueRelease(q.h) }

// Submit sends command buffers to the GPU and releases them.
func (q *Queue) Submit(bufs ...*CommandBuffer) {
	handles := make([]uintptr, len(bufs))
	for i, b := range bufs {
		handles[i] = b.h
	}
	if q.pending != nil {
		*q.pending = 0
	}
	wgpuQueueSubmit(q.h, uintptr(len(handles)), &handles[0])
	runtime.KeepAlive(handles)
	noteSubmitTrace()
	for _, b := range bufs {
		wgpuCommandBufferRelease(b.h)
	}
}

// DebugAlphaModes lists the compositor alpha modes the surface supports
// (WGPUCompositeAlphaMode values) — a diagnostic for transparency issues.
func (s *Surface) DebugAlphaModes(a *Adapter) []uint32 {
	var caps surfaceCapabilities
	wgpuSurfaceGetCapabilities(s.h, a.h, &caps)
	if caps.alphaModeCount == 0 || caps.alphaModes == 0 {
		return nil
	}
	p := caps.alphaModes
	base := *(*unsafe.Pointer)(unsafe.Pointer(&p))
	return append([]uint32(nil), unsafe.Slice((*uint32)(base), caps.alphaModeCount)...)
}
