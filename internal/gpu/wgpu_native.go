//go:build !js

package gpu

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"unsafe"

	"light/internal/ffi"
)

// This file is the phase-0 risk gate (DESIGN.md §11): prove that wgpu-native
// works through purego — library load, function binding, and C→Go request
// callbacks — before any renderer is built on top.
//
// Pinned release: wgpu-native v29.0.x (wgpu-core 29). The modern webgpu.h ABI
// uses WGPUStringView for strings, by-value callback-info structs + Futures
// for async requests, and 1-based enums. purego marshals the by-value structs
// (struct_amd64.go/struct_arm64.go) and a callback receives a StringView as
// two scalar register args (data, length).

// stringView mirrors WGPUStringView. Pass {data, length}; length is the byte
// count (wgpu accepts an explicit length, no NUL needed).
type stringView struct {
	data   uintptr
	length uintptr
}

// strView makes a stringView over a Go []byte's storage (keep the bytes alive
// with runtime.KeepAlive across the call).
func strView(b []byte) stringView {
	if len(b) == 0 {
		return stringView{}
	}
	return stringView{data: uintptr(unsafe.Pointer(&b[0])), length: uintptr(len(b))}
}

// reqCallbackInfo mirrors WGPURequest{Adapter,Device}CallbackInfo (40 bytes):
// nextInChain, mode (WGPUCallbackMode), callback, userdata1, userdata2.
type reqCallbackInfo struct {
	nextInChain uintptr
	mode        uint32
	_           uint32
	callback    uintptr
	userdata1   uintptr
	userdata2   uintptr
}

const callbackModeAllowSpontaneous = 3 // WGPUCallbackMode_AllowSpontaneous

// DefaultLibNames returns the platform spellings of the wgpu-native shared
// library, for use with ffi.Open.
func DefaultLibNames() []string {
	switch runtime.GOOS {
	case "windows":
		return []string{"wgpu_native.dll"}
	case "darwin":
		return []string{"libwgpu_native.dylib"}
	default:
		return []string{"libwgpu_native.so"}
	}
}

func bindAll(lib *ffi.Lib) {
	bindABI(lib) // request fns: struct-by-value (SysV) vs by-pointer (Win64)
}

// adapterInfo mirrors WGPUAdapterInfo: StringView fields, then ids. Strings
// are wgpu-owned until wgpuAdapterInfoFreeMembers.
type adapterInfo struct {
	nextInChain     uintptr
	vendor          stringView
	architecture    stringView
	device          stringView
	description     stringView
	backendType     uint32
	adapterType     uint32
	vendorID        uint32
	deviceID        uint32
	subgroupMinSize uint32
	subgroupMaxSize uint32
}

var backendNames = map[uint32]string{
	0: "undefined", 1: "null", 2: "webgpu", 3: "d3d11",
	4: "d3d12", 5: "metal", 6: "vulkan", 7: "opengl", 8: "opengles",
}

var adapterTypeNames = map[uint32]string{
	0: "discrete-gpu", 1: "integrated-gpu", 2: "cpu", 3: "unknown",
}

// request carries one wgpu request callback's results. Adapter/device
// requests are serialized by the lock; with AllowSpontaneous mode the native
// backend resolves synchronously (and we ProcessEvents to be safe).
var request struct {
	sync.Mutex
	status  uintptr
	handle  uintptr
	message string
}

// svMessage decodes a by-value WGPUStringView callback argument. SysV
// (linux/android) passes the 16-byte struct as two register args (data,
// length); Win64 passes a pointer to a caller-allocated temporary, which
// lands in one arg (the next declared arg is then the following real
// parameter — safe to ignore when only the message is wanted).
func svMessage(a, b uintptr) string {
	if runtime.GOOS == "windows" {
		if a == 0 {
			return ""
		}
		// a is a C-owned pointer (never a Go pointer); the indirection
		// through &a is the idiom vet accepts for that case.
		sv := (*stringView)(*(*unsafe.Pointer)(unsafe.Pointer(&a)))
		return goStringView(sv.data, sv.length)
	}
	return goStringView(a, b)
}

// requestCallback is shared by adapter and device requests. The callback
// signature is (status, handle, WGPUStringView message, userdata1,
// userdata2); the by-value StringView's ABI differs per OS (see svMessage).
var requestCallback = sync.OnceValue(func() uintptr {
	return ffi.Callback(func(status, handle, msgA, msgB, _, _ uintptr) uintptr {
		request.status = status
		request.handle = handle
		request.message = svMessage(msgA, msgB)
		return 0
	})
})

// ProbeInfo describes the adapter/device pair Probe acquired.
type ProbeInfo struct {
	WGPUVersion string
	Adapter     string
	Vendor      string
	Driver      string
	Backend     string
	AdapterType string
	VendorID    uint32
	DeviceID    uint32
}

func (p *ProbeInfo) String() string {
	return fmt.Sprintf("wgpu %s: %s (%s, %s, %s, vendor=0x%04x device=0x%04x, driver %s)",
		p.WGPUVersion, p.Adapter, p.AdapterType, p.Backend, p.Vendor, p.VendorID, p.DeviceID, p.Driver)
}

var (
	loadOnce sync.Once
	loadErr  error
)

// The typed stubs in zffi_gen.go are generated from the remaining
// `lib.Bind(&x, "sym")` declarations: after adding a symbol, bind it the
// usual way and run `go generate`; ffigen moves every integer-only
// signature onto the reflection-free ffi.Call trampoline.
//
//go:generate go run ../ffi/cmd/ffigen -dir .

// Load opens wgpu-native and binds every symbol the gpu package uses. The
// first call wins; subsequent calls return its result. With no paths the
// platform default library names are tried.
func Load(libPaths ...string) error {
	loadOnce.Do(func() {
		if len(libPaths) == 0 {
			libPaths = DefaultLibNames()
		}
		lib, err := ffi.Open(libPaths...)
		if err != nil {
			loadErr = err
			return
		}
		bindGenerated(lib)
		bindAll(lib)
		bindResources(lib)
		bindLight(lib)
		initLogging(lib)
	})
	return loadErr
}

var logLevels = map[string]uint32{
	"error": 1, "warn": 2, "info": 3, "debug": 4, "trace": 5,
}

// debugLog is set when GFX_WGPU_LOG is active; hot paths gate their stderr
// diagnostics on it.
var debugLog bool

// initLogging routes wgpu-native's log to stderr when GFX_WGPU_LOG is set
// (error|warn|info|debug|trace). The callback's WGPUStringView parameter is
// passed by value: on Win64 a 16-byte struct arrives as a pointer to a
// temporary; on SysV (linux) it arrives as two register args.
func initLogging(lib *ffi.Lib) {
	level, ok := logLevels[os.Getenv("GFX_WGPU_LOG")]
	if !ok {
		return
	}
	debugLog = true
	var setLogCallback func(cb uintptr, userdata uintptr)
	var setLogLevel func(level uint32)
	lib.Bind(&setLogCallback, "wgpuSetLogCallback")
	lib.Bind(&setLogLevel, "wgpuSetLogLevel")
	names := map[uintptr]string{1: "ERROR", 2: "WARN", 3: "INFO", 4: "DEBUG", 5: "TRACE"}
	cb := ffi.Callback(func(level, msgA, msgB, _ uintptr) uintptr {
		fmt.Fprintf(os.Stderr, "wgpu[%s] %s\n", names[level], svMessage(msgA, msgB))
		return 0
	})
	setLogCallback(cb, 0)
	setLogLevel(level)
}

// Probe loads wgpu-native from the first path that opens, then acquires and
// releases an adapter, device and queue headlessly (no window, no surface).
func Probe(libPaths ...string) (*ProbeInfo, error) {
	if err := Load(libPaths...); err != nil {
		return nil, err
	}

	v := wgpuGetVersion()
	info := &ProbeInfo{
		WGPUVersion: fmt.Sprintf("%d.%d.%d.%d", v>>24, v>>16&0xff, v>>8&0xff, v&0xff),
	}

	instance := createInstance()
	if instance == 0 {
		return nil, fmt.Errorf("gpu: wgpuCreateInstance returned nil")
	}
	defer wgpuInstanceRelease(instance)

	adapterOpts := &requestAdapterOptions{
		featureLevel:    featureLevelCore,
		powerPreference: powerPreferenceHighPerformance,
	}
	adapter, err := await(instance, func(cb reqCallbackInfo) {
		issueRequestAdapter(instance, uintptr(unsafe.Pointer(adapterOpts)), cb)
	}, "request adapter")
	runtime.KeepAlive(adapterOpts)
	if err != nil {
		return nil, err
	}
	defer wgpuAdapterRelease(adapter)

	var ai adapterInfo
	wgpuAdapterGetInfo(adapter, &ai)
	info.Adapter = goStringView(ai.device.data, ai.device.length)
	info.Vendor = goStringView(ai.vendor.data, ai.vendor.length)
	info.Driver = goStringView(ai.description.data, ai.description.length)
	info.Backend = backendNames[ai.backendType]
	info.AdapterType = adapterTypeNames[ai.adapterType]
	info.VendorID = ai.vendorID
	info.DeviceID = ai.deviceID

	device, err := await(instance, func(cb reqCallbackInfo) {
		issueRequestDevice(adapter, 0, cb)
	}, "request device")
	if err != nil {
		return nil, err
	}
	defer wgpuDeviceRelease(device)

	queue := wgpuDeviceGetQueue(device)
	if queue == 0 {
		return nil, fmt.Errorf("gpu: wgpuDeviceGetQueue returned nil")
	}
	wgpuQueueRelease(queue)

	return info, nil
}

// await issues a wgpu request with a spontaneous-mode callback and returns
// the produced handle. ProcessEvents flushes the native backend's resolution.
func await(instance uintptr, issue func(cb reqCallbackInfo), what string) (uintptr, error) {
	request.Lock()
	defer request.Unlock()
	request.status, request.handle, request.message = ^uintptr(0), 0, ""
	issue(reqCallbackInfo{mode: callbackModeAllowSpontaneous, callback: requestCallback()})
	wgpuInstanceProcessEvents(instance)
	if request.status == ^uintptr(0) {
		return 0, fmt.Errorf("gpu: %s: callback never fired", what)
	}
	if request.status != 1 || request.handle == 0 { // 1 = WGPU*Status_Success
		return 0, fmt.Errorf("gpu: %s failed (status %d): %s", what, request.status, request.message)
	}
	return request.handle, nil
}

// goStringView copies a WGPUStringView (pointer + byte length). The pointer
// is C-owned, never a Go pointer, so the uintptr round-trip is safe.
func goStringView(data, length uintptr) string {
	if data == 0 || length == 0 {
		return ""
	}
	base := *(*unsafe.Pointer)(unsafe.Pointer(&data))
	return string(unsafe.Slice((*byte)(base), int(length)))
}
