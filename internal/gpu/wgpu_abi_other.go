//go:build !js && !windows

package gpu

import (
	"unsafe"

	"light/internal/ffi"
)

// The three wgpu request functions take a 40-byte callback-info struct by
// value, and the two SysV architectures pass such a struct differently:
// amd64 copies its five words onto the stack after the register
// arguments (ffi.CallStack), arm64 passes a pointer to a copy in the next
// integer register (callWithStruct, per arch). purego could marshal the
// struct on linux and darwin but refuses on android, where the ABI is the
// same; our own trampoline makes every SysV platform one code path.
// Windows uses a by-pointer variant (see wgpu_abi_windows.go): Win64
// passes large by-value structs by reference too.
var (
	addrRequestAdapter uintptr
	addrRequestDevice  uintptr
	addrBufferMapAsync uintptr
)

func bindABI(lib *ffi.Lib) {
	addrRequestAdapter = lib.MustSym("wgpuInstanceRequestAdapter")
	addrRequestDevice = lib.MustSym("wgpuAdapterRequestDevice")
	addrBufferMapAsync = lib.MustSym("wgpuBufferMapAsync")
}

func issueRequestAdapter(instance, opts uintptr, cb reqCallbackInfo) uint64 {
	return callWithStruct(addrRequestAdapter, []uintptr{instance, opts}, unsafe.Pointer(&cb), unsafe.Sizeof(cb))
}

func issueRequestDevice(adapter, desc uintptr, cb reqCallbackInfo) uint64 {
	return callWithStruct(addrRequestDevice, []uintptr{adapter, desc}, unsafe.Pointer(&cb), unsafe.Sizeof(cb))
}

func issueBufferMapAsync(buffer uintptr, mode uint64, offset, size uintptr, cb bufferMapCallbackInfo) uint64 {
	return callWithStruct(addrBufferMapAsync, []uintptr{buffer, uintptr(mode), offset, size}, unsafe.Pointer(&cb), unsafe.Sizeof(cb))
}
