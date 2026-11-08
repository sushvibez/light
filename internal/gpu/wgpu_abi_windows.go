//go:build windows

package gpu

import (
	"runtime"
	"unsafe"

	"light/internal/ffi"
)

// purego can't pass struct arguments on Windows. Fortunately the Win64
// calling convention passes any by-value struct larger than 8 bytes as a
// pointer to a caller-allocated copy — exactly what we do here, passing
// &callbackInfo as a uintptr. So these bind as pointer-arg functions.

func bindABI(lib *ffi.Lib) {
}

func issueRequestAdapter(instance, opts uintptr, cb reqCallbackInfo) uint64 {
	r := rawRequestAdapter(instance, opts, uintptr(unsafe.Pointer(&cb)))
	runtime.KeepAlive(&cb)
	return r
}

func issueRequestDevice(adapter, desc uintptr, cb reqCallbackInfo) uint64 {
	r := rawRequestDevice(adapter, desc, uintptr(unsafe.Pointer(&cb)))
	runtime.KeepAlive(&cb)
	return r
}

func issueBufferMapAsync(buffer uintptr, mode uint64, offset, size uintptr, cb bufferMapCallbackInfo) uint64 {
	r := rawBufferMapAsync(buffer, mode, offset, size, uintptr(unsafe.Pointer(&cb)))
	runtime.KeepAlive(&cb)
	return r
}
