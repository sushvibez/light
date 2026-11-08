//go:build !js && !windows && arm64

package gpu

import (
	"runtime"
	"unsafe"

	"light/internal/ffi"
)

// callWithStruct passes a by-value struct the AAPCS64 way: a struct
// larger than 16 bytes goes by reference — the caller makes a copy and
// passes its address in the next integer register.
func callWithStruct(fn uintptr, regs []uintptr, st unsafe.Pointer, size uintptr) uint64 {
	var a [6]uintptr
	n := copy(a[:], regs)
	a[n] = uintptr(st)
	r := ffi.Call(fn, a[0], a[1], a[2], a[3], a[4], a[5])
	runtime.KeepAlive(st)
	return uint64(r)
}
