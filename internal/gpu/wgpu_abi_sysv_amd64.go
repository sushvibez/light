//go:build !js && !windows && amd64

package gpu

import (
	"unsafe"

	"light/internal/ffi"
)

// callWithStruct passes a by-value struct the System V amd64 way: a
// struct larger than two words is copied onto the stack, word by word,
// after the register arguments.
func callWithStruct(fn uintptr, regs []uintptr, st unsafe.Pointer, size uintptr) uint64 {
	var a [6]uintptr
	copy(a[:], regs)
	words := unsafe.Slice((*uintptr)(st), size/unsafe.Sizeof(uintptr(0)))
	return uint64(ffi.CallStack(fn, a[0], a[1], a[2], a[3], a[4], a[5], words))
}
