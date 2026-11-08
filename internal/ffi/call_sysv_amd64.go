//go:build (linux || android || darwin || freebsd) && amd64

package ffi

import (
	"unsafe"
	_ "unsafe" // for go:linkname

	_ "github.com/ebitengine/purego" // sets up the (fake) cgo runtime
)

// runtime.cgocall switches to the g0 stack, tells the scheduler we are in
// a syscall, and calls fn(arg) with the C convention. It is the same entry
// purego uses; the difference is that our declaration is go:noescape, so
// the argument block stays on the Go stack: zero allocations per call.
//
//go:noescape
//go:linkname cgocall runtime.cgocall
func cgocall(fn uintptr, arg unsafe.Pointer) int32

// call6ABI0 holds the address of the assembly trampoline (call_amd64.s).
var call6ABI0 uintptr

//go:uintptrescapes
func Call(fn, a1, a2, a3, a4, a5, a6 uintptr) uintptr {
	c := callArgs{fn: fn, a: [6]uintptr{a1, a2, a3, a4, a5, a6}}
	cgocall(call6ABI0, unsafe.Pointer(&c))
	return c.r1
}

// CallStack is Call with up to eight further integer arguments — the
// seventh onwards — placed where the ABI puts them: on amd64 all on the
// stack, on arm64 the first two in X6/X7 and the rest on the stack. The
// words are laid out whether or not the callee reads them. (A by-value
// struct is NOT the same on the two: amd64 copies one larger than two
// words onto the stack as these words, arm64 passes a pointer to it in
// the next integer register — see gpu's request calls.)
//
//go:uintptrescapes
func CallStack(fn, a1, a2, a3, a4, a5, a6 uintptr, stack []uintptr) uintptr {
	c := callArgs{fn: fn, a: [6]uintptr{a1, a2, a3, a4, a5, a6}}
	copy(c.stack[:], stack)
	cgocall(call6ABI0, unsafe.Pointer(&c))
	return c.r1
}

// CallF is Call with up to eight floating-point arguments in the vector
// registers (XMM0–7 / D0–7), given as F32 / F64 words, in the order the
// signature lists its float parameters — the SysV convention counts
// integer and float arguments separately.
//
//go:uintptrescapes
func CallF(fn, a1, a2, a3, a4, a5, a6 uintptr, floats []uint64) uintptr {
	c := callArgs{fn: fn, a: [6]uintptr{a1, a2, a3, a4, a5, a6}}
	copy(c.f[:], floats)
	cgocall(call6ABI0, unsafe.Pointer(&c))
	return c.r1
}
