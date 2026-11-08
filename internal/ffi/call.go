//go:build !js

package ffi

import (
	"math"
	"unsafe"
)

// Call is the fast path for integer-only native calls: up to six integer or
// pointer arguments (the register-passed set on every 64-bit ABI we target),
// one integer result. It is what generated stubs (ffigen) use. The slow
// path for anything else — floats, by-value structs, more arguments — stays
// purego.RegisterFunc via Bind.
//
// Pointers passed through Call must stay alive and unmoved for the call:
// callers pass unsafe.Pointer(&x) converted to uintptr in the call
// expression itself (the uintptrescapes rule), or runtime.KeepAlive.
//
// callArgs is the block the trampoline reads; its layout is shared with the
// assembly, so the offsets are asserted below.
type callArgs struct {
	fn    uintptr
	a     [6]uintptr
	r1    uintptr
	r2    uintptr
	stack [maxStack]uintptr // words placed on the stack, in order, for CallStack
	f     [8]uint64         // the eight floating-point argument registers, for CallF
}

// maxStack is how many stack words CallStack passes: enough for one
// by-value struct of the size WebGPU's callback-info blocks are (40
// bytes), after the register arguments.
const maxStack = 8

// A wrong offset is a constant index out of range: it fails to compile.
var (
	_ = [1]struct{}{}[unsafe.Offsetof(callArgs{}.fn)-0]
	_ = [1]struct{}{}[unsafe.Offsetof(callArgs{}.a)-8]
	_ = [1]struct{}{}[unsafe.Offsetof(callArgs{}.r1)-56]
	_ = [1]struct{}{}[unsafe.Offsetof(callArgs{}.r2)-64]
	_ = [1]struct{}{}[unsafe.Offsetof(callArgs{}.stack)-72]
	_ = [1]struct{}{}[unsafe.Offsetof(callArgs{}.f)-136]
	_ = [1]struct{}{}[unsafe.Sizeof(callArgs{})-200]
)

// F32 is a float32 as the register word CallF passes it in: the bits in
// the low half, which is where a single-precision argument sits in XMM0
// on amd64 and S0 on arm64.
func F32(v float32) uint64 { return uint64(math.Float32bits(v)) }

// F64 is a float64 as the register word CallF passes it in.
func F64(v float64) uint64 { return math.Float64bits(v) }
