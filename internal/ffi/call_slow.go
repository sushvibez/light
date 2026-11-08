//go:build !js && !windows && !((linux || android || darwin || freebsd) && (amd64 || arm64))

package ffi

import "github.com/ebitengine/purego"

// Platforms without a trampoline of their own go through purego.SyscallN:
// correct, one allocation per call.
//
//go:uintptrescapes
func Call(fn, a1, a2, a3, a4, a5, a6 uintptr) uintptr {
	r1, _, _ := purego.SyscallN(fn, a1, a2, a3, a4, a5, a6)
	return r1
}

// CallStack: the seventh argument onwards; SyscallN places them per ABI.
//
//go:uintptrescapes
func CallStack(fn, a1, a2, a3, a4, a5, a6 uintptr, stack []uintptr) uintptr {
	args := append([]uintptr{a1, a2, a3, a4, a5, a6}, stack...)
	r1, _, _ := purego.SyscallN(fn, args...)
	return r1
}
