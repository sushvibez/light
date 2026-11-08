package ffi

import "syscall"

// Windows: the standard library's SyscallN is already a direct,
// allocation-free native call.
func Call(fn, a1, a2, a3, a4, a5, a6 uintptr) uintptr {
	r1, _, _ := syscall.SyscallN(fn, a1, a2, a3, a4, a5, a6)
	return r1
}

// CallStack: the seventh argument onwards; Win64 puts everything after
// the fourth on the stack, which SyscallN does.
func CallStack(fn, a1, a2, a3, a4, a5, a6 uintptr, stack []uintptr) uintptr {
	args := append([]uintptr{a1, a2, a3, a4, a5, a6}, stack...)
	r1, _, _ := syscall.SyscallN(fn, args...)
	return r1
}
