package ffi

import "syscall"

// Windows never needed cgo for this: the stdlib loads DLLs natively, and
// purego.RegisterFunc works with any function address.

func load(name string) (uintptr, error) {
	h, err := syscall.LoadLibrary(name)
	return uintptr(h), err
}

func sym(handle uintptr, name string) (uintptr, error) {
	return syscall.GetProcAddress(syscall.Handle(handle), name)
}

func unload(handle uintptr) error {
	return syscall.FreeLibrary(syscall.Handle(handle))
}
