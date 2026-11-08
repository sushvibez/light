//go:build linux || darwin || freebsd || android

package ffi

import "github.com/ebitengine/purego"

func load(name string) (uintptr, error) {
	return purego.Dlopen(name, purego.RTLD_LAZY|purego.RTLD_GLOBAL)
}

func sym(handle uintptr, name string) (uintptr, error) {
	return purego.Dlsym(handle, name)
}

func unload(handle uintptr) error {
	return purego.Dlclose(handle)
}
