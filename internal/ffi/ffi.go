//go:build !js

package ffi

import (
	"fmt"

	"github.com/ebitengine/purego"
)

// Lib is a loaded native shared library.
type Lib struct {
	handle uintptr
	name   string
}

// Open loads the first of names that can be found, so callers can list
// platform spellings and soname fallbacks in one call:
//
//	ffi.Open("libwgpu_native.so", "wgpu_native.dll")
func Open(names ...string) (*Lib, error) {
	var firstErr error
	for _, n := range names {
		h, err := load(n)
		if err == nil {
			return &Lib{handle: h, name: n}, nil
		}
		if firstErr == nil {
			firstErr = err
		}
	}
	return nil, fmt.Errorf("ffi: cannot open any of %v: %w", names, firstErr)
}

// Name returns the name the library was opened under.
func (l *Lib) Name() string { return l.name }

// Sym resolves a symbol address.
func (l *Lib) Sym(name string) (uintptr, error) {
	addr, err := sym(l.handle, name)
	if err != nil {
		return 0, fmt.Errorf("ffi: %s: symbol %q: %w", l.name, name, err)
	}
	return addr, nil
}

// Bind makes *fptr (a pointer to a Go func variable) call the native
// function named name. Supported signatures follow purego.RegisterFunc:
// integer, float, pointer and string arguments. It panics if the symbol is
// missing or the signature is unsupported — binding errors are programming
// errors, caught the first time the code runs on a platform.
func (l *Lib) Bind(fptr any, name string) {
	addr, err := l.Sym(name)
	if err != nil {
		panic(err)
	}
	purego.RegisterFunc(fptr, addr)
}

// Close unloads the library. Bound functions must not be called afterwards.
func (l *Lib) Close() error {
	return unload(l.handle)
}

// Callback wraps a Go function as a native function pointer that C code can
// call (purego.NewCallback). The function must use integer-sized arguments
// and at most one integer return. Callbacks are never freed, so create them
// once at init time, not per call.
func Callback(fn any) uintptr {
	return purego.NewCallback(fn)
}

// MustSym resolves a symbol address or panics; generated stubs use it.
func (l *Lib) MustSym(name string) uintptr {
	addr, err := l.Sym(name)
	if err != nil {
		panic(err)
	}
	return addr
}
