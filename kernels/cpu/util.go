package cpu

import (
	"unsafe"

	"light/tensor"
)

// asSlice reinterprets the CPU buffer behind tensor t as []T. T must match
// the tensor's dtype. Callers are kernel implementations — no type-checking
// here; the dispatcher guarantees the match.
func asSlice[T any](t *tensor.Tensor) []T {
	buf := t.Storage().CPUBytes()
	if len(buf) == 0 {
		return nil
	}
	var zero T
	elemSize := int(unsafe.Sizeof(zero))
	return unsafe.Slice((*T)(unsafe.Pointer(&buf[0])), len(buf)/elemSize)
}
