package cpu

import "unsafe"

// unsafeSizeof returns sizeof(T) as uintptr. Pulled into its own file so the
// `unsafe` import stays visible and readable.
func unsafeSizeof[T any](_ T) uintptr {
	var z T
	return unsafe.Sizeof(z)
}

// unsafeSliceFromBytes reinterprets a []byte as []T of length `elems`.
func unsafeSliceFromBytes[T any](buf []byte, elems int) []T {
	if elems == 0 {
		return nil
	}
	return unsafe.Slice((*T)(unsafe.Pointer(&buf[0])), elems)
}
