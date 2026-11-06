//go:build nocuda || !cgo

package cudart

import "errors"

// ErrNotBuilt is returned by every cudart entry point in a -tags nocuda
// build so that example programs can call them unconditionally and just
// skip GPU-only reporting.
var ErrNotBuilt = errors.New("cudart: CUDA support not compiled in (built with -tags nocuda)")

// MemInfo reports free/total device memory; always fails in nocuda builds.
func MemInfo() (free, total uintptr, err error) { return 0, 0, ErrNotBuilt }
