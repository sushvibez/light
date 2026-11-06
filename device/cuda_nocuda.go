//go:build nocuda || !cgo

package device

const cudaCompiledIn = false

// CUDA returns ErrCUDANotBuilt on builds without the "cuda" tag.
func CUDA(ordinal int) (Device, error) {
	_ = ordinal
	return nil, ErrCUDANotBuilt
}
