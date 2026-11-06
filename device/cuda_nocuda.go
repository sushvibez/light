//go:build nocuda || !cgo

package device

const cudaCompiledIn = false

// CUDA returns ErrCUDANotBuilt on builds without the "cuda" tag.
func CUDA(ordinal int) (Device, error) {
	_ = ordinal
	return nil, ErrCUDANotBuilt
}

// CUDAComputeCapability returns ErrCUDANotBuilt on builds without CUDA.
func CUDAComputeCapability(ordinal int) (major, minor int, err error) {
	_ = ordinal
	return 0, 0, ErrCUDANotBuilt
}
