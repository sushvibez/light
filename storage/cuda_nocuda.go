//go:build nocuda

package storage

// DrainCUDAPool is a no-op without CUDA.
func DrainCUDAPool() {}
