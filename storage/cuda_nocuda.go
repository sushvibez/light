//go:build nocuda || !cgo

package storage

// DrainCUDAPool is a no-op without CUDA.
func DrainCUDAPool() {}
