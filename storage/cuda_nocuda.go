//go:build nocuda || !cgo

package storage

// DrainCUDAPool is a no-op without CUDA.
func DrainCUDAPool() {}

// SetCUDAAllocObserver is a no-op without CUDA (no CUDA storage is ever
// allocated); it returns nil.
func SetCUDAAllocObserver(fn func(*Storage)) func(*Storage) { return nil }

// CUDAPtr panics without CUDA: no storage lives on a CUDA device.
func (s *Storage) CUDAPtr() uintptr { panic("storage.CUDAPtr: CUDA support not compiled in") }

// CUDAPoolStats reports zeros without CUDA.
func CUDAPoolStats() (forcedGCs, oomRetries, evictions int64) { return 0, 0, 0 }
