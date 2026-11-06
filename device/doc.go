// Package device abstracts execution locations. A Device identifies where a
// tensor lives and where its ops run: CPU() or CUDA(ordinal).
//
// Builds without the "cuda" build tag still import this package; CUDA(...)
// returns ErrCUDANotBuilt. This keeps higher-level packages backend-agnostic
// while still allowing CUDA capabilities to be compiled out entirely.
package device
