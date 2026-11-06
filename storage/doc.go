// Package storage owns the raw byte buffers that back tensors. A Storage
// bundles a device-specific buffer, its dtype, and its length, and attaches a
// runtime.AddCleanup callback that releases the backing memory (cudaFree on
// CUDA; GC-tracked slice on CPU).
//
// Storage is refcounted so that views (Reshape, Permute, Slice) can share
// underlying memory without preventing cleanup. The refcount is atomic; the
// cleanup fires when it reaches zero.
package storage
