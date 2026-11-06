package tensor

import "light/device"

// ContigFn copies src (arbitrary strides/offset) into dst (freshly
// allocated, contiguous, same shape and dtype) on a non-CPU device. Each
// accelerator backend registers one at init so tensor.Contiguous can run
// on-device without a host round-trip.
type ContigFn func(dst, src *Tensor) error

// CUDAContigFn is the historical name of ContigFn, kept for the CUDA
// kernel package.
type CUDAContigFn = ContigFn

var contigFns = map[device.Kind]ContigFn{}

// RegisterContig installs the strided-copy implementation for a device
// kind. Exported for cross-package plumbing only.
func RegisterContig(kind device.Kind, fn ContigFn) { contigFns[kind] = fn }

// RegisterCUDAContig is called from kernels/cuda at init. It is a thin
// wrapper over RegisterContig(device.KindCUDA, fn).
func RegisterCUDAContig(fn CUDAContigFn) { RegisterContig(device.KindCUDA, fn) }
