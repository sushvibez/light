package device

import (
	"errors"
	"fmt"
)

// Kind identifies a broad device category.
type Kind uint8

const (
	KindCPU    Kind = 0
	KindCUDA   Kind = 1
	KindWebGPU Kind = 2
)

// String returns a short, human-readable name for the kind.
func (k Kind) String() string {
	switch k {
	case KindCPU:
		return "CPU"
	case KindCUDA:
		return "CUDA"
	case KindWebGPU:
		return "WebGPU"
	default:
		return fmt.Sprintf("Kind(%d)", k)
	}
}

// ErrCUDANotBuilt is returned by CUDA() when the process was built without
// the "cuda" build tag.
var ErrCUDANotBuilt = errors.New("device: CUDA support not compiled in (build with -tags cuda)")

// ErrWebGPUNotBuilt is returned by WebGPU() when the process was built with
// the "nowebgpu" tag. (Otherwise WebGPU is always compiled in and loads
// wgpu-native at run time, without cgo; WebGPU(0) reports a missing library.)
var ErrWebGPUNotBuilt = errors.New("device: WebGPU support not compiled in (built with -tags nowebgpu)")

// Device identifies a computation target. Implementations are comparable by
// Kind + Ordinal.
type Device interface {
	Kind() Kind
	Ordinal() int
	String() string
	// SameAs reports whether two Devices are logically identical. A generic
	// == comparison on interface values is unreliable because different
	// backends may return distinct pointer instances.
	SameAs(other Device) bool
}

// CPU returns the (process-wide, singleton) CPU device.
func CPU() Device { return cpuDevice{} }

type cpuDevice struct{}

func (cpuDevice) Kind() Kind     { return KindCPU }
func (cpuDevice) Ordinal() int   { return 0 }
func (cpuDevice) String() string { return "CPU" }
func (cpuDevice) SameAs(other Device) bool {
	return other != nil && other.Kind() == KindCPU
}

// IsCUDA reports whether the given device is a CUDA device.
func IsCUDA(d Device) bool { return d != nil && d.Kind() == KindCUDA }

// IsCPU reports whether the given device is the CPU device.
func IsCPU(d Device) bool { return d != nil && d.Kind() == KindCPU }

// IsWebGPU reports whether the given device is a WebGPU device.
func IsWebGPU(d Device) bool { return d != nil && d.Kind() == KindWebGPU }

// WebGPUAvailable reports whether WebGPU support is compiled in (every build
// except -tags nowebgpu). It does not load wgpu-native or probe for an
// adapter; call WebGPU(0) and inspect the error for that.
func WebGPUAvailable() bool { return webgpuCompiledIn }

// MustCUDA returns CUDA(ordinal) or panics. Convenience for example programs
// that must not silently fall back to CPU.
func MustCUDA(ordinal int) Device {
	d, err := CUDA(ordinal)
	if err != nil {
		panic(err)
	}
	return d
}

// CUDAAvailable reports whether a CUDA device can be constructed at all. This
// checks build-tag presence; it does not probe the driver. Callers who want
// driver-level checks should attempt CUDA(0) and inspect the error.
func CUDAAvailable() bool { return cudaCompiledIn }

// Auto returns the most capable available device: CUDA(0) if the cuda build
// tag is present and the driver works, then WebGPU(0) if wgpu-native loads
// and an adapter can be opened, otherwise CPU(). Use this in
// example programs and quick scripts where you want GPU when it's available
// without hard-failing on machines without one.
func Auto() Device {
	if cudaCompiledIn {
		if d, err := CUDA(0); err == nil {
			return d
		}
	}
	if webgpuCompiledIn {
		if d, err := WebGPU(0); err == nil {
			return d
		}
	}
	return CPU()
}
