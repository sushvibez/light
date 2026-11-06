//go:build !nocuda && cgo

package device

import (
	"fmt"

	"light/internal/cudart"
)

const cudaCompiledIn = true

// cudaDevice is a CUDA target identified by an ordinal (0 = first GPU).
type cudaDevice struct {
	ordinal int
}

func (d cudaDevice) Kind() Kind     { return KindCUDA }
func (d cudaDevice) Ordinal() int   { return d.ordinal }
func (d cudaDevice) String() string { return fmt.Sprintf("CUDA(%d)", d.ordinal) }
func (d cudaDevice) SameAs(other Device) bool {
	return other != nil && other.Kind() == KindCUDA && other.Ordinal() == d.ordinal
}

// CUDA returns a device handle for the given GPU ordinal, verifying it's in
// range against cudaGetDeviceCount().
func CUDA(ordinal int) (Device, error) {
	if ordinal < 0 {
		return nil, fmt.Errorf("device: CUDA ordinal must be >= 0, got %d", ordinal)
	}
	n, err := cudart.DeviceCount()
	if err != nil {
		return nil, fmt.Errorf("device: %w", err)
	}
	if ordinal >= n {
		return nil, fmt.Errorf("device: CUDA ordinal %d out of range (have %d device(s))", ordinal, n)
	}
	return cudaDevice{ordinal: ordinal}, nil
}

// CUDAComputeCapability returns the compute capability (major, minor) of the
// GPU at ordinal, e.g. 8, 0 for an A100 or 8, 9 for an L4 / RTX 40-series.
func CUDAComputeCapability(ordinal int) (major, minor int, err error) {
	p, err := cudart.GetDeviceProperties(ordinal)
	if err != nil {
		return 0, 0, fmt.Errorf("device: %w", err)
	}
	return p.ComputeMajor, p.ComputeMinor, nil
}
