//go:build webgpu

package device

import (
	"fmt"

	"light/internal/wgpudev"
)

const webgpuCompiledIn = true

// webgpuDevice is a WebGPU (wgpu-native) target. Only ordinal 0 exists: the
// backend opens a single high-performance adapter per process.
type webgpuDevice struct {
	ordinal int
}

func (d webgpuDevice) Kind() Kind     { return KindWebGPU }
func (d webgpuDevice) Ordinal() int   { return d.ordinal }
func (d webgpuDevice) String() string { return fmt.Sprintf("WebGPU(%d)", d.ordinal) }
func (d webgpuDevice) SameAs(other Device) bool {
	return other != nil && other.Kind() == KindWebGPU && other.Ordinal() == d.ordinal
}

// WebGPU returns a device handle for the given WebGPU adapter ordinal. Only
// ordinal 0 is supported. The adapter and logical device are opened on the
// first call; an error is returned when no adapter is available.
func WebGPU(ordinal int) (Device, error) {
	if ordinal != 0 {
		return nil, fmt.Errorf("device: WebGPU ordinal %d out of range (have 1 device)", ordinal)
	}
	if _, err := wgpudev.Get(); err != nil {
		return nil, fmt.Errorf("device: %w", err)
	}
	return webgpuDevice{ordinal: ordinal}, nil
}
