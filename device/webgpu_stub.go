//go:build nowebgpu

package device

const webgpuCompiledIn = false

// WebGPU returns ErrWebGPUNotBuilt on builds without the "webgpu" tag.
func WebGPU(ordinal int) (Device, error) {
	_ = ordinal
	return nil, ErrWebGPUNotBuilt
}
