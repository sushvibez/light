//go:build webgpu

package ops

// Blank-import the WebGPU kernels so their init() registers with the op
// dispatcher and installs the tensor Contiguous / CopyFrom hooks for
// device.KindWebGPU.
import _ "light/kernels/webgpu"
