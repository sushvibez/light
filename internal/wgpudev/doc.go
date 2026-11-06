// Package wgpudev owns the process-wide WebGPU (wgpu-native) instance,
// adapter, logical device and queue, plus the buffer pool and the shader /
// pipeline caches shared by every WebGPU kernel. It is the WebGPU analogue
// of internal/cudart: storage, tensor and kernels/webgpu all talk to the GPU
// through this package, and nothing outside it imports the wgpu bindings.
//
// The whole package is behind the opt-in "webgpu" build tag so default
// builds never link wgpu-native.
package wgpudev
