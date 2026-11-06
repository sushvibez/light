// Package webgpu registers WebGPU (WGSL) kernels with internal/registry
// under device.KindWebGPU. It is only compiled with the opt-in "webgpu"
// build tag; without it the package is empty and no wgpu-native code is
// linked.
//
// Coverage is F32-only (WGSL has no f64/i64 arithmetic). Ops without a WGSL
// kernel are simply not registered, so dispatch returns
// registry.ErrKernelNotFound — there is no silent CPU fallback here.
package webgpu
