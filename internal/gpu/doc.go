// Vendored from github.com/sushvibez/gfx (gpu package, native backend,
// Apache-2.0), trimmed of its js/wasm files; Light uses the compute half
// through internal/wgpudev. light_compute.go holds Light's additions;
// the rest is gfx's code with only the ffi import path changed.
//
// Package gpu exposes WebGPU as the framework's one and only graphics API.
// WebGPU already is a portable hardware abstraction, so no extra HAL is
// layered underneath it — higher-level concepts (render graph, materials)
// live in the render module instead.
//
// Backends (see /DESIGN.md §5):
//   - Native (Windows/Linux/Android): wgpu-native shared library loaded
//     through the ffi module; bindings hand-mirrored from webgpu.h.
//   - Web: the browser's own WebGPU via syscall/js. Each encoder call is
//     one JS call today; batching command recording to amortize the
//     boundary cost is planned. Texture/buffer readback (screenshots,
//     compute results) is native-only so far.
//
// WGSL is the single shader language. Pipelines use auto-derived bind group
// layouts (Pipeline.BindGroupLayout); the WGSL-reflection codegen from
// /DESIGN.md §5 is not built yet. The inverse direction exists: gfx gen
// (script/gen) emits WGSL structs from Go GPU-shared component types.
package gpu
