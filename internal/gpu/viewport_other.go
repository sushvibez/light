//go:build !js && !((linux || android || darwin || freebsd) && (amd64 || arm64))

package gpu

import "light/internal/ffi"

// Floats through purego.RegisterFunc where the trampoline has no float
// path (Windows, other architectures).
var wgpuRenderPassEncoderSetViewport func(pass uintptr, x, y, w, h, minDepth, maxDepth float32)

func bindResources(lib *ffi.Lib) {
	lib.Bind(&wgpuRenderPassEncoderSetViewport, "wgpuRenderPassEncoderSetViewport")
}

func setViewport(pass uintptr, x, y, w, h, minDepth, maxDepth float32) {
	wgpuRenderPassEncoderSetViewport(pass, x, y, w, h, minDepth, maxDepth)
}
