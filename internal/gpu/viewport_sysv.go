//go:build (linux || android || darwin || freebsd) && (amd64 || arm64)

package gpu

import "light/internal/ffi"

// wgpuRenderPassEncoderSetViewport takes six floats: the one wgpu entry
// with floating-point arguments. On the SysV ABI they ride in the vector
// registers, which ffi.CallF fills, so this platform has no call left on
// purego.RegisterFunc.
var addrSetViewport uintptr

func bindResources(lib *ffi.Lib) {
	addrSetViewport = lib.MustSym("wgpuRenderPassEncoderSetViewport")
}

func setViewport(pass uintptr, x, y, w, h, minDepth, maxDepth float32) {
	ffi.CallF(addrSetViewport, pass, 0, 0, 0, 0, 0,
		[]uint64{ffi.F32(x), ffi.F32(y), ffi.F32(w), ffi.F32(h), ffi.F32(minDepth), ffi.F32(maxDepth)})
}
