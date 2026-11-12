//go:build !nocuda && cgo

package cuda

import (
	"fmt"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
	"light/tensor"
)

func init() {
	registry.Register(registry.Key{Op: "gemv_multi", Dev: device.KindCUDA, DType: dtype.BF16}, gemvMultiLauncher)
	registry.Register(registry.Key{Op: "gemv_gated", Dev: device.KindCUDA, DType: dtype.BF16}, gemvGatedLauncher)
}

const (
	gemvMaxSeg   = 4
	gemvWarps    = 8
	gemvMaxK     = 12288 // K floats of shared memory within the 48 KB default
	gemvSegBytes = 24    // sizeof(GemvSeg): two pointers, int n, int pad
)

// GemvSupported mirrors the kernel's limits for ops: K a multiple of 8
// (16-byte row loads) up to gemvMaxK, at most four weights.
func GemvSupported(K, nseg int) bool {
	return K%8 == 0 && K <= gemvMaxK && nseg >= 1 && nseg <= gemvMaxSeg
}

// gemvMultiLauncher: Inputs = [x (1, K) BF16, w₁..wₙ (Nᵢ, K) BF16];
// Attrs outs []*tensor.Tensor (one (1, Nᵢ) per weight, BF16 or all F32).
func gemvMultiLauncher(ctx *registry.Context) error {
	x, ws := ctx.Inputs[0], ctx.Inputs[1:]
	outs := ctx.Attrs["outs"].([]*tensor.Tensor)
	K := x.Dim(-1)
	if !GemvSupported(K, len(ws)) {
		return fmt.Errorf("cuda gemv_multi: K %d / %d weights unsupported", K, len(ws))
	}
	if err := cudart.SetDevice(x.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("gemv_bf16_multi")
	if err != nil {
		return err
	}
	// GemvSegs by value: 4 × {w ptr, y ptr, int32 n, int32 pad}.
	var segs [gemvMaxSeg * gemvSegBytes]byte
	total := 0
	for i, w := range ws {
		base := i * gemvSegBytes
		*(*uintptr)(unsafe.Pointer(&segs[base])) = devPtr(w)
		*(*uintptr)(unsafe.Pointer(&segs[base+8])) = devPtr(outs[i])
		*(*int32)(unsafe.Pointer(&segs[base+16])) = int32(w.Dim(0))
		total += w.Dim(0)
	}
	xP := devPtr(x)
	nseg, k32 := int32(len(ws)), int32(K)
	outF32 := int32(0)
	if outs[0].DType() == dtype.F32 {
		outF32 = 1
	}
	args := []unsafe.Pointer{unsafe.Pointer(&xP), unsafe.Pointer(&segs[0]), unsafe.Pointer(&nseg), unsafe.Pointer(&k32), unsafe.Pointer(&outF32)}
	grid := uint32((total + gemvWarps - 1) / gemvWarps)
	return cudart.LaunchKernel(fn, grid, 1, 1, 32*gemvWarps, 1, 1, uint32(K*4), 0, args)
}

// gemvGatedLauncher: Inputs = [x (1, K), wg, wu (N, K)] BF16, Output (1, N)
// BF16; Attrs gelu bool.
func gemvGatedLauncher(ctx *registry.Context) error {
	x, wg, wu, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	K, N := x.Dim(-1), wg.Dim(0)
	if !GemvSupported(K, 1) {
		return fmt.Errorf("cuda gemv_gated: K %d unsupported", K)
	}
	if err := cudart.SetDevice(x.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("gemv_bf16_gated")
	if err != nil {
		return err
	}
	xP, gP, uP, yP := devPtr(x), devPtr(wg), devPtr(wu), devPtr(out)
	n32, k32 := int32(N), int32(K)
	gelu := int32(0)
	if ctx.Attrs["gelu"].(bool) {
		gelu = 1
	}
	args := []unsafe.Pointer{unsafe.Pointer(&xP), unsafe.Pointer(&gP), unsafe.Pointer(&uP), unsafe.Pointer(&yP), unsafe.Pointer(&n32), unsafe.Pointer(&k32), unsafe.Pointer(&gelu)}
	grid := uint32((N + gemvWarps - 1) / gemvWarps)
	return cudart.LaunchKernel(fn, grid, 1, 1, 32*gemvWarps, 1, 1, uint32(K*4), 0, args)
}
