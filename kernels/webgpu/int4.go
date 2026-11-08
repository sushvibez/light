//go:build !nowebgpu

package webgpu

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/internal/wgpudev"
)

func init() {
	registry.Register(registry.Key{Op: "int4_dequant", Dev: device.KindWebGPU, DType: dtype.F32}, int4DequantF32)
	registry.Register(registry.Key{Op: "int4_gemv", Dev: device.KindWebGPU, DType: dtype.F32}, int4GEMVF32)
}

// int4DequantF32 expands quant.Int4 weights (codes U32 (N, K/8), scales /
// zeros F32 (N, K/G)) to F32 (N, K), at element offset attr "offset".
func int4DequantF32(ctx *registry.Context) error {
	codes, s, z, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	N, K := codes.Dim(0), codes.Dim(1)*8
	G := ctx.Attrs["group"].(int)
	off, _ := ctx.Attrs["offset"].(int)
	d, err := dev()
	if err != nil {
		return err
	}
	params, err := d.Uniform(uint32(N), uint32(K), uint32(G), uint32(off))
	if err != nil {
		return err
	}
	defer params.Release()
	return run1D(d, "int4_dequant", []wgpudev.Binding{rd(raw(codes)), rd(raw(s)), rd(raw(z)), rw(raw(out)), uni(params)}, N*K)
}

// int4GEMVF32 is the fused decode matmul: x (M ≤ 8, K) F32 times the 4-bit
// weight (N, K) transposed, into out (M, N).
func int4GEMVF32(ctx *registry.Context) error {
	x, codes, s, z, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Inputs[3], ctx.Output
	M, K, N := x.Dim(0), x.Dim(1), codes.Dim(0)
	G := ctx.Attrs["group"].(int)
	if M > 8 || K%8 != 0 || G%8 != 0 {
		return fmt.Errorf("webgpu int4_gemv: M %d (≤ 8), K %d and group %d (multiples of 8)", M, K, G)
	}
	d, err := dev()
	if err != nil {
		return err
	}
	x, err = contig(x)
	if err != nil {
		return err
	}
	params, err := d.Uniform(uint32(M), uint32(N), uint32(K), uint32(G))
	if err != nil {
		return err
	}
	defer params.Release()
	return runRows(d, "int4_gemv", []wgpudev.Binding{rd(raw(x)), rd(raw(codes)), rd(raw(s)), rd(raw(z)), rw(raw(out)), uni(params)}, N)
}
