//go:build !nowebgpu

package webgpu

import (
	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/internal/wgpudev"
)

func init() {
	registry.Register(registry.Key{Op: "upsample_nearest", Dev: device.KindWebGPU, DType: dtype.F32}, upsampleNearest)
}

// upsampleNearest: ops.UpsampleNearest on (N, C, H, W) F32 (VAE decoders).
func upsampleNearest(ctx *registry.Context) error {
	x, out := ctx.Inputs[0], ctx.Output
	x, err := contig(x)
	if err != nil {
		return err
	}
	d, err := dev()
	if err != nil {
		return err
	}
	n := out.NumElements()
	if n == 0 {
		return nil
	}
	params, err := d.Uniform(uint32(out.Dim(2)), uint32(out.Dim(3)), uint32(x.Dim(2)), uint32(x.Dim(3)))
	if err != nil {
		return err
	}
	defer params.Release()
	return run1D(d, "upsample_nearest", []wgpudev.Binding{rd(raw(x)), rw(raw(out)), uni(params)}, n)
}

func init() {
	registry.Register(registry.Key{Op: "pad", Dev: device.KindWebGPU, DType: dtype.F32}, padF32)
}

// padF32: ops.Pad (zeros along one axis).
func padF32(ctx *registry.Context) error {
	x, out := ctx.Inputs[0], ctx.Output
	dim, _ := ctx.Attrs["dim"].(int)
	left, _ := ctx.Attrs["left"].(int)
	x, err := contig(x)
	if err != nil {
		return err
	}
	d, err := dev()
	if err != nil {
		return err
	}
	n := out.NumElements()
	if n == 0 {
		return nil
	}
	inner := 1
	for _, s := range out.Shape().Dims()[dim+1:] {
		inner *= s
	}
	params, err := d.Uniform(uint32(out.Dim(dim)), uint32(inner), uint32(left), uint32(x.Dim(dim)))
	if err != nil {
		return err
	}
	defer params.Release()
	return run1D(d, "pad", []wgpudev.Binding{rd(raw(x)), rw(raw(out)), uni(params)}, n)
}
