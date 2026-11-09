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
