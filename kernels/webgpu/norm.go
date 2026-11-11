//go:build !nowebgpu

package webgpu

import (
	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/internal/wgpudev"
)

func init() {
	registry.Register(registry.Key{Op: "layernorm", Dev: device.KindWebGPU, DType: dtype.F32}, layernormF32)
	registry.Register(registry.Key{Op: "rmsnorm", Dev: device.KindWebGPU, DType: dtype.F32}, rmsnormF32)
	registry.Register(registry.Key{Op: "groupnorm", Dev: device.KindWebGPU, DType: dtype.F32}, groupnormF32)
}

// groupnormF32: ops.GroupNorm on (N, C, H, W) F32 (VAE / UNet blocks).
func groupnormF32(ctx *registry.Context) error {
	x, g, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	G, _ := ctx.Attrs["groups"].(int)
	eps, _ := ctx.Attrs["eps"].(float64)
	x, err := contig(x)
	if err != nil {
		return err
	}
	if g, err = contig(g); err != nil {
		return err
	}
	if b, err = contig(b); err != nil {
		return err
	}
	d, err := dev()
	if err != nil {
		return err
	}
	N, C, HW := x.Dim(0), x.Dim(1), x.Dim(2)*x.Dim(3)
	if N*C*HW == 0 {
		return nil
	}
	M := (C / G) * HW
	params, err := d.Uniform(uint32(M), uint32(HW), f32bits(float32(eps)), uint32(G))
	if err != nil {
		return err
	}
	defer params.Release()
	return runRows(d, "groupnorm", []wgpudev.Binding{rd(raw(x)), rd(raw(g)), rd(raw(b)), rw(raw(out)), uni(params)}, N*G)
}

func layernormF32(ctx *registry.Context) error {
	x, g, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	eps, _ := ctx.Attrs["eps"].(float64)
	x, err := contig(x)
	if err != nil {
		return err
	}
	g, err = contig(g)
	if err != nil {
		return err
	}
	b, err = contig(b)
	if err != nil {
		return err
	}
	d, err := dev()
	if err != nil {
		return err
	}
	cols := x.Dim(-1)
	if cols == 0 || x.NumElements() == 0 {
		return nil
	}
	rows := x.NumElements() / cols
	params, err := d.Uniform(uint32(cols), uint32(rows), f32bits(float32(eps)), 0)
	if err != nil {
		return err
	}
	defer params.Release()
	return runRows(d, "layernorm", []wgpudev.Binding{rd(raw(x)), rd(raw(g)), rd(raw(b)), rw(raw(out)), uni(params)}, rows)
}

func rmsnormF32(ctx *registry.Context) error {
	x, w, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	eps, _ := ctx.Attrs["eps"].(float64)
	x, err := contig(x)
	if err != nil {
		return err
	}
	w, err = contig(w)
	if err != nil {
		return err
	}
	d, err := dev()
	if err != nil {
		return err
	}
	cols := x.Dim(-1)
	if cols == 0 || x.NumElements() == 0 {
		return nil
	}
	rows := x.NumElements() / cols
	params, err := d.Uniform(uint32(cols), uint32(rows), f32bits(float32(eps)), 0)
	if err != nil {
		return err
	}
	defer params.Release()
	return runRows(d, "rmsnorm", []wgpudev.Binding{rd(raw(x)), rd(raw(w)), rw(raw(out)), uni(params)}, rows)
}
