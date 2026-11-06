//go:build webgpu

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
