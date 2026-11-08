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
	registry.Register(registry.Key{Op: "conv2d", Dev: device.KindWebGPU, DType: dtype.F32}, conv2dF32)
	registry.Register(registry.Key{Op: "maxpool2d", Dev: device.KindWebGPU, DType: dtype.F32}, pool2dLauncher(0))
	registry.Register(registry.Key{Op: "avgpool2d", Dev: device.KindWebGPU, DType: dtype.F32}, pool2dLauncher(1))
	registry.Register(registry.Key{Op: "adaptive_avgpool2d", Dev: device.KindWebGPU, DType: dtype.F32}, adaptiveAvgPool2dF32)
	registry.Register(registry.Key{Op: "batchnorm_eval", Dev: device.KindWebGPU, DType: dtype.F32}, batchnormEvalF32)
}

func conv2dF32(ctx *registry.Context) error {
	in, w, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	stride := ctx.Attrs["stride"].([2]int)
	pad := ctx.Attrs["padding"].([2]int)
	if in.DType() != dtype.F32 || w.DType() != dtype.F32 {
		return fmt.Errorf("webgpu conv2d: F32 only (in=%v w=%v)", in.DType(), w.DType())
	}
	d, err := dev()
	if err != nil {
		return err
	}
	inD, wD, outD := in.Shape().Dims(), w.Shape().Dims(), out.Shape().Dims()
	biasBuf := raw(in) // dummy binding when there is no bias
	hasBias := uint32(0)
	if len(ctx.Inputs) > 2 && ctx.Inputs[2] != nil {
		if ctx.Inputs[2].DType() != dtype.F32 {
			return fmt.Errorf("webgpu conv2d: bias must be F32")
		}
		biasBuf = raw(ctx.Inputs[2])
		hasBias = 1
	}
	p0, err := d.Uniform(uint32(inD[0]), uint32(inD[1]), uint32(wD[0]), uint32(inD[2]))
	if err != nil {
		return err
	}
	defer p0.Release()
	p1, err := d.Uniform(uint32(inD[3]), uint32(wD[2]), uint32(wD[3]), uint32(stride[0]))
	if err != nil {
		return err
	}
	defer p1.Release()
	p2, err := d.Uniform(uint32(stride[1]), uint32(pad[0]), uint32(pad[1]), uint32(outD[2]))
	if err != nil {
		return err
	}
	defer p2.Release()
	p3, err := d.Uniform(uint32(outD[3]), hasBias, 0, 0)
	if err != nil {
		return err
	}
	defer p3.Release()
	n := out.NumElements()
	return run1D(d, "conv2d", []wgpudev.Binding{
		rd(raw(in)), rd(raw(w)), rd(biasBuf), rw(raw(out)), uni(p0), uni(p1), uni(p2), uni(p3),
	}, n)
}

func pool2dLauncher(mode uint32) registry.Kernel {
	return func(ctx *registry.Context) error {
		in, out := ctx.Inputs[0], ctx.Output
		k := ctx.Attrs["kernel"].([2]int)
		s := ctx.Attrs["stride"].([2]int)
		p := ctx.Attrs["padding"].([2]int)
		d, err := dev()
		if err != nil {
			return err
		}
		inD, outD := in.Shape().Dims(), out.Shape().Dims()
		p0, err := d.Uniform(uint32(inD[0]), uint32(inD[1]), uint32(inD[2]), uint32(inD[3]))
		if err != nil {
			return err
		}
		defer p0.Release()
		p1, err := d.Uniform(uint32(outD[2]), uint32(outD[3]), uint32(k[0]), uint32(k[1]))
		if err != nil {
			return err
		}
		defer p1.Release()
		p2, err := d.Uniform(uint32(s[0]), uint32(s[1]), uint32(p[0]), uint32(p[1]))
		if err != nil {
			return err
		}
		defer p2.Release()
		p3, err := d.Uniform(mode, 0, 0, 0)
		if err != nil {
			return err
		}
		defer p3.Release()
		return run1D(d, "pool2d", []wgpudev.Binding{rd(raw(in)), rw(raw(out)), uni(p0), uni(p1), uni(p2), uni(p3)}, out.NumElements())
	}
}

func adaptiveAvgPool2dF32(ctx *registry.Context) error {
	in, out := ctx.Inputs[0], ctx.Output
	d, err := dev()
	if err != nil {
		return err
	}
	inD, outD := in.Shape().Dims(), out.Shape().Dims()
	p0, err := d.Uniform(uint32(inD[0]), uint32(inD[1]), uint32(inD[2]), uint32(inD[3]))
	if err != nil {
		return err
	}
	defer p0.Release()
	p1, err := d.Uniform(uint32(outD[2]), uint32(outD[3]), 0, 0)
	if err != nil {
		return err
	}
	defer p1.Release()
	return run1D(d, "adaptive_avgpool2d", []wgpudev.Binding{rd(raw(in)), rw(raw(out)), uni(p0), uni(p1)}, out.NumElements())
}

// batchnormEvalF32 applies per-channel running statistics to a contiguous
// (N, C, H, W) input. Inputs: x, gamma, beta, running_mean, running_var.
// Attrs: "eps" float64, "xhat" bool (write (x-mean)/sigma without affine).
func batchnormEvalF32(ctx *registry.Context) error {
	x, g, b, rm, rv, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Inputs[3], ctx.Inputs[4], ctx.Output
	eps, _ := ctx.Attrs["eps"].(float64)
	xhat, _ := ctx.Attrs["xhat"].(bool)
	mode := uint32(0)
	if xhat {
		mode = 1
	}
	d, err := dev()
	if err != nil {
		return err
	}
	dims := x.Shape().Dims()
	C := dims[1]
	HW := dims[2] * dims[3]
	n := x.NumElements()
	if n == 0 {
		return nil
	}
	p0, err := d.Uniform(uint32(C), uint32(HW), f32bits(float32(eps)), mode)
	if err != nil {
		return err
	}
	defer p0.Release()
	p1, err := d.Uniform(uint32(n), 0, 0, 0)
	if err != nil {
		return err
	}
	defer p1.Release()
	return run1D(d, "batchnorm", []wgpudev.Binding{
		rd(raw(x)), rd(raw(g)), rd(raw(b)), rd(raw(rm)), rd(raw(rv)), rw(raw(out)), uni(p0), uni(p1),
	}, n)
}
