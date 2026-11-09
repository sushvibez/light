//go:build !nowebgpu

package webgpu

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/internal/wgpudev"
	"light/shape"
)

var binaryOp = map[string]uint32{"add": 0, "sub": 1, "mul": 2, "div": 3, "pow": 4}

var unaryOp = map[string]uint32{
	"relu": 0, "sigmoid": 1, "tanh": 2, "neg": 3, "exp": 4, "log": 5, "sqrt": 6,
	"gelu": 7, "silu": 8, "abs": 9, "sin": 10, "cos": 11, "tan": 12,
	"floor": 13, "ceil": 14, "round": 15, "softplus": 16, "gelu_erf": 17,
}

func init() {
	for op, code := range binaryOp {
		registry.Register(registry.Key{Op: op, Dev: device.KindWebGPU, DType: dtype.F32}, makeBinary(op, code))
	}
	for op, code := range unaryOp {
		registry.Register(registry.Key{Op: op, Dev: device.KindWebGPU, DType: dtype.F32}, makeUnary(op, code))
	}
}

// makeBinary mirrors the CUDA launcher: a same-shape contiguous fast path,
// and a broadcast/strided path driven by a shape+stride meta buffer.
func makeBinary(op string, code uint32) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
		d, err := dev()
		if err != nil {
			return err
		}
		n := out.NumElements()
		if n == 0 {
			return nil
		}
		if a.Shape().Equal(out.Shape()) && b.Shape().Equal(out.Shape()) &&
			a.IsContiguous() && b.IsContiguous() && a.Offset() == 0 && b.Offset() == 0 {
			params, err := d.Uniform(code, uint32(n), 0, 0)
			if err != nil {
				return err
			}
			defer params.Release()
			return run1D(d, "binary", []wgpudev.Binding{rd(raw(a)), rd(raw(b)), rw(raw(out)), uni(params)}, n)
		}
		aStr, err := shape.BroadcastStrides(a.Shape(), a.Strides(), out.Shape())
		if err != nil {
			return fmt.Errorf("webgpu %s: %w", op, err)
		}
		bStr, err := shape.BroadcastStrides(b.Shape(), b.Strides(), out.Shape())
		if err != nil {
			return fmt.Errorf("webgpu %s: %w", op, err)
		}
		dims := out.Shape().Dims()
		rank := len(dims)
		meta, err := d.StorageTemp(i32Bytes(dims, aStr.Slice(), bStr.Slice()))
		if err != nil {
			return err
		}
		defer meta.Release()
		params, err := d.Uniform(code, uint32(n), uint32(rank), 0)
		if err != nil {
			return err
		}
		defer params.Release()
		params2, err := d.Uniform(uint32(a.Offset()), uint32(b.Offset()), 0, 0)
		if err != nil {
			return err
		}
		defer params2.Release()
		return run1D(d, "binary_strided", []wgpudev.Binding{
			rd(raw(a)), rd(raw(b)), rw(raw(out)), rdp(meta), uni(params), uni(params2),
		}, n)
	}
}

func makeUnary(op string, code uint32) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, out := ctx.Inputs[0], ctx.Output
		if !a.Shape().Equal(out.Shape()) {
			return fmt.Errorf("webgpu %s: shape mismatch %v vs %v", op, a.Shape(), out.Shape())
		}
		a, err := contig(a)
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
		params, err := d.Uniform(code, uint32(n), 0, 0)
		if err != nil {
			return err
		}
		defer params.Release()
		return run1D(d, "unary", []wgpudev.Binding{rd(raw(a)), rw(raw(out)), uni(params)}, n)
	}
}
