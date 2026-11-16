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

// Idol's fused block kernels (src/idol.cu; ops/idol_fused.go checks the
// shapes and makes the inputs contiguous F32).
func init() {
	for op, fn := range map[string]registry.Kernel{
		"idol_ln_mod_f16":    idolLNModCUDA,
		"idol_head_f16":      idolHeadCUDA,
		"idol_gate_res_f32":  idolGateResCUDA,
		"idol_bias_gelu_f16": idolBiasGeluCUDA,
		"idol_silu_mul_f16":  idolSiLUMulCUDA,
	} {
		// Keyed by the dtype of the GEMM output they read (F16: the _h kernels).
		registry.Register(registry.Key{Op: op, Dev: device.KindCUDA, DType: dtype.F32}, fn)
		registry.Register(registry.Key{Op: op, Dev: device.KindCUDA, DType: dtype.F16}, fn)
	}
}

// idolPtr is t's device address, 0 for a nil (optional) tensor.
func idolPtr(t *tensor.Tensor) uintptr {
	if t == nil {
		return 0
	}
	return t.Storage().CUDAPtr()
}

// idolKernel binds out's device and returns kernel name, or its _h
// variant when in (the GEMM output it reads) is F16.
func idolKernel(out *tensor.Tensor, name string, in *tensor.Tensor) (*cudart.Function, error) {
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return nil, err
	}
	if in != nil && in.DType() == dtype.F16 {
		name += "_h"
	}
	return GetKernel(name)
}

func idolAttr(ctx *registry.Context, k string) int64 {
	v, _ := ctx.Attrs[k].(int)
	return int64(v)
}

func idolLNModCUDA(ctx *registry.Context) error {
	in, out := ctx.Inputs, ctx.Output
	if out.DType() != dtype.F16 {
		return fmt.Errorf("cuda idol_ln_mod_f16: out %v", out.DType())
	}
	fn, err := idolKernel(out, "idol_ln_mod_f16", nil)
	if err != nil {
		return err
	}
	rows, cols, rpb := idolAttr(ctx, "rows"), int32(idolAttr(ctx, "cols")), idolAttr(ctx, "rpb")
	eps64, _ := ctx.Attrs["eps"].(float64)
	eps := float32(eps64)
	x, w, b, sc, sh, o := idolPtr(in[0]), idolPtr(in[1]), idolPtr(in[2]), idolPtr(in[3]), idolPtr(in[4]), idolPtr(out)
	args := []unsafe.Pointer{unsafe.Pointer(&x), unsafe.Pointer(&w), unsafe.Pointer(&b), unsafe.Pointer(&sc),
		unsafe.Pointer(&sh), unsafe.Pointer(&o), unsafe.Pointer(&rows), unsafe.Pointer(&cols), unsafe.Pointer(&rpb),
		unsafe.Pointer(&eps)}
	if rows == 0 {
		return nil
	}
	return cudart.LaunchKernel(fn, uint32(rows), 1, 1, 256, 1, 1, 0, 0, args)
}

func idolHeadCUDA(ctx *registry.Context) error {
	in, out := ctx.Inputs, ctx.Output
	if out.DType() != dtype.F16 {
		return fmt.Errorf("cuda idol_head_f16: out %v", out.DType())
	}
	fn, err := idolKernel(out, "idol_head_f16", in[0])
	if err != nil {
		return err
	}
	rows, stride, off, heads := idolAttr(ctx, "rows"), idolAttr(ctx, "stride"), idolAttr(ctx, "off"), int32(idolAttr(ctx, "heads"))
	x, bias, rd, rb, fq, ax, g, o := idolPtr(in[0]), idolPtr(in[1]), idolPtr(in[2]), idolPtr(in[3]), idolPtr(in[4]), idolPtr(in[5]), idolPtr(in[6]), idolPtr(out)
	args := []unsafe.Pointer{unsafe.Pointer(&x), unsafe.Pointer(&stride), unsafe.Pointer(&off), unsafe.Pointer(&bias),
		unsafe.Pointer(&rd), unsafe.Pointer(&rb), unsafe.Pointer(&fq), unsafe.Pointer(&ax), unsafe.Pointer(&g),
		unsafe.Pointer(&o), unsafe.Pointer(&rows), unsafe.Pointer(&heads)}
	if rows == 0 {
		return nil
	}
	return Launch1D(fn, uint64(rows)*uint64(heads)*32, args)
}

func idolGateResCUDA(ctx *registry.Context) error {
	in, out := ctx.Inputs, ctx.Output
	fn, err := idolKernel(out, "idol_gate_res_f32", in[1])
	if err != nil {
		return err
	}
	rows, cols, rpb := idolAttr(ctx, "rows"), int32(idolAttr(ctx, "cols")), idolAttr(ctx, "rpb")
	x, y, b, g, o := idolPtr(in[0]), idolPtr(in[1]), idolPtr(in[2]), idolPtr(in[3]), idolPtr(out)
	args := []unsafe.Pointer{unsafe.Pointer(&x), unsafe.Pointer(&y), unsafe.Pointer(&b), unsafe.Pointer(&g),
		unsafe.Pointer(&o), unsafe.Pointer(&rows), unsafe.Pointer(&cols), unsafe.Pointer(&rpb)}
	n := uint64(rows) * uint64(cols)
	if n == 0 {
		return nil
	}
	return Launch1D(fn, (n+3)/4, args)
}

func idolBiasGeluCUDA(ctx *registry.Context) error {
	in, out := ctx.Inputs, ctx.Output
	fn, err := idolKernel(out, "idol_bias_gelu_f16", in[0])
	if err != nil {
		return err
	}
	n, cols := idolAttr(ctx, "n"), int32(idolAttr(ctx, "cols"))
	y, b, o := idolPtr(in[0]), idolPtr(in[1]), idolPtr(out)
	args := []unsafe.Pointer{unsafe.Pointer(&y), unsafe.Pointer(&b), unsafe.Pointer(&o), unsafe.Pointer(&n), unsafe.Pointer(&cols)}
	if n == 0 {
		return nil
	}
	return Launch1D(fn, uint64(n+3)/4, args)
}

func idolSiLUMulCUDA(ctx *registry.Context) error {
	in, out := ctx.Inputs, ctx.Output
	fn, err := idolKernel(out, "idol_silu_mul_f16", nil)
	if err != nil {
		return err
	}
	n, cols := idolAttr(ctx, "n"), int32(idolAttr(ctx, "cols"))
	g, u, bg, bu, o := idolPtr(in[0]), idolPtr(in[1]), idolPtr(in[2]), idolPtr(in[3]), idolPtr(out)
	args := []unsafe.Pointer{unsafe.Pointer(&g), unsafe.Pointer(&u), unsafe.Pointer(&bg), unsafe.Pointer(&bu),
		unsafe.Pointer(&o), unsafe.Pointer(&n), unsafe.Pointer(&cols)}
	if n == 0 {
		return nil
	}
	return Launch1D(fn, uint64(n+3)/4, args)
}
