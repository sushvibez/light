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
	registry.Register(registry.Key{Op: "gated_delta_rule", Dev: device.KindWebGPU, DType: dtype.F32}, gatedDeltaF32)
	registry.Register(registry.Key{Op: "causal_conv1d_depthwise", Dev: device.KindWebGPU, DType: dtype.F32}, causalConv1dF32)
}

// gatedDeltaF32: ops.GatedDeltaRule on WebGPU, always the recurrent form
// (the chunked form computes the same function).
//
//	Inputs: q, k (B,T,H,dk), v (B,T,H,dv), g, beta (B,T,H), state, final (B,H,dk,dv)
//	Output: (B,T,H,dv)
func gatedDeltaF32(ctx *registry.Context) error {
	q, k, v, g, beta, st, fin := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Inputs[3], ctx.Inputs[4], ctx.Inputs[5], ctx.Inputs[6]
	B, T, H, dk, dv := q.Dim(0), q.Dim(1), q.Dim(2), q.Dim(3), v.Dim(3)
	if dk > 128 || dv > 128 {
		return fmt.Errorf("webgpu gated_delta_rule: head dims dk %d, dv %d (≤ 128 supported)", dk, dv)
	}
	d, err := dev()
	if err != nil {
		return err
	}
	l2 := uint32(0)
	if ctx.Attrs["l2norm"].(bool) {
		l2 = 1
	}
	params, err := d.Uniform(uint32(B*H), uint32(T), uint32(H), uint32(dk)|uint32(dv)<<8|l2<<16)
	if err != nil {
		return err
	}
	defer params.Release()
	return d.Run("gated_delta", shader("gated_delta"), []wgpudev.Binding{
		rd(raw(q)), rd(raw(k)), rd(raw(v)), rd(raw(g)), rd(raw(beta)), rd(raw(st)), rw(raw(fin)), rw(raw(ctx.Output)), uni(params),
	}, uint32(B*H), 1, 1)
}

// causalConv1dF32: ops.CausalConv1dDepthwise on WebGPU.
//
//	Inputs: x (B,C,T), w (C,K), state (B,C,K-1), new state (B,C,K-1) written
//	Output: (B,C,T)
func causalConv1dF32(ctx *registry.Context) error {
	x, w, st, nst := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Inputs[3]
	B, C, T, K := x.Dim(0), x.Dim(1), x.Dim(2), w.Dim(1)
	if K < 2 || K > 0xFFFF {
		return fmt.Errorf("webgpu causal_conv1d_depthwise: kernel size %d (2 … 65535 supported)", K)
	}
	d, err := dev()
	if err != nil {
		return err
	}
	silu := uint32(0)
	if ctx.Attrs["silu"].(bool) {
		silu = 1
	}
	params, err := d.Uniform(uint32(B*C), uint32(C), uint32(T), uint32(K)|silu<<16)
	if err != nil {
		return err
	}
	defer params.Release()
	return run1D(d, "causal_conv1d", []wgpudev.Binding{
		rd(raw(x)), rd(raw(w)), rd(raw(st)), rw(raw(nst)), rw(raw(ctx.Output)), uni(params),
	}, B*C)
}
