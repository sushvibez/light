//go:build !nocuda && cgo

package cuda

import (
	"fmt"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "gated_delta_rule", Dev: device.KindCUDA, DType: dtype.F32}, gatedDeltaLauncher)
	registry.Register(registry.Key{Op: "causal_conv1d_depthwise", Dev: device.KindCUDA, DType: dtype.F32}, causalConv1dDepthwiseLauncher)
	registry.Register(registry.Key{Op: "softplus", Dev: device.KindCUDA, DType: dtype.F32}, softplusLauncher)
}

// gdMaxDK is GD_MAX_DK in gated_delta.cu (per-thread state column length).
const gdMaxDK = 256

// gdChunkD is GDC_D in gated_delta.cu: the chunked kernel's head dim.
const gdChunkD = 128

// gatedDeltaLauncher runs ops.GatedDeltaRule on CUDA. Chunked calls over
// several tokens with 128 × 128 heads (every Qwen3.5 size) use
// gated_delta_chunked_f32, which keeps the state in registers and walks the
// prompt 16 tokens at a time; everything else, decode included, uses the
// token-by-token gated_delta_recurrent_f32. Both compute the same function
// (the tests pin recurrent == chunked to 1e-5).
//
//	Inputs: q, k (B,T,H,dk), v (B,T,H,dv), g, beta (B,T,H), state (B,H,dk,dv),
//	        final (B,H,dk,dv) written
//	Output: (B,T,H,dv)
func gatedDeltaLauncher(ctx *registry.Context) error {
	q, k, v, g, beta, st, fin := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Inputs[3], ctx.Inputs[4], ctx.Inputs[5], ctx.Inputs[6]
	out := ctx.Output
	B, T, H, dk, dv := q.Dim(0), q.Dim(1), q.Dim(2), q.Dim(3), v.Dim(3)
	if ctx.Attrs["chunked"].(bool) && T > 1 && dk == gdChunkD && dv == gdChunkD {
		return gatedDeltaChunked(ctx, B, T, H)
	}
	if dk > gdMaxDK {
		return fmt.Errorf("cuda gated_delta_rule: key head dim %d > %d", dk, gdMaxDK)
	}
	if dv > 1024 {
		return fmt.Errorf("cuda gated_delta_rule: value head dim %d > 1024", dv)
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("gated_delta_recurrent_f32")
	if err != nil {
		return err
	}
	l2 := int32(0)
	if ctx.Attrs["l2norm"].(bool) {
		l2 = 1
	}
	qP, kP, vP := q.Storage().CUDAPtr(), k.Storage().CUDAPtr(), v.Storage().CUDAPtr()
	gP, bP := g.Storage().CUDAPtr(), beta.Storage().CUDAPtr()
	sP, fP, oP := st.Storage().CUDAPtr(), fin.Storage().CUDAPtr(), out.Storage().CUDAPtr()
	b32, t32, h32, dk32, dv32 := int32(B), int32(T), int32(H), int32(dk), int32(dv)
	args := []unsafe.Pointer{
		unsafe.Pointer(&qP), unsafe.Pointer(&kP), unsafe.Pointer(&vP), unsafe.Pointer(&gP), unsafe.Pointer(&bP),
		unsafe.Pointer(&sP), unsafe.Pointer(&fP), unsafe.Pointer(&oP),
		unsafe.Pointer(&b32), unsafe.Pointer(&t32), unsafe.Pointer(&h32), unsafe.Pointer(&dk32), unsafe.Pointer(&dv32),
		unsafe.Pointer(&l2),
	}
	block := uint32(dv)
	if block < 32 {
		block = 32 // a full warp; threads j ≥ dv only help stage q / k
	}
	shared := uint32(2 * dk * 4)
	return cudart.LaunchKernel(fn, uint32(B*H), 1, 1, block, 1, 1, shared, 0, args)
}

// gatedDeltaChunked launches gated_delta_chunked_f32: one 256-thread block
// per (batch, head), static shared memory only.
func gatedDeltaChunked(ctx *registry.Context, B, T, H int) error {
	q, k, v, g, beta, st, fin := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Inputs[3], ctx.Inputs[4], ctx.Inputs[5], ctx.Inputs[6]
	out := ctx.Output
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("gated_delta_chunked_f32")
	if err != nil {
		return err
	}
	l2 := int32(0)
	if ctx.Attrs["l2norm"].(bool) {
		l2 = 1
	}
	qP, kP, vP := q.Storage().CUDAPtr(), k.Storage().CUDAPtr(), v.Storage().CUDAPtr()
	gP, bP := g.Storage().CUDAPtr(), beta.Storage().CUDAPtr()
	sP, fP, oP := st.Storage().CUDAPtr(), fin.Storage().CUDAPtr(), out.Storage().CUDAPtr()
	b32, t32, h32 := int32(B), int32(T), int32(H)
	args := []unsafe.Pointer{
		unsafe.Pointer(&qP), unsafe.Pointer(&kP), unsafe.Pointer(&vP), unsafe.Pointer(&gP), unsafe.Pointer(&bP),
		unsafe.Pointer(&sP), unsafe.Pointer(&fP), unsafe.Pointer(&oP),
		unsafe.Pointer(&b32), unsafe.Pointer(&t32), unsafe.Pointer(&h32), unsafe.Pointer(&l2),
	}
	return cudart.LaunchKernel(fn, uint32(B*H), 1, 1, 256, 1, 1, 0, 0, args)
}

// causalConv1dDepthwiseLauncher: see ops.CausalConv1dDepthwise.
//
//	Inputs: x (B,C,T), w (C,K), state (B,C,K-1), new state (B,C,K-1) written
//	Output: (B,C,T)
func causalConv1dDepthwiseLauncher(ctx *registry.Context) error {
	x, w, st, nst, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Inputs[3], ctx.Output
	B, C, T, K := x.Dim(0), x.Dim(1), x.Dim(2), w.Dim(1)
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	convFn, err := GetKernel("causal_conv1d_depthwise_f32")
	if err != nil {
		return err
	}
	stateFn, err := GetKernel("causal_conv1d_state_f32")
	if err != nil {
		return err
	}
	silu := int32(0)
	if ctx.Attrs["silu"].(bool) {
		silu = 1
	}
	xP, wP, sP, nP, oP := x.Storage().CUDAPtr(), w.Storage().CUDAPtr(), st.Storage().CUDAPtr(), nst.Storage().CUDAPtr(), out.Storage().CUDAPtr()
	b32, c32, t32, k32 := int32(B), int32(C), int32(T), int32(K)
	if err := Launch1D(convFn, uint64(B*C*T), []unsafe.Pointer{
		unsafe.Pointer(&xP), unsafe.Pointer(&wP), unsafe.Pointer(&sP), unsafe.Pointer(&oP),
		unsafe.Pointer(&b32), unsafe.Pointer(&c32), unsafe.Pointer(&t32), unsafe.Pointer(&k32), unsafe.Pointer(&silu),
	}); err != nil {
		return err
	}
	if K <= 1 {
		return nil
	}
	return Launch1D(stateFn, uint64(B*C*(K-1)), []unsafe.Pointer{
		unsafe.Pointer(&xP), unsafe.Pointer(&sP), unsafe.Pointer(&nP),
		unsafe.Pointer(&b32), unsafe.Pointer(&c32), unsafe.Pointer(&t32), unsafe.Pointer(&k32),
	})
}

// softplusLauncher: unary softplus over a contiguous F32 tensor.
func softplusLauncher(ctx *registry.Context) error {
	a, out := ctx.Inputs[0], ctx.Output
	if !a.IsContiguous() || a.Offset() != 0 {
		return fmt.Errorf("cuda softplus: non-contiguous input not supported")
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("softplus_f32")
	if err != nil {
		return err
	}
	aP, oP := a.Storage().CUDAPtr(), out.Storage().CUDAPtr()
	n := int64(out.NumElements())
	return Launch1D(fn, uint64(n), []unsafe.Pointer{unsafe.Pointer(&aP), unsafe.Pointer(&oP), unsafe.Pointer(&n)})
}
