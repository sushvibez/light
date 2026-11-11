package ops

import (
	"fmt"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// GatedDeltaRuleFused is GatedDeltaRule fed straight from Gated DeltaNet's
// mixed projection and raw gate projections (transformers'
// Qwen3_5GatedDeltaNet after the short conv):
//
//	qkv    (B, T, C) F32, C = 2·Hk·dk + Hv·dv: q heads, k heads, v heads
//	a, b   (B, T, Hv) F32 or BF16: g = negA · softplus(a + dtBias), beta = σ(b)
//	dtBias, negA (Hv,) F32
//	state  (B, Hv, dk, dv) F32 or nil
//
// Value head h reads key head h / (Hv/Hk) (tiled = false, HF's
// repeat_interleave) or h mod Hk (tiled = true, llama.cpp's order). The
// result is GatedDeltaRule's: out (B, T, Hv, dv) and the final state.
//
// On a device with the fused kernel (CUDA) a single launch does the head
// split, the gate math and the recurrence, which on a decode step replaces
// three strided copies, two casts and four element-wise launches per layer.
// Elsewhere, and for the chunked prefill form, it runs those ops and
// GatedDeltaRule, so every device computes the same function.
func GatedDeltaRuleFused(qkv, a, b, dtBias, negA, state *tensor.Tensor, Hk, dk, Hv, dv int, tiled bool, opts GatedDeltaOptions) (out, final *tensor.Tensor, err error) {
	if qkv.Rank() != 3 || qkv.DType() != dtype.F32 {
		return nil, nil, fmt.Errorf("GatedDeltaRuleFused: qkv must be (B, T, C) F32, got %v %v", qkv.Shape(), qkv.DType())
	}
	B, T, C := qkv.Dim(0), qkv.Dim(1), qkv.Dim(2)
	if Hk <= 0 || Hv%Hk != 0 || C != 2*Hk*dk+Hv*dv {
		return nil, nil, fmt.Errorf("GatedDeltaRuleFused: C = %d does not match Hk %d · dk %d, Hv %d · dv %d", C, Hk, dk, Hv, dv)
	}
	for name, t := range map[string]*tensor.Tensor{"a": a, "b": b} {
		if t.Rank() != 3 || t.Dim(0) != B || t.Dim(1) != T || t.Dim(2) != Hv {
			return nil, nil, fmt.Errorf("GatedDeltaRuleFused: %s must be (B, T, Hv) = (%d, %d, %d), got %v", name, B, T, Hv, t.Shape())
		}
		if t.DType() != a.DType() || (t.DType() != dtype.F32 && t.DType() != dtype.BF16) {
			return nil, nil, fmt.Errorf("GatedDeltaRuleFused: a and b must both be F32 or BF16, got %v / %v", a.DType(), b.DType())
		}
	}
	for name, t := range map[string]*tensor.Tensor{"dtBias": dtBias, "negA": negA} {
		if t.Rank() != 1 || t.Dim(0) != Hv || t.DType() != dtype.F32 {
			return nil, nil, fmt.Errorf("GatedDeltaRuleFused: %s must be (Hv,) F32, got %v %v", name, t.Shape(), t.DType())
		}
	}
	dev := qkv.Device()
	chunk := opts.ChunkSize
	if chunk <= 0 {
		chunk = 64
	}
	chunked := opts.Mode == GatedDeltaChunked || (opts.Mode == GatedDeltaAuto && T > 1)
	kern, err := registry.Lookup(registry.Key{Op: "gated_delta_rule_fused", Dev: dev.Kind(), DType: dtype.F32})
	if err != nil || chunked {
		return gatedDeltaUnfused(qkv, a, b, dtBias, negA, state, Hk, dk, Hv, dv, tiled, opts)
	}
	ins := []*tensor.Tensor{qkv, a, b, dtBias, negA}
	for i, t := range ins {
		if !t.Device().SameAs(dev) {
			return nil, nil, fmt.Errorf("GatedDeltaRuleFused: input %d on %v, want %v", i, t.Device(), dev)
		}
		if ins[i], err = ensureContig(t); err != nil {
			return nil, nil, err
		}
	}
	if state == nil {
		if state, err = tensor.Zeros(dtype.F32, []int{B, Hv, dk, dv}, dev); err != nil {
			return nil, nil, err
		}
	} else {
		if state.Rank() != 4 || state.Dim(0) != B || state.Dim(1) != Hv || state.Dim(2) != dk || state.Dim(3) != dv || state.DType() != dtype.F32 {
			return nil, nil, fmt.Errorf("GatedDeltaRuleFused: state must be (B, Hv, dk, dv) F32 = (%d, %d, %d, %d), got %v %v", B, Hv, dk, dv, state.Shape(), state.DType())
		}
		if state, err = ensureContig(state); err != nil {
			return nil, nil, err
		}
	}
	if out, err = tensor.Empty(dtype.F32, []int{B, T, Hv, dv}, dev); err != nil {
		return nil, nil, err
	}
	if final, err = tensor.Empty(dtype.F32, []int{B, Hv, dk, dv}, dev); err != nil {
		return nil, nil, err
	}
	if err := kern(&registry.Context{
		Inputs: append(ins, state, final),
		Output: out,
		Attrs:  map[string]any{"hk": Hk, "dk": dk, "hv": Hv, "dv": dv, "tiled": tiled, "l2norm": opts.QKL2Norm, "chunk": chunk},
	}); err != nil {
		return nil, nil, err
	}
	return out, final, nil
}

// gatedDeltaUnfused is GatedDeltaRuleFused out of the existing ops: the
// reference for the fused kernel and the path for devices without it.
func gatedDeltaUnfused(qkv, a, b, dtBias, negA, state *tensor.Tensor, Hk, dk, Hv, dv int, tiled bool, opts GatedDeltaOptions) (*tensor.Tensor, *tensor.Tensor, error) {
	B, T := qkv.Dim(0), qkv.Dim(1)
	split := func(off, heads, dim int) (*tensor.Tensor, error) {
		x, err := qkv.Narrow(2, off, heads*dim)
		if err != nil {
			return nil, err
		}
		if x, err = x.Contiguous(); err != nil {
			return nil, err
		}
		return x.Reshape(B, T, heads, dim)
	}
	q, err := split(0, Hk, dk)
	if err != nil {
		return nil, nil, err
	}
	k, err := split(Hk*dk, Hk, dk)
	if err != nil {
		return nil, nil, err
	}
	v, err := split(2*Hk*dk, Hv, dv)
	if err != nil {
		return nil, nil, err
	}
	if r := Hv / Hk; r > 1 {
		if q, err = repeatHeads(q, r, tiled); err != nil {
			return nil, nil, err
		}
		if k, err = repeatHeads(k, r, tiled); err != nil {
			return nil, nil, err
		}
	}
	toF32 := func(x *tensor.Tensor) (*tensor.Tensor, error) {
		if x.DType() == dtype.F32 {
			return x, nil
		}
		return Cast(x, dtype.F32)
	}
	if b, err = toF32(b); err != nil {
		return nil, nil, err
	}
	beta, err := Sigmoid(b)
	if err != nil {
		return nil, nil, err
	}
	if a, err = toF32(a); err != nil {
		return nil, nil, err
	}
	if a, err = Add(a, dtBias); err != nil {
		return nil, nil, err
	}
	if a, err = Softplus(a); err != nil {
		return nil, nil, err
	}
	g, err := Mul(a, negA)
	if err != nil {
		return nil, nil, err
	}
	return GatedDeltaRule(q, k, v, g, beta, state, opts)
}

// repeatHeads repeats the head axis of x (B, T, H, d) r times: interleaved
// (head j → j / r) or tiled (head j → j mod H).
func repeatHeads(x *tensor.Tensor, r int, tiled bool) (*tensor.Tensor, error) {
	B, T, H, d := x.Dim(0), x.Dim(1), x.Dim(2), x.Dim(3)
	parts := make([]*tensor.Tensor, r)
	if tiled {
		for i := range parts {
			parts[i] = x
		}
		return Concat(parts, 2)
	}
	x5, err := x.Reshape(B, T, H, 1, d)
	if err != nil {
		return nil, err
	}
	for i := range parts {
		parts[i] = x5
	}
	cat, err := Concat(parts, 3)
	if err != nil {
		return nil, err
	}
	return cat.Reshape(B, T, H*r, d)
}

// GatedRMSNormSiLU is Gated DeltaNet's output norm: rmsnorm(x) · w · silu(z)
// over the last axis, in F32, written as outDT (transformers'
// Qwen3_5RMSNormGated with the gate applied after the norm).
//
//	x (..., d) F32   z (..., d) F32 or BF16   w (d,) F32
//
// One launch on CUDA; elsewhere RMSNorm, Cast, SiLU and Mul.
func GatedRMSNormSiLU(x, z, w *tensor.Tensor, eps float64, outDT dtype.DType) (*tensor.Tensor, error) {
	if x.DType() != dtype.F32 || w.DType() != dtype.F32 || w.Rank() != 1 || w.Dim(0) != x.Dim(-1) {
		return nil, fmt.Errorf("GatedRMSNormSiLU: x (..., d) and w (d,) must be F32, got %v %v / %v %v", x.Shape(), x.DType(), w.Shape(), w.DType())
	}
	if !z.Shape().Equal(x.Shape()) || (z.DType() != dtype.F32 && z.DType() != dtype.BF16) {
		return nil, fmt.Errorf("GatedRMSNormSiLU: z must be x's shape %v in F32 or BF16, got %v %v", x.Shape(), z.Shape(), z.DType())
	}
	if outDT != dtype.F32 && outDT != dtype.BF16 {
		return nil, fmt.Errorf("GatedRMSNormSiLU: output must be F32 or BF16, got %v", outDT)
	}
	dev := x.Device()
	kern, err := registry.Lookup(registry.Key{Op: "gated_rmsnorm_silu", Dev: dev.Kind(), DType: dtype.F32})
	if err != nil {
		out, err := RMSNorm(x, w, eps)
		if err != nil {
			return nil, err
		}
		if z.DType() != dtype.F32 {
			if z, err = Cast(z, dtype.F32); err != nil {
				return nil, err
			}
		}
		if z, err = SiLU(z); err != nil {
			return nil, err
		}
		if out, err = Mul(out, z); err != nil {
			return nil, err
		}
		if outDT != dtype.F32 {
			return Cast(out, outDT)
		}
		return out, nil
	}
	ins := []*tensor.Tensor{x, z, w}
	for i, t := range ins {
		if !t.Device().SameAs(dev) {
			return nil, fmt.Errorf("GatedRMSNormSiLU: input %d on %v, want %v", i, t.Device(), dev)
		}
		if ins[i], err = ensureContig(t); err != nil {
			return nil, err
		}
	}
	out, err := tensor.Empty(outDT, x.Shape().Dims(), dev)
	if err != nil {
		return nil, err
	}
	if err := kern(&registry.Context{Inputs: ins, Output: out, Attrs: map[string]any{"eps": eps}}); err != nil {
		return nil, err
	}
	return out, nil
}
