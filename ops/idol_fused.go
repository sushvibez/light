package ops

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// Fused kernels for Idol's transformer blocks on CUDA (kernels/cuda/src/
// idol.cu): each folds a chain of F32 elementwise passes into one, and
// those that feed a GEMM write its 16-bit input directly. Only CUDA has
// them: IdolFusedOK reports whether they run for x; models/idol.go keeps
// the unfused ops elsewhere. Optional tensors may be nil.

// IdolFusedOK reports whether the fused Idol kernels run on x's device.
func IdolFusedOK(x *tensor.Tensor) bool {
	if x == nil || !device.IsCUDA(x.Device()) {
		return false
	}
	_, err := registry.Lookup(registry.Key{Op: "idol_ln_mod_f16", Dev: x.Device().Kind(), DType: dtype.F32})
	return err == nil
}

// idolRun runs op on contiguous copies of ins: all F32 but input half,
// which may be F16 (a GEMM output; -1: none).
func idolRun(op string, out *tensor.Tensor, attrs map[string]any, half int, ins ...*tensor.Tensor) error {
	dt := dtype.F32
	if half >= 0 {
		dt = ins[half].DType()
	}
	k, err := registry.Lookup(registry.Key{Op: op, Dev: out.Device().Kind(), DType: dt})
	if err != nil {
		return err
	}
	cs := make([]*tensor.Tensor, len(ins))
	for i, t := range ins {
		if t == nil {
			continue
		}
		if t.DType() != dtype.F32 && !(i == half && t.DType() == dtype.F16) {
			return fmt.Errorf("%s: input %d is %v, want F32", op, i, t.DType())
		}
		if cs[i], err = ensureContig(t); err != nil {
			return err
		}
	}
	return k(&registry.Context{Inputs: cs, Output: out, Attrs: attrs})
}

// batchRows returns the rows per modulation batch for x (..., C) against a
// per-batch table m (B, C) (or (B, 1, C)), or all rows when m is nil.
func batchRows(op string, rows, C int, m *tensor.Tensor) (int, error) {
	if m == nil {
		return rows, nil
	}
	B := m.Dim(0)
	if m.NumElements() != B*C || rows%B != 0 {
		return 0, fmt.Errorf("%s: per-batch table %v does not fit %d rows of %d", op, m.Shape(), rows, C)
	}
	return rows / B, nil
}

// IdolLNModF16 is LayerNorm(x)·w + b (w, b nil: no affine), then
// ·(1 + scale) + shift per batch (scale, shift (B, C), nil: none), rounded
// to F16. x (..., C) F32, C a multiple of 4 up to 4096.
func IdolLNModF16(x, w, b, scale, shift *tensor.Tensor, eps float64) (*tensor.Tensor, error) {
	C := x.Dim(-1)
	rows := x.NumElements() / C
	if C%4 != 0 || C > 4096 || (w == nil) != (b == nil) || (scale == nil) != (shift == nil) {
		return nil, fmt.Errorf("IdolLNModF16: unsupported shape %v or arguments", x.Shape())
	}
	rpb, err := batchRows("IdolLNModF16", rows, C, scale)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Empty(dtype.F16, x.Shape().Dims(), x.Device())
	if err != nil {
		return nil, err
	}
	return out, idolRun("idol_ln_mod_f16", out, map[string]any{"rows": rows, "cols": C, "rpb": rpb, "eps": eps}, -1, x, w, b, scale, shift)
}

// IdolLN is one LayerNorm of IdolResLN: affine W, B and per-batch
// modulation Scale, Shift ((B, C)), each pair optional.
type IdolLN struct {
	W, B, Scale, Shift *tensor.Tensor
	Eps                float64
}

// IdolResLN is a block boundary in one pass: x' = x + (y + bias)·gate
// (when y is set: F32 or F16, bias and per-batch gate optional; else x' =
// x), then LayerNorm a of x' (and b, when set) as IdolLNModF16 does.
// Returns x' (x itself without y) and the F16 norms (ob nil without b).
func IdolResLN(x, y, bias, gate *tensor.Tensor, a IdolLN, b *IdolLN) (xo, oa, ob *tensor.Tensor, err error) {
	C := x.Dim(-1)
	rows := x.NumElements() / C
	bad := C%4 != 0 || C > 4096 || x.DType() != dtype.F32 || (y != nil && !y.Shape().Equal(x.Shape()))
	lns := []IdolLN{a}
	if b != nil {
		lns = append(lns, *b)
	}
	tab := gate
	for _, l := range lns {
		bad = bad || (l.W == nil) != (l.B == nil) || (l.Scale == nil) != (l.Shift == nil)
		if l.Scale != nil {
			tab = l.Scale
		}
	}
	if bad {
		ys := "none"
		if y != nil {
			ys = fmt.Sprint(y.Shape())
		}
		return nil, nil, nil, fmt.Errorf("IdolResLN: unsupported x %v %v, y %v, or norm arguments", x.Shape(), x.DType(), ys)
	}
	rpb, err := batchRows("IdolResLN", rows, C, tab)
	if err != nil {
		return nil, nil, nil, err
	}
	dt := dtype.F32
	if y != nil {
		dt = y.DType()
	}
	k, err := registry.Lookup(registry.Key{Op: "idol_res_ln_f16", Dev: x.Device().Kind(), DType: dt})
	if err != nil {
		return nil, nil, nil, err
	}
	contig := func(t *tensor.Tensor, want dtype.DType) *tensor.Tensor {
		if t == nil || err != nil {
			return nil
		}
		if t.DType() != want {
			err = fmt.Errorf("IdolResLN: %v input, want %v", t.DType(), want)
			return nil
		}
		var c *tensor.Tensor
		c, err = ensureContig(t)
		return c
	}
	ins := make([]*tensor.Tensor, 15)
	ins[0] = contig(x, dtype.F32)
	if y != nil {
		ins[1] = contig(y, y.DType())
		if y.DType() != dtype.F32 && y.DType() != dtype.F16 {
			return nil, nil, nil, fmt.Errorf("IdolResLN: y %v", y.DType())
		}
	}
	ins[2], ins[3] = contig(bias, dtype.F32), contig(gate, dtype.F32)
	for i, l := range lns {
		ins[4+4*i], ins[5+4*i], ins[6+4*i], ins[7+4*i] = contig(l.W, dtype.F32), contig(l.B, dtype.F32), contig(l.Scale, dtype.F32), contig(l.Shift, dtype.F32)
	}
	if err != nil {
		return nil, nil, nil, err
	}
	empty := func(dt dtype.DType) *tensor.Tensor {
		if err != nil {
			return nil
		}
		var t *tensor.Tensor
		t, err = tensor.Empty(dt, x.Shape().Dims(), x.Device())
		return t
	}
	oa = empty(dtype.F16)
	if b != nil {
		ob = empty(dtype.F16)
	}
	xo = x
	if y != nil {
		xo = empty(dtype.F32)
	}
	if err != nil {
		return nil, nil, nil, err
	}
	ins[12], ins[13] = oa, ob
	if y != nil {
		ins[14] = xo
	}
	attrs := map[string]any{"rows": rows, "cols": C, "rpb": rpb, "epsa": a.Eps, "epsb": 0.0}
	if b != nil {
		attrs["epsb"] = b.Eps
	}
	if err := k(&registry.Context{Inputs: ins, Output: oa, Attrs: attrs}); err != nil {
		return nil, nil, nil, err
	}
	return xo, oa, ob, nil
}

// IdolHeadF16 takes one of q / k / v from a projection output x (rows,
// W) F32 or F16: columns [off, off + H·64) plus bias[off:] (bias (W,), nil:
// none), turned by RePo's rotary angles when ropeD (rows, H, 3) is set
// (offsets + ropeB (H·3), times freq (32) along axis (32, the offset each
// pair uses, as floats)), then RMS-normalised per head times gamma (H, 64)
// when set. Returns (rows, H, 64) F16.
func IdolHeadF16(x *tensor.Tensor, off, H int, bias, ropeD, ropeB, freq, axis, gamma *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Rank() != 2 || off+H*64 > x.Dim(1) || (ropeD != nil && (ropeD.NumElements() != x.Dim(0)*H*3 || freq == nil || axis == nil)) {
		return nil, fmt.Errorf("IdolHeadF16: x %v, off %d, %d heads of 64", x.Shape(), off, H)
	}
	out, err := tensor.Empty(dtype.F16, []int{x.Dim(0), H, 64}, x.Device())
	if err != nil {
		return nil, err
	}
	return out, idolRun("idol_head_f16", out, map[string]any{"rows": x.Dim(0), "stride": x.Dim(1), "off": off, "heads": H},
		0, x, bias, ropeD, ropeB, freq, axis, gamma)
}

// IdolGateRes is x + (y + bias)·gate, gate per batch (B, C); bias and
// gate may be nil. x, y (..., C) of one shape, x F32, y F32 or F16, C a
// multiple of 4.
func IdolGateRes(x, y, bias, gate *tensor.Tensor) (*tensor.Tensor, error) {
	C := x.Dim(-1)
	if !x.Shape().Equal(y.Shape()) || C%4 != 0 {
		return nil, fmt.Errorf("IdolGateRes: x %v, y %v", x.Shape(), y.Shape())
	}
	rows := x.NumElements() / C
	rpb, err := batchRows("IdolGateRes", rows, C, gate)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Empty(dtype.F32, x.Shape().Dims(), x.Device())
	if err != nil {
		return nil, err
	}
	return out, idolRun("idol_gate_res_f32", out, map[string]any{"rows": rows, "cols": C, "rpb": rpb}, 1, x, y, bias, gate)
}

// IdolBiasGeluF16 is tanh-GELU(y + bias) rounded to F16 (y F32 or F16,
// bias (C,), nil: none; C a multiple of 4).
func IdolBiasGeluF16(y, bias *tensor.Tensor) (*tensor.Tensor, error) {
	C := y.Dim(-1)
	if C%4 != 0 {
		return nil, fmt.Errorf("IdolBiasGeluF16: %v", y.Shape())
	}
	out, err := tensor.Empty(dtype.F16, y.Shape().Dims(), y.Device())
	if err != nil {
		return nil, err
	}
	return out, idolRun("idol_bias_gelu_f16", out, map[string]any{"n": y.NumElements(), "cols": C}, 0, y, bias)
}

// IdolSiLUMulF16 is silu(g + bg)·(u + bu) rounded to F16 (biases (C,) or
// nil; g, u (..., C) of one shape, C a multiple of 4).
func IdolSiLUMulF16(g, u, bg, bu *tensor.Tensor) (*tensor.Tensor, error) {
	C := g.Dim(-1)
	if !g.Shape().Equal(u.Shape()) || C%4 != 0 {
		return nil, fmt.Errorf("IdolSiLUMulF16: g %v, u %v", g.Shape(), u.Shape())
	}
	out, err := tensor.Empty(dtype.F16, g.Shape().Dims(), g.Device())
	if err != nil {
		return nil, err
	}
	return out, idolRun("idol_silu_mul_f16", out, map[string]any{"n": g.NumElements(), "cols": C}, -1, g, u, bg, bu)
}
