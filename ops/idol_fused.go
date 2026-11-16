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
