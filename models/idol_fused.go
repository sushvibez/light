package models

import (
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// Idol's blocks on CUDA with fused kernels (ops.Idol*, kernels/cuda/src/
// idol.cu): the F16 GEMMs take their input already rounded by the kernel
// before them (LayerNorm + modulation, GELU, RePo's gated product) and
// leave their bias to the kernel after them (gated residual, q / k / v
// with RePo rotary and RMS norm). Same arithmetic as the unfused path up
// to F32 rounding order; used when the block's linears compute at F16
// (WithWeightStorage16Mixed) on a device that has the kernels.

// fusedOK reports whether the fused path runs for x through these linears
// (F16 weights computing at 16 bits, heads of 64, kernels on x's device).
func (c *idolCommon) fusedOK(x *tensor.Tensor, ls ...*nn.Linear) bool {
	if c.C != c.H*64 || !ops.IdolFusedOK(x) {
		return false
	}
	for _, l := range ls {
		if l == nil || !l.Mixed16 || l.W == nil || l.W.DType() != dtype.F16 || l.W4 != nil || l.LoRA != nil {
			return false
		}
	}
	return true
}

// gemm is x16 (..., in) F16 times l's weight: F32 (rows, out), no bias.
func (c *idolCommon) gemm(o *iop, l *nn.Linear, x16 *tensor.Tensor) *tensor.Tensor {
	return c.gemmTo(o, l, x16, false)
}

// gemmH is gemm with the product rounded to F16 (as idol-sdk's FP16
// linears, before their bias): half the traffic to the kernel that reads it.
func (c *idolCommon) gemmH(o *iop, l *nn.Linear, x16 *tensor.Tensor) *tensor.Tensor {
	return c.gemmTo(o, l, x16, true)
}

func (c *idolCommon) gemmTo(o *iop, l *nn.Linear, x16 *tensor.Tensor, f16 bool) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) {
		in := l.W.Dim(1)
		flat, err := x16.Contiguous()
		if err != nil {
			return nil, err
		}
		if flat, err = flat.Reshape(x16.NumElements()/in, in); err != nil {
			return nil, err
		}
		wT, err := l.W.Transpose(0, 1)
		if err != nil {
			return nil, err
		}
		if f16 {
			return ops.MatMulMixed(flat, wT)
		}
		return ops.MatMulMixedF32Out(flat, wT)
	})
}

// f32 returns t at F32 (cached per tensor: biases, gammas, norm weights).
func (c *idolCommon) f32(o *iop, t *tensor.Tensor) *tensor.Tensor {
	if t == nil || o.err != nil {
		return nil
	}
	if t.DType() == dtype.F32 {
		return t
	}
	if v, ok := c.f32Cache[t]; ok {
		return v
	}
	v := o.do(func() (*tensor.Tensor, error) { return ops.Cast(t, dtype.F32) })
	if v != nil {
		if c.f32Cache == nil {
			c.f32Cache = map[*tensor.Tensor]*tensor.Tensor{}
		}
		c.f32Cache[t] = v
	}
	return v
}

func (c *idolCommon) bias(o *iop, l *nn.Linear) *tensor.Tensor { return c.f32(o, l.B) }

// lnMod16 is the block's LayerNorm (n's affine, or none) then
// ·(1 + scale) + shift when scale is set, in F16.
func (c *idolCommon) lnMod16(o *iop, n idolNorm, x, scale, shift *tensor.Tensor) *tensor.Tensor {
	eps := n.Eps
	if eps == 0 {
		eps = 1e-6
	}
	w, b := c.f32(o, n.W), c.f32(o, n.B)
	return o.do(func() (*tensor.Tensor, error) { return ops.IdolLNModF16(x, w, b, scale, shift, eps) })
}

// ropeTables returns RePo's per-pair frequency (× π) and axis tables (D/2).
func (r *idolRePo) ropeTables(o *iop, c *idolCommon) (freq, axis *tensor.Tensor) {
	if r.freqTab == nil && o.err == nil {
		parts := make([]*tensor.Tensor, 3)
		var ax []float32
		for k := range parts {
			parts[k] = o.affine(r.Freqs[k], 3.141592653589793, 0)
			if o.err != nil {
				return nil, nil
			}
			for range r.Freqs[k].NumElements() {
				ax = append(ax, float32(k))
			}
		}
		r.freqTab = o.cat(0, parts...)
		r.axisTab = o.do(func() (*tensor.Tensor, error) { return tensor.New(ax, []int{len(ax)}, c.dev) })
	}
	return r.freqTab, r.axisTab
}

// offsets returns RePo's per-head 3-D offsets (rows, H·3) without the final
// bias (the head kernel adds it).
func (r *idolRePo) offsets(o *iop, c *idolCommon, h *tensor.Tensor) *tensor.Tensor {
	n := c.lnMod16(o, r.Norm, h, nil, nil)
	g, u := c.gemm(o, r.Gate, n), c.gemm(o, r.Content, n)
	bg, bu := c.bias(o, r.Gate), c.bias(o, r.Content)
	s := o.do(func() (*tensor.Tensor, error) { return ops.IdolSiLUMulF16(g, u, bg, bu) })
	return c.gemm(o, r.Final, s)
}

// heads splits a projection output (rows, k·C) into F16 (B, L, H, 64)
// q / k / v at column offsets offs, rotated by rope when set and
// RMS-normalised by gammas (nil entries: none).
func (c *idolCommon) heads(o *iop, proj *nn.Linear, y *tensor.Tensor, B, L int, offs []int, gammas []*tensor.Tensor,
	rope *idolRePo, ropeD *tensor.Tensor) []*tensor.Tensor {
	out := make([]*tensor.Tensor, len(offs))
	b := c.bias(o, proj)
	var freq, axis, rb *tensor.Tensor
	if rope != nil {
		freq, axis = rope.ropeTables(o, c)
		rb = c.bias(o, rope.Final)
	}
	for i, off := range offs {
		g := c.f32(o, gammas[i])
		d, f, a, bb := ropeD, freq, axis, rb
		if g == nil { // v: no rotary, no norm
			d, f, a, bb = nil, nil, nil, nil
		}
		out[i] = o.do(func() (*tensor.Tensor, error) {
			t, err := ops.IdolHeadF16(y, off, c.H, b, d, bb, f, a, g)
			if err != nil {
				return nil, err
			}
			return t.Reshape(B, L, c.H, c.C/c.H)
		})
	}
	return out
}

// flash runs FP16 flash attention over (B, L, H, 64) q, k, v (as idol-sdk's
// FP16 SDPA): (B, L, C) F16 for the output projection.
func (c *idolCommon) flash(o *iop, q, k, v *tensor.Tensor) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) {
		a, err := ops.FlashAttentionF16SeqMajor(q, k, v, false)
		if err != nil {
			return nil, err
		}
		return a.Reshape(q.Dim(0), q.Dim(1), c.C)
	})
}

// ffn16 is the FFN on an F16 input: F16 (rows, C) without L2's bias.
func (c *idolCommon) ffn16(o *iop, f idolFFN, h16 *tensor.Tensor) *tensor.Tensor {
	y := c.gemmH(o, f.L1, h16)
	b := c.bias(o, f.L1)
	g := o.do(func() (*tensor.Tensor, error) { return ops.IdolBiasGeluF16(y, b) })
	return c.gemmH(o, f.L2, g)
}

// gateRes is x + (y + bias)·gate, y (rows, C) reshaped as x.
func (c *idolCommon) gateRes(o *iop, x, y, bias, gate *tensor.Tensor) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) {
		yy, err := y.Reshape(x.Shape().Dims()...)
		if err != nil {
			return nil, err
		}
		return ops.IdolGateRes(x, yy, bias, gate)
	})
}

// modTable returns the six (B, C) modulation rows of tmod (+ table).
func (c *idolCommon) modTable(o *iop, tmod, table *tensor.Tensor) []*tensor.Tensor {
	m := tmod
	if table != nil {
		m = o.add(tmod, table)
	}
	out := make([]*tensor.Tensor, 6)
	for i := range out {
		out[i] = o.do(func() (*tensor.Tensor, error) {
			n, err := m.Narrow(1, i*c.C, c.C)
			if err != nil {
				return nil, err
			}
			return n.Contiguous()
		})
	}
	return out
}

// forwardFused is forward on the fused kernels.
func (b *idolUnifiedBlock) forwardFused(o *iop, c *idolCommon, x, tmod *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	B, L := x.Dim(0), x.Dim(1)
	var m []*tensor.Tensor
	if b.ShiftTable != nil {
		m = c.modTable(o, tmod, b.ShiftTable)
	} else {
		m = make([]*tensor.Tensor, 6)
	}
	d := b.RePo.offsets(o, c, x)
	h := c.lnMod16(o, b.Norm1, x, m[1], m[0])
	qkv := c.gemmH(o, b.QKV, h)
	C := c.C
	t := c.heads(o, b.QKV, qkv, B, L, []int{0, C, 2 * C}, []*tensor.Tensor{b.QNorm, b.KNorm, nil}, b.RePo, d)
	a := c.flash(o, t[0], t[1], t[2])
	x = c.gateRes(o, x, c.gemmH(o, b.Out, a), c.bias(o, b.Out), m[2])
	h = c.lnMod16(o, b.Norm2, x, m[4], m[3])
	return c.gateRes(o, x, c.ffn16(o, b.FFN, h), c.bias(o, b.FFN.L2), m[5])
}

// cross is cross attention from h16 (B, L, C) F16 to the context ctx16
// (B, S, ·) F16 (q from h, k / v from ctx, both RMS-normalised): (B, L, C)
// F16 before the output projection.
func (c *idolCommon) cross(o *iop, a *idolMHA, h16, ctx16 *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	B, L, S := h16.Dim(0), h16.Dim(1), ctx16.Dim(1)
	q := c.heads(o, a.Q, c.gemmH(o, a.Q, h16), B, L, []int{0}, []*tensor.Tensor{a.QNorm}, nil, nil)[0]
	kv := c.heads(o, a.KV, c.gemmH(o, a.KV, ctx16), B, S, []int{0, c.C}, []*tensor.Tensor{a.KNorm, nil}, nil, nil)
	return c.flash(o, q, kv[0], kv[1])
}

// crossFusedOK reports whether the block's cross attention and FFN run on
// the fused kernels for x against ctx.
func (b *idolCrossBlock) crossFusedOK(c *idolCommon, x, ctx *tensor.Tensor) bool {
	a := b.CrossAttn
	return a.QNorm != nil && x.Dim(0) == ctx.Dim(0) && c.fusedOK(x, a.Q, a.KV, a.Out, b.FFN.L1, b.FFN.L2)
}

// forwardFused is the Gaussian decoder's block (self, cross, FFN) on the
// fused kernels.
func (b *idolCrossBlock) forwardFused(o *iop, c *idolCommon, x, ctx *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	B, L := x.Dim(0), x.Dim(1)
	a := b.SelfAttn
	h := c.lnMod16(o, idolNorm{}, x, nil, nil)
	t := c.heads(o, a.QKV, c.gemmH(o, a.QKV, h), B, L, []int{0, c.C, 2 * c.C}, []*tensor.Tensor{a.QNorm, a.KNorm, nil}, nil, nil)
	x = c.gateRes(o, x, c.gemmH(o, a.Out, c.flash(o, t[0], t[1], t[2])), c.bias(o, a.Out), nil)
	h = c.lnMod16(o, b.Norm2, x, nil, nil)
	ca := b.CrossAttn
	x = c.gateRes(o, x, c.gemmH(o, ca.Out, c.cross(o, ca, h, cast16(o, ctx))), c.bias(o, ca.Out), nil)
	h = c.lnMod16(o, idolNorm{}, x, nil, nil)
	return c.gateRes(o, x, c.ffn16(o, b.FFN, h), c.bias(o, b.FFN.L2), nil)
}

// forwardFusedMod is the octree decoder's block (cross attention and FFN,
// adaLN-modulated by mod (B, 6C)) on the fused kernels.
func (b *idolCrossBlock) forwardFusedMod(o *iop, c *idolCommon, x, ctx, mod *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	m := c.modTable(o, mod, nil)
	h := c.lnMod16(o, idolNorm{}, x, m[1], m[0])
	ca := b.CrossAttn
	x = c.gateRes(o, x, c.gemmH(o, ca.Out, c.cross(o, ca, h, cast16(o, ctx))), c.bias(o, ca.Out), m[2])
	h = c.lnMod16(o, idolNorm{}, x, m[4], m[3])
	return c.gateRes(o, x, c.ffn16(o, b.FFN, h), c.bias(o, b.FFN.L2), m[5])
}

// cast16 rounds x to F16 (a GEMM's input).
func cast16(o *iop, x *tensor.Tensor) *tensor.Tensor {
	return o.do(func() (*tensor.Tensor, error) { return ops.Cast(x, dtype.F16) })
}
