//go:build !nocuda && cgo

package ops_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// The fused Idol kernels against the unfused ops they replace (on the CPU,
// then rounded as the kernels round): LayerNorm + modulation, the head
// kernel with RePo rotary and RMS norm, the gated residual, bias + GELU and
// RePo's gated product, with two modulation batches.
func TestIdolFusedMatchesUnfused(t *testing.T) {
	gpu := gpuOrSkip(t)
	rng := rand.New(rand.NewPCG(3, 4))
	cpu := device.CPU()
	const B, L, H = 2, 37, 4
	C := H * 64
	mk := func(dims ...int) *tensor.Tensor { return randTensor(t, rng, dims, 1, cpu, dtype.F32) }
	up := func(x *tensor.Tensor) *tensor.Tensor {
		if x == nil {
			return nil
		}
		g, err := x.To(gpu)
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	must := func(x *tensor.Tensor, err error) *tensor.Tensor {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	cmp := func(name string, got, want *tensor.Tensor, tol float64) {
		t.Helper()
		g, w := f32(t, must(got.To(cpu))), f32(t, must(ops.Cast(want, dtype.F32)))
		if len(g) != len(w) {
			t.Fatalf("%s: %d values, want %d", name, len(g), len(w))
		}
		md := 0.0
		for i := range g {
			md = math.Max(md, math.Abs(float64(g[i]-w[i]))/math.Max(1, math.Abs(float64(w[i]))))
		}
		t.Logf("%s: max rel diff %.3g", name, md)
		if md > tol {
			t.Errorf("%s differs by %.3g", name, md)
		}
	}
	x := mk(B, L, C)
	w, b, scale, shift := mk(C), mk(C), mk(B, C), mk(B, C)
	ones := must(tensor.Ones(dtype.F32, []int{C}, cpu))
	zeros := must(tensor.Zeros(dtype.F32, []int{C}, cpu))
	for _, c := range []struct {
		name         string
		w, b, sc, sh *tensor.Tensor
	}{{"ln", nil, nil, nil, nil}, {"ln affine", w, b, nil, nil}, {"ln mod", nil, nil, scale, shift}, {"ln affine mod", w, b, scale, shift}} {
		ww, bb := c.w, c.b
		if ww == nil {
			ww, bb = ones, zeros
		}
		want := must(ops.LayerNorm(x, ww, bb, 1e-6))
		if c.sc != nil {
			s1 := must(ops.Affine(must(c.sc.Reshape(B, 1, C)), 1, 1))
			want = must(ops.Add(must(ops.Mul(want, s1)), must(c.sh.Reshape(B, 1, C))))
		}
		got := must(ops.IdolLNModF16(up(x), up(c.w), up(c.b), up(c.sc), up(c.sh), 1e-6))
		cmp("IdolLNModF16 "+c.name, got, must(ops.Cast(want, dtype.F16)), 2e-3)
	}

	// Heads: q at offset C of a (rows, 3C) projection, with bias, RePo
	// offsets (rows, H, 3) + bias, frequencies along axes 10 / 10 / 12.
	rows := B * L
	y, bias, d, db, gamma := mk(rows, 3*C), mk(3*C), mk(rows, H, 3), mk(H*3), mk(H, 64)
	fv, av := make([]float32, 32), make([]float32, 32)
	for j := range fv {
		fv[j] = float32(rng.Float64()*4) * math.Pi
		av[j] = float32(min(j/10, 2))
	}
	freq, axis := must(tensor.New(fv, []int{32}, cpu)), must(tensor.New(av, []int{32}, cpu))
	// Reference: slice + bias, angles, rotate pairs, RMS norm · gamma.
	yv, bv, dv, dbv, gv := f32(t, y), f32(t, bias), f32(t, d), f32(t, db), f32(t, gamma)
	ref := make([]float32, rows*H*64)
	refV := make([]float32, rows*H*64)
	for r := range rows {
		for h := range H {
			var p [64]float64
			for c := range 64 {
				col := C + h*64 + c
				p[c] = float64(yv[r*3*C+col] + bv[col])
				refV[(r*H+h)*64+c] = yv[r*3*C+2*C+h*64+c] + bv[2*C+h*64+c]
			}
			ss := 0.0
			for j := range 32 {
				a := int(av[j])
				ang := float64((dv[(r*H+h)*3+a] + dbv[h*3+a]) * fv[j])
				cs, sn := math.Cos(ang), math.Sin(ang)
				x0, x1 := p[2*j], p[2*j+1]
				p[2*j], p[2*j+1] = x0*cs-x1*sn, x0*sn+x1*cs
				ss += p[2*j]*p[2*j] + p[2*j+1]*p[2*j+1]
			}
			rms := 1 / math.Sqrt(ss/64+1e-30)
			for c := range 64 {
				ref[(r*H+h)*64+c] = float32(p[c] * rms * float64(gv[h*64+c]))
			}
		}
	}
	q := must(ops.IdolHeadF16(up(y), C, H, up(bias), up(d), up(db), up(freq), up(axis), up(gamma)))
	cmp("IdolHeadF16 q (rope + norm)", q, must(tensor.New(ref, []int{rows, H, 64}, cpu)), 1e-2)
	v := must(ops.IdolHeadF16(up(y), 2*C, H, up(bias), nil, nil, nil, nil, nil))
	cmp("IdolHeadF16 v", v, must(tensor.New(refV, []int{rows, H, 64}, cpu)), 1e-2)

	// FP16 flash attention against an F64 softmax(q·kᵀ/8)·v of the same
	// F16 q, k, v (sequence-major (B, L, H, 64)).
	qh := must(q.Reshape(B, L, H, 64))
	kh := must(must(ops.IdolHeadF16(up(y), 0, H, up(bias), nil, nil, nil, nil, up(gamma))).Reshape(B, L, H, 64))
	vh := must(v.Reshape(B, L, H, 64))
	cmp("FlashAttentionF16SeqMajor", must(ops.FlashAttentionF16SeqMajor(qh, kh, vh, false)), idolAttRef(t, qh, kh, vh), 3e-3)
	// Cross attention: 150 keys (two full tiles and a ragged one) for 37 queries.
	kx := must(ops.Cast(up(mk(B, 150, H, 64)), dtype.F16))
	vx := must(ops.Cast(up(mk(B, 150, H, 64)), dtype.F16))
	cmp("FlashAttentionF16SeqMajor (cross)", must(ops.FlashAttentionF16SeqMajor(qh, kx, vx, false)), idolAttRef(t, qh, kx, vx), 3e-3)

	// Gated residual, bias + GELU, gated product.
	yy, gate := mk(B, L, C), mk(B, C)
	want := must(ops.Add(x, must(ops.Mul(must(ops.Add(yy, b)), must(gate.Reshape(B, 1, C))))))
	cmp("IdolGateRes", must(ops.IdolGateRes(up(x), up(yy), up(b), up(gate))), want, 1e-6)
	cmp("IdolGateRes (no bias / gate)", must(ops.IdolGateRes(up(x), up(yy), nil, nil)), must(ops.Add(x, yy)), 1e-6)
	want = must(ops.Gelu(must(ops.Add(yy, b))))
	cmp("IdolBiasGeluF16", must(ops.IdolBiasGeluF16(up(yy), up(b))), must(ops.Cast(want, dtype.F16)), 2e-3)
	u := mk(B, L, C)
	want = must(ops.Mul(must(ops.SiLU(must(ops.Add(yy, b)))), must(ops.Add(u, w))))
	cmp("IdolSiLUMulF16", must(ops.IdolSiLUMulF16(up(yy), up(u), up(b), up(w))), must(ops.Cast(want, dtype.F16)), 2e-3)

	// The F16-input (_h) variants read an F16 GEMM output: the same as the
	// F32 kernels on the F16 values widened.
	h16 := func(x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
		xh := must(ops.Cast(up(x), dtype.F16))
		return xh, must(ops.Cast(xh, dtype.F32))
	}
	yh, y32 := h16(y)
	cmp("IdolHeadF16 (F16 x)", must(ops.IdolHeadF16(yh, C, H, up(bias), up(d), up(db), up(freq), up(axis), up(gamma))),
		must(must(ops.IdolHeadF16(y32, C, H, up(bias), up(d), up(db), up(freq), up(axis), up(gamma))).To(cpu)), 0)
	yyh, yy32 := h16(yy)
	cmp("IdolGateRes (F16 y)", must(ops.IdolGateRes(up(x), yyh, up(b), up(gate))), must(must(ops.IdolGateRes(up(x), yy32, up(b), up(gate))).To(cpu)), 0)
	cmp("IdolBiasGeluF16 (F16 y)", must(ops.IdolBiasGeluF16(yyh, up(b))), must(must(ops.IdolBiasGeluF16(yy32, up(b))).To(cpu)), 0)
}

// idolAttRef is softmax(q·kᵀ/8)·v in F64 for (B, L, H, 64) q and (B, S, H,
// 64) k, v.
func idolAttRef(t *testing.T, q, k, v *tensor.Tensor) *tensor.Tensor {
	cpu := device.CPU()
	host := func(x *tensor.Tensor) []float32 {
		x, err := x.To(cpu)
		if err != nil {
			t.Fatal(err)
		}
		if x, err = ops.Cast(x, dtype.F32); err != nil {
			t.Fatal(err)
		}
		return f32(t, x)
	}
	qv, kv, vv := host(q), host(k), host(v)
	B, L, H, S := q.Dim(0), q.Dim(1), q.Dim(2), k.Dim(1)
	out := make([]float32, B*L*H*64)
	for bi := range B {
		for h := range H {
			for i := range L {
				p := make([]float64, S)
				mx := math.Inf(-1)
				for j := range S {
					for c := range 64 {
						p[j] += float64(qv[((bi*L+i)*H+h)*64+c]) * float64(kv[((bi*S+j)*H+h)*64+c])
					}
					p[j] /= 8
					mx = math.Max(mx, p[j])
				}
				sum := 0.0
				for j := range p {
					p[j] = math.Exp(p[j] - mx)
					sum += p[j]
				}
				for c := range 64 {
					o := 0.0
					for j := range S {
						o += p[j] * float64(vv[((bi*S+j)*H+h)*64+c])
					}
					out[((bi*L+i)*H+h)*64+c] = float32(o / sum)
				}
			}
		}
	}
	r, err := tensor.New(out, []int{B, L, H, 64}, cpu)
	if err != nil {
		t.Fatal(err)
	}
	return r
}
