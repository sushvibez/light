package cpu

import (
	"math"
	"sync"

	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "gated_delta_rule", Dev: device.KindCPU, DType: dtype.F32}, gatedDeltaKernel)
	registry.Register(registry.Key{Op: "causal_conv1d_depthwise", Dev: device.KindCPU, DType: dtype.F32}, causalConv1dDepthwiseKernel)
}

// gatedDeltaKernel is the CPU gated delta rule (see ops.GatedDeltaRule).
//
//	Inputs: q, k (B,T,H,dk), v (B,T,H,dv), g, beta (B,T,H), state (B,H,dk,dv),
//	        final (B,H,dk,dv) written
//	Output: (B,T,H,dv)
//	Attrs:  "chunked" bool, "chunk" int, "l2norm" bool
//
// Every (batch, head) pair is independent and runs on its own goroutine;
// arithmetic is float64 internally, stored as float32.
func gatedDeltaKernel(ctx *registry.Context) error {
	q, k, v, g, beta, st, fin := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Inputs[3], ctx.Inputs[4], ctx.Inputs[5], ctx.Inputs[6]
	chunked := ctx.Attrs["chunked"].(bool)
	chunk := ctx.Attrs["chunk"].(int)
	l2 := ctx.Attrs["l2norm"].(bool)
	B, T, H, dk := q.Dim(0), q.Dim(1), q.Dim(2), q.Dim(3)
	dv := v.Dim(3)
	p := gdParams{
		q: asSlice[float32](q), k: asSlice[float32](k), v: asSlice[float32](v),
		g: asSlice[float32](g), beta: asSlice[float32](beta),
		st: asSlice[float32](st), fin: asSlice[float32](fin), out: asSlice[float32](ctx.Output),
		T: T, H: H, dk: dk, dv: dv, l2: l2,
	}
	var wg sync.WaitGroup
	for b := range B {
		for h := range H {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if chunked {
					p.chunked(b, h, chunk)
				} else {
					p.recurrent(b, h)
				}
			}()
		}
	}
	wg.Wait()
	return nil
}

type gdParams struct {
	q, k, v, g, beta, st, fin, out []float32
	T, H, dk, dv                   int
	l2                             bool
}

// loadQK returns token t's q and k for head (b, h): L2-normalised when
// requested (transformers' l2norm: x · rsqrt(Σx² + 1e-6)), q scaled by
// 1/√dk.
func (p *gdParams) loadQK(b, h, t int, qo, ko []float64) {
	base := ((b*p.T+t)*p.H + h) * p.dk
	var qq, kk float64
	for i := range p.dk {
		qo[i], ko[i] = float64(p.q[base+i]), float64(p.k[base+i])
		qq += qo[i] * qo[i]
		kk += ko[i] * ko[i]
	}
	qs, ks := 1/math.Sqrt(float64(p.dk)), 1.0
	if p.l2 {
		qs /= math.Sqrt(qq + 1e-6)
		ks = 1 / math.Sqrt(kk+1e-6)
	}
	for i := range p.dk {
		qo[i] *= qs
		ko[i] *= ks
	}
}

func (p *gdParams) loadState(b, h int) []float64 {
	S := make([]float64, p.dk*p.dv)
	base := (b*p.H + h) * p.dk * p.dv
	for i := range S {
		S[i] = float64(p.st[base+i])
	}
	return S
}

func (p *gdParams) storeState(b, h int, S []float64) {
	base := (b*p.H + h) * p.dk * p.dv
	for i, s := range S {
		p.fin[base+i] = float32(s)
	}
}

// recurrent: torch_recurrent_gated_delta_rule, one token at a time.
func (p *gdParams) recurrent(b, h int) {
	dk, dv := p.dk, p.dv
	S := p.loadState(b, h)
	qt, kt := make([]float64, dk), make([]float64, dk)
	delta := make([]float64, dv)
	for t := range p.T {
		p.loadQK(b, h, t, qt, kt)
		gi := (b*p.T+t)*p.H + h
		decay, bt := math.Exp(float64(p.g[gi])), float64(p.beta[gi])
		for i := range S {
			S[i] *= decay
		}
		vBase := gi * dv
		for j := range dv {
			var mem float64
			for i := range dk {
				mem += S[i*dv+j] * kt[i]
			}
			delta[j] = (float64(p.v[vBase+j]) - mem) * bt
		}
		for i := range dk {
			row := S[i*dv : (i+1)*dv]
			for j := range dv {
				row[j] += kt[i] * delta[j]
			}
		}
		for j := range dv {
			var o float64
			for i := range dk {
				o += S[i*dv+j] * qt[i]
			}
			p.out[vBase+j] = float32(o)
		}
	}
	p.storeState(b, h, S)
}

// chunked: torch_chunk_gated_delta_rule. The sequence is cut into blocks
// of C tokens (the last one zero-padded: β = 0 and g = 0 there, so padding
// leaves the state alone). Within a block, with cumulative log decay γ and
// kβ_c = β_c k_c, vβ_c = β_c v_c:
//
//	A = I + strictly-lower[(kβ_i · k_j) e^{γ_i − γ_j}]
//	U = A⁻¹ vβ,  W = A⁻¹ (kβ ∘ e^{γ})            (forward substitution)
//	V' = U − W S
//	o_c = e^{γ_c} q_c S + Σ_{m ≤ c} (q_c · k_m) e^{γ_c − γ_m} V'_m
//	S ← e^{γ_C} S + Σ_c e^{γ_C − γ_c} k_c ⊗ V'_c
func (p *gdParams) chunked(b, h, C int) {
	dk, dv, T := p.dk, p.dv, p.T
	S := p.loadState(b, h)
	q := make([][]float64, C)
	k := make([][]float64, C)
	vb := make([][]float64, C)  // β v, then U = A⁻¹ β v
	kbd := make([][]float64, C) // β k e^{γ}, then W
	kb := make([][]float64, C)  // β k
	vNew := make([][]float64, C)
	for c := range C {
		q[c], k[c], kb[c] = make([]float64, dk), make([]float64, dk), make([]float64, dk)
		kbd[c], vb[c], vNew[c] = make([]float64, dk), make([]float64, dv), make([]float64, dv)
	}
	cum := make([]float64, C)
	L := make([]float64, C*C)
	for t0 := 0; t0 < T; t0 += C {
		var acc float64
		for c := range C {
			t := t0 + c
			if t >= T {
				clear(q[c])
				clear(k[c])
				clear(kb[c])
				clear(kbd[c])
				clear(vb[c])
				cum[c] = acc
				continue
			}
			p.loadQK(b, h, t, q[c], k[c])
			gi := (b*T+t)*p.H + h
			acc += float64(p.g[gi])
			cum[c] = acc
			bt := float64(p.beta[gi])
			e := math.Exp(acc)
			for i := range dk {
				kb[c][i] = k[c][i] * bt
				kbd[c][i] = kb[c][i] * e
			}
			for j := range dv {
				vb[c][j] = float64(p.v[gi*dv+j]) * bt
			}
		}
		// Strictly lower part of A (the unit diagonal is implicit).
		for i := range C {
			for j := range i {
				var d float64
				for x := range dk {
					d += kb[i][x] * k[j][x]
				}
				L[i*C+j] = d * math.Exp(cum[i]-cum[j])
			}
		}
		// Forward substitution, in place: x_i = b_i − Σ_{j<i} L_ij x_j.
		for i := range C {
			for j := range i {
				l := L[i*C+j]
				if l == 0 {
					continue
				}
				for x := range dv {
					vb[i][x] -= l * vb[j][x]
				}
				for x := range dk {
					kbd[i][x] -= l * kbd[j][x]
				}
			}
		}
		// V' = U − W S.
		for c := range C {
			for j := range dv {
				var ws float64
				for i := range dk {
					ws += kbd[c][i] * S[i*dv+j]
				}
				vNew[c][j] = vb[c][j] - ws
			}
		}
		// Outputs: inter-chunk (from the carried state) + intra-chunk.
		for c := range C {
			t := t0 + c
			if t >= T {
				break
			}
			e := math.Exp(cum[c])
			base := ((b*T+t)*p.H + h) * dv
			for j := range dv {
				var o float64
				for i := range dk {
					o += q[c][i] * S[i*dv+j]
				}
				o *= e
				for m := 0; m <= c; m++ {
					var qk float64
					for x := range dk {
						qk += q[c][x] * k[m][x]
					}
					o += qk * math.Exp(cum[c]-cum[m]) * vNew[m][j]
				}
				p.out[base+j] = float32(o)
			}
		}
		// Carry the state to the next chunk.
		last := cum[C-1]
		el := math.Exp(last)
		for i := range S {
			S[i] *= el
		}
		for c := range C {
			w := math.Exp(last - cum[c])
			for i := range dk {
				ki := k[c][i] * w
				if ki == 0 {
					continue
				}
				row := S[i*dv : (i+1)*dv]
				for j := range dv {
					row[j] += ki * vNew[c][j]
				}
			}
		}
	}
	p.storeState(b, h, S)
}

// causalConv1dDepthwiseKernel: see ops.CausalConv1dDepthwise.
//
//	Inputs: x (B,C,T), w (C,K), state (B,C,K-1), new state (B,C,K-1) written
//	Output: (B,C,T)
//	Attrs:  "silu" bool
func causalConv1dDepthwiseKernel(ctx *registry.Context) error {
	x, w, st, nst, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Inputs[3], ctx.Output
	silu := ctx.Attrs["silu"].(bool)
	B, C, T, K := x.Dim(0), x.Dim(1), x.Dim(2), w.Dim(1)
	xs, ws, ss, ns, os := asSlice[float32](x), asSlice[float32](w), asSlice[float32](st), asSlice[float32](nst), asSlice[float32](out)
	P := K - 1
	for b := range B {
		for c := range C {
			xRow := xs[(b*C+c)*T : (b*C+c+1)*T]
			sRow := ss[(b*C+c)*P : (b*C+c+1)*P]
			wRow := ws[c*K : (c+1)*K]
			// in(i) is the input at time i in [-P, T): state, then x.
			in := func(i int) float64 {
				if i < 0 {
					return float64(sRow[P+i])
				}
				return float64(xRow[i])
			}
			for t := range T {
				var acc float64
				for j := range K {
					acc += float64(wRow[j]) * in(t-P+j)
				}
				if silu {
					acc /= 1 + math.Exp(-acc)
				}
				os[(b*C+c)*T+t] = float32(acc)
			}
			nRow := ns[(b*C+c)*P : (b*C+c+1)*P]
			for i := range P {
				nRow[i] = float32(in(T - P + i))
			}
		}
	}
	return nil
}
