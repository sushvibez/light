package ops_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/quant"
	"light/tensor"
)

// moeReference is the per-expert loop (index_select → matmul → scale →
// index_add) that the grouped GEMM must reproduce, in F32 on the CPU.
func moeReference(t *testing.T, x, w *tensor.Tensor, route *ops.MoERouting, topk int, weighted bool) []float32 {
	t.Helper()
	T, E, N := x.Dim(0), w.Dim(0), w.Dim(1)
	ids := tensor.MustSlice[int64](route.TopKIDs)
	ws := tensor.MustSlice[float32](route.Weights)
	xs := tensor.MustSlice[float32](x)
	out := make([]float32, T*topk*N)
	K := x.Dim(1)
	for e := range E {
		we, _ := w.Narrow(0, e, 1)
		wc, _ := we.Contiguous()
		wk, _ := wc.Reshape(N, K)
		wf := tensor.MustSlice[float32](wk)
		for tok := range T {
			for s := range topk {
				if int(ids[tok*topk+s]) != e {
					continue
				}
				p := tok*topk + s
				scale := float32(1)
				if weighted {
					scale = ws[p]
				}
				for n := range N {
					var acc float32
					for k := range K {
						acc += xs[tok*K+k] * wf[n*K+k]
					}
					out[p*N+n] = acc * scale
				}
			}
		}
	}
	return out
}

func TestMoEGemmMatchesExpertLoop(t *testing.T) {
	cpu := device.CPU()
	T, E, K, N, topk := 7, 4, 16, 12, 2
	x := fillT(t, []int{T, K}, 0.2, cpu)
	w := fillT(t, []int{E, N, K}, 0.5, cpu)
	logits := fillT(t, []int{T, E}, 0.9, cpu)
	probs, _ := ops.Softmax(logits, -1)
	route, err := ops.MoERoute(probs, topk, true)
	if err != nil {
		t.Fatal(err)
	}
	// Normalised weights sum to 1 per token.
	ws := tensor.MustSlice[float32](route.Weights)
	for tok := range T {
		var s float32
		for k := range topk {
			s += ws[tok*topk+k]
		}
		if math.Abs(float64(s-1)) > 1e-5 {
			t.Errorf("token %d weights sum %v", tok, s)
		}
	}
	// Sorted experts are non-decreasing and pairs are a permutation.
	se := tensor.MustSlice[int64](route.SortedExpts)
	sp := tensor.MustSlice[int64](route.SortedPairs)
	seen := make([]bool, T*topk)
	for i := range se {
		if i > 0 && se[i] < se[i-1] {
			t.Fatalf("sorted experts not sorted at %d", i)
		}
		seen[sp[i]] = true
	}
	for p, ok := range seen {
		if !ok {
			t.Fatalf("pair %d missing from SortedPairs", p)
		}
	}
	ref := moeReference(t, x, w, route, topk, false)
	out, err := ops.MoEGemm(x, w, route, nil, topk)
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "moe_gemm per-token input", f32s(t, out), ref, 1e-5)
	// Per-pair input with routing weights.
	xp := fillT(t, []int{T * topk, K}, 0.4, cpu)
	refW := make([]float32, T*topk*N)
	{
		xs := tensor.MustSlice[float32](xp)
		wf := tensor.MustSlice[float32](w)
		ids := tensor.MustSlice[int64](route.TopKIDs)
		for p := range T * topk {
			e := int(ids[p])
			for n := range N {
				var acc float32
				for k := range K {
					acc += xs[p*K+k] * wf[(e*N+n)*K+k]
				}
				refW[p*N+n] = acc * ws[p]
			}
		}
	}
	outW, err := ops.MoEGemm(xp, w, route, route.Weights, topk)
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "moe_gemm per-pair weighted", f32s(t, outW), refW, 1e-5)
	comb, err := ops.MoECombine(outW, topk)
	if err != nil {
		t.Fatal(err)
	}
	cs := f32s(t, comb)
	for tok := range T {
		for n := range N {
			want := refW[(tok*topk)*N+n] + refW[(tok*topk+1)*N+n]
			if math.Abs(float64(cs[tok*N+n]-want)) > 1e-5 {
				t.Fatalf("combine [%d,%d] = %v want %v", tok, n, cs[tok*N+n], want)
			}
		}
	}
}

func TestMoEGemmCUDAParity(t *testing.T) {
	gpu := cudaOrSkip(t)
	cpu := device.CPU()
	// Larger than one tile in every direction: 70 pairs, N=50, K=48.
	T, E, K, N, topk := 35, 6, 48, 50, 2
	x := fillT(t, []int{T, K}, 0.2, cpu)
	w := fillT(t, []int{E, N, K}, 0.5, cpu)
	logits := fillT(t, []int{T, E}, 0.9, cpu)
	probs, _ := ops.Softmax(logits, -1)
	for _, dt := range []dtype.DType{dtype.F32, dtype.BF16, dtype.F16} {
		xd, _ := ops.Cast(x, dt)
		wd, _ := ops.Cast(w, dt)
		run := func(dev device.Device) ([]float32, []float32) {
			xx, _ := xd.To(dev)
			ww, _ := wd.To(dev)
			pp, _ := probs.To(dev)
			route, err := ops.MoERoute(pp, topk, true)
			if err != nil {
				t.Fatalf("route on %v: %v", dev, err)
			}
			a, err := ops.MoEGemm(xx, ww, route, nil, topk)
			if err != nil {
				t.Fatalf("%v moe_gemm on %v: %v", dt, dev, err)
			}
			// Second GEMM shape: per-pair input (M, N) against (E, K, N) weights.
			wt, _ := ww.Permute(0, 2, 1)
			wt, _ = wt.Contiguous()
			b, err := ops.MoEGemm(a, wt, route, route.Weights, topk)
			if err != nil {
				t.Fatalf("%v moe_gemm 2 on %v: %v", dt, dev, err)
			}
			c, err := ops.MoECombine(b, topk)
			if err != nil {
				t.Fatal(err)
			}
			return f32s(t, a), f32s(t, c)
		}
		ca, cc := run(cpu)
		ga, gc := run(gpu)
		tol := 1e-4
		if dt != dtype.F32 {
			tol = 3e-2
		}
		if e := relErr(ga, ca); e > tol {
			t.Errorf("%v moe_gemm CPU/CUDA rel err %.4g", dt, e)
		}
		if e := relErr(gc, cc); e > tol {
			t.Errorf("%v moe pipeline CPU/CUDA rel err %.4g", dt, e)
		}
	}
}

// Quantized experts: the fused CUDA kernels against the dequantize +
// MoEGemm reference (which is also the CPU path), all six formats.
func TestMoEGemmQCUDAParity(t *testing.T) {
	gpu := cudaOrSkip(t)
	cpu := device.CPU()
	T, E, K, N, topk := 21, 3, 256, 40, 2
	x := fillT(t, []int{T, K}, 0.2, cpu)
	logits := fillT(t, []int{T, E}, 0.9, cpu)
	probs, _ := ops.Softmax(logits, -1)
	xp := fillT(t, []int{T * topk, K}, 0.4, cpu)
	seed := uint32(7)
	next := func() byte { seed = seed*1664525 + 1013904223; return byte(seed >> 24) }
	for _, dt := range []dtype.DType{dtype.Q4_0, dtype.Q5_0, dtype.Q8_0, dtype.Q4_K, dtype.Q5_K, dtype.Q6_K} {
		var w *tensor.Tensor
		if dt == dtype.Q4_K || dt == dtype.Q5_K || dt == dtype.Q6_K {
			// Random block bytes with the f16 scale fields pinned to finite values.
			bb := dt.BlockBytes()
			nb := E * N * K / dt.BlockElems()
			buf := make([]byte, nb*bb)
			for i := range buf {
				buf[i] = next()
			}
			for b := range nb {
				blk := buf[b*bb : (b+1)*bb]
				if dt == dtype.Q6_K {
					blk[208], blk[209] = 0x00, 0x2C // d = 2^-4
				} else {
					blk[0], blk[1] = 0x00, 0x2C // d
					blk[2], blk[3] = 0x00, 0x24 // dmin = 2^-6
				}
			}
			var err error
			if w, err = tensor.FromBytes(buf, dt, []int{E, N, K}, cpu); err != nil {
				t.Fatal(err)
			}
		} else {
			wf := fillT(t, []int{E, N, K}, 0.5, cpu)
			var err error
			if w, err = quant.Quantize(wf, dt); err != nil {
				t.Fatal(err)
			}
		}
		run := func(dev device.Device) ([]float32, []float32) {
			xx, _ := x.To(dev)
			xpp, _ := xp.To(dev)
			ww, _ := w.To(dev)
			pp, _ := probs.To(dev)
			route, err := ops.MoERoute(pp, topk, true)
			if err != nil {
				t.Fatal(err)
			}
			a, err := ops.MoEGemmQ(xx, ww, route, nil, topk)
			if err != nil {
				t.Fatalf("%v moe_gemm_q on %v: %v", dt, dev, err)
			}
			b, err := ops.MoEGemmQ(xpp, ww, route, route.Weights, topk)
			if err != nil {
				t.Fatalf("%v moe_gemm_q (per pair) on %v: %v", dt, dev, err)
			}
			return f32s(t, a), f32s(t, b)
		}
		ca, cb := run(cpu)
		ga, gb := run(gpu)
		if e := relErr(ga, ca); e > 1e-4 {
			t.Errorf("%v per-token: CPU/CUDA rel err %.4g", dt, e)
		}
		if e := relErr(gb, cb); e > 1e-4 {
			t.Errorf("%v per-pair weighted: CPU/CUDA rel err %.4g", dt, e)
		}
	}
}
