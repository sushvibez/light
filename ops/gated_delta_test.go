package ops_test

import (
	"math"
	"path/filepath"
	"testing"

	"light/device"
	"light/npy"
	"light/ops"
	"light/tensor"
)

// Golden fixtures from scripts/verify/ref_gated_delta.py: transformers'
// torch_recurrent_gated_delta_rule / torch_chunk_gated_delta_rule and
// causal_conv1d_fn on seeded random inputs.
const gatedDeltaDir = "testdata/gated_delta"

func gdLoadNpz(t *testing.T, name string) map[string]*tensor.Tensor {
	t.Helper()
	m, err := npy.ReadNpz(filepath.Join(gatedDeltaDir, name), device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func gdMaxDiff(t *testing.T, a, b *tensor.Tensor) float64 {
	t.Helper()
	if !a.Shape().Equal(b.Shape()) {
		t.Fatalf("shape %v vs %v", a.Shape(), b.Shape())
	}
	as, bs := tensor.MustSlice[float32](gdContig(t, a)), tensor.MustSlice[float32](gdContig(t, b))
	var d float64
	for i := range as {
		d = math.Max(d, math.Abs(float64(as[i]-bs[i])))
	}
	return d
}

func gdContig(t *testing.T, x *tensor.Tensor) *tensor.Tensor {
	t.Helper()
	c, err := x.Contiguous()
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func gdNarrow(t *testing.T, x *tensor.Tensor, dim, start, n int) *tensor.Tensor {
	t.Helper()
	v, err := x.Narrow(dim, start, n)
	if err != nil {
		t.Fatal(err)
	}
	return gdContig(t, v)
}

func TestGatedDeltaRuleMatchesTransformers(t *testing.T) {
	for _, name := range []string{
		"delta_recurrent_t1_state.npz", "delta_recurrent_t37.npz", "delta_chunked_t37_c16.npz",
		"delta_chunked_t150_state.npz", "delta_chunked_t20_d128.npz",
	} {
		t.Run(name, func(t *testing.T) {
			f := gdLoadNpz(t, name)
			meta := tensor.MustSlice[int64](f["meta"])
			opts := ops.GatedDeltaOptions{Mode: ops.GatedDeltaRecurrent, ChunkSize: int(meta[1]), QKL2Norm: meta[2] == 1}
			if meta[0] == 1 {
				opts.Mode = ops.GatedDeltaChunked
			}
			out, final, err := ops.GatedDeltaRule(f["q"], f["k"], f["v"], f["g"], f["beta"], f["state"], opts)
			if err != nil {
				t.Fatal(err)
			}
			if d := gdMaxDiff(t, out, f["out"]); d > 1e-5 {
				t.Errorf("output differs from transformers by %.2e", d)
			}
			if d := gdMaxDiff(t, final, f["final"]); d > 1e-5 {
				t.Errorf("final state differs from transformers by %.2e", d)
			}
		})
	}
}

// The two forms are the same function, and a sequence fed in pieces (chunked
// prefill, then one token at a time carrying the state, as the model
// decodes) equals one call over all of it.
func TestGatedDeltaRuleFormsAndSplitsAgree(t *testing.T) {
	f := gdLoadNpz(t, "delta_chunked_t150_state.npz")
	q, k, v, g, beta, state := f["q"], f["k"], f["v"], f["g"], f["beta"], f["state"]
	run := func(mode ops.GatedDeltaMode, q, k, v, g, beta, s *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
		out, fin, err := ops.GatedDeltaRule(q, k, v, g, beta, s, ops.GatedDeltaOptions{Mode: mode, QKL2Norm: true})
		if err != nil {
			t.Fatal(err)
		}
		return out, fin
	}
	recOut, recFin := run(ops.GatedDeltaRecurrent, q, k, v, g, beta, state)
	chOut, chFin := run(ops.GatedDeltaChunked, q, k, v, g, beta, state)
	if d := gdMaxDiff(t, recOut, chOut); d > 1e-5 {
		t.Errorf("recurrent vs chunked output: %.2e", d)
	}
	if d := gdMaxDiff(t, recFin, chFin); d > 1e-5 {
		t.Errorf("recurrent vs chunked state: %.2e", d)
	}

	T, split := q.Dim(1), 100
	piece := func(x *tensor.Tensor, start, n int) *tensor.Tensor { return gdNarrow(t, x, 1, start, n) }
	pre, s := run(ops.GatedDeltaAuto, piece(q, 0, split), piece(k, 0, split), piece(v, 0, split), piece(g, 0, split), piece(beta, 0, split), state)
	if d := gdMaxDiff(t, pre, gdNarrow(t, chOut, 1, 0, split)); d > 1e-5 {
		t.Errorf("prefill piece: %.2e", d)
	}
	for i := split; i < T; i++ {
		var o *tensor.Tensor
		o, s = run(ops.GatedDeltaAuto, piece(q, i, 1), piece(k, i, 1), piece(v, i, 1), piece(g, i, 1), piece(beta, i, 1), s)
		if d := gdMaxDiff(t, o, gdNarrow(t, chOut, 1, i, 1)); d > 1e-5 {
			t.Fatalf("decode step %d: %.2e", i, d)
		}
	}
	if d := gdMaxDiff(t, s, chFin); d > 1e-5 {
		t.Errorf("state after prefill + decode: %.2e", d)
	}
}

func TestGatedDeltaRuleErrors(t *testing.T) {
	cpu := device.CPU()
	q, _ := tensor.New(make([]float32, 2*3*4), []int{1, 2, 3, 4}, cpu)
	v, _ := tensor.New(make([]float32, 2*3*5), []int{1, 2, 3, 5}, cpu)
	g, _ := tensor.New(make([]float32, 2*3), []int{1, 2, 3}, cpu)
	bad, _ := tensor.New(make([]float32, 2*2), []int{1, 2, 2}, cpu)
	if _, _, err := ops.GatedDeltaRule(q, q, v, g, bad, nil, ops.GatedDeltaOptions{}); err == nil {
		t.Error("beta with the wrong head count accepted")
	}
	st, _ := tensor.New(make([]float32, 3*4*4), []int{1, 3, 4, 4}, cpu)
	if _, _, err := ops.GatedDeltaRule(q, q, v, g, g, st, ops.GatedDeltaOptions{}); err == nil {
		t.Error("state with the wrong dv accepted")
	}
	q64, _ := tensor.New(make([]float64, 2*3*4), []int{1, 2, 3, 4}, cpu)
	if _, _, err := ops.GatedDeltaRule(q64, q64, v, g, g, nil, ops.GatedDeltaOptions{}); err == nil {
		t.Error("F64 input accepted")
	}
}

func TestCausalConv1dDepthwiseMatchesTransformers(t *testing.T) {
	for _, name := range []string{"conv_k4.npz", "conv_k4_t2.npz"} {
		t.Run(name, func(t *testing.T) {
			f := gdLoadNpz(t, name)
			out, _, err := ops.CausalConv1dDepthwise(f["x"], f["w"], nil, true)
			if err != nil {
				t.Fatal(err)
			}
			if d := gdMaxDiff(t, out, f["out"]); d > 1e-6 {
				t.Errorf("differs from transformers by %.2e", d)
			}
		})
	}
	// Prefill 7 steps, then one at a time through the carried state.
	f := gdLoadNpz(t, "conv_k4.npz")
	x, w, want := f["x"], f["w"], f["out"]
	o, s, err := ops.CausalConv1dDepthwise(gdNarrow(t, x, 2, 0, 7), w, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	if d := gdMaxDiff(t, o, gdNarrow(t, want, 2, 0, 7)); d > 1e-6 {
		t.Errorf("prefill piece: %.2e", d)
	}
	for i := 7; i < x.Dim(2); i++ {
		if o, s, err = ops.CausalConv1dDepthwise(gdNarrow(t, x, 2, i, 1), w, s, true); err != nil {
			t.Fatal(err)
		}
		if d := gdMaxDiff(t, o, gdNarrow(t, want, 2, i, 1)); d > 1e-6 {
			t.Fatalf("step %d: %.2e", i, d)
		}
	}
}

func TestSoftplus(t *testing.T) {
	in, _ := tensor.New([]float32{-30, -1, 0, 1, 19.5, 25}, []int{6}, device.CPU())
	out, err := ops.Softplus(in)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)
	for i, x := range []float64{-30, -1, 0, 1, 19.5, 25} {
		want := math.Log1p(math.Exp(x))
		if x > 20 {
			want = x
		}
		if math.Abs(float64(got[i])-want) > 1e-6*math.Max(1, math.Abs(want)) {
			t.Errorf("softplus(%v) = %v, want %v", x, got[i], want)
		}
	}
}
