package nn_test

import (
	"math"
	"testing"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

func f32s(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	c, err := x.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	return tensor.MustSlice[float32](c)
}

// splitHeads turns (S, H) into (1, NH, S, HD).
func splitHeads(t *testing.T, x *tensor.Tensor, S, NH, HD int) *tensor.Tensor {
	t.Helper()
	r, err := autograd.Reshape(x, 1, S, NH, HD)
	if err != nil {
		t.Fatal(err)
	}
	p, err := autograd.Permute(r, 0, 2, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestSDPAMatchesMultiHeadAttention(t *testing.T) {
	cpu := device.CPU()
	const S, H, NH = 6, 16, 4
	const HD = H / NH
	for _, causal := range []bool{false, true} {
		m, err := nn.NewMultiHeadAttention(H, NH, causal, true, cpu)
		if err != nil {
			t.Fatal(err)
		}
		x, _ := tensor.RandN(dtype.F32, []int{S, H}, cpu)
		want, err := m.Forward(x)
		if err != nil {
			t.Fatal(err)
		}
		// Same math, spelled with SDPA + the module's own projections.
		q, _ := m.QProj.Forward(x)
		k, _ := m.KProj.Forward(x)
		v, _ := m.VProj.Forward(x)
		var mask *tensor.Tensor
		if causal {
			if mask, err = nn.CausalMask(S, cpu); err != nil {
				t.Fatal(err)
			}
		}
		ctx, err := nn.SDPA(splitHeads(t, q, S, NH, HD), splitHeads(t, k, S, NH, HD), splitHeads(t, v, S, NH, HD), mask)
		if err != nil {
			t.Fatal(err)
		}
		if ctx.Rank() != 4 || ctx.Dim(0) != 1 || ctx.Dim(1) != NH || ctx.Dim(2) != S || ctx.Dim(3) != HD {
			t.Fatalf("SDPA shape %v", ctx.Shape())
		}
		merged, _ := autograd.Permute(ctx, 0, 2, 1, 3)
		mergedC, _ := autograd.Contiguous(merged)
		flat, _ := autograd.Reshape(mergedC, S, H)
		got, err := m.OProj.Forward(flat)
		if err != nil {
			t.Fatal(err)
		}
		gs, ws := f32s(t, got), f32s(t, want)
		for i := range ws {
			if math.Abs(float64(gs[i]-ws[i])) > 1e-5 {
				t.Fatalf("causal=%v [%d]: SDPA path %v vs MHA %v", causal, i, gs[i], ws[i])
			}
		}
	}
}

func TestSDPACausalVsManual(t *testing.T) {
	cpu := device.CPU()
	q, _ := tensor.New([]float32{1, 0, 0, 1, 1, 1}, []int{1, 1, 3, 2}, cpu)
	k, _ := tensor.New([]float32{1, 0, 0, 1, 1, 1}, []int{1, 1, 3, 2}, cpu)
	v, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{1, 1, 3, 2}, cpu)
	mask, _ := nn.CausalMask(3, cpu)
	out, err := nn.SDPA(q, k, v, mask)
	if err != nil {
		t.Fatal(err)
	}
	o := f32s(t, out)
	// Row 0 attends only to key 0 → v[0] = [1, 2].
	if math.Abs(float64(o[0]-1)) > 1e-5 || math.Abs(float64(o[1]-2)) > 1e-5 {
		t.Errorf("row0 = %v,%v want 1,2", o[0], o[1])
	}
	// Row 1: scores q1·k0 = 0, q1·k1 = 1 (scaled by 1/√2) → softmax weights.
	s := 1 / math.Sqrt2
	w1 := math.Exp(s) / (1 + math.Exp(s))
	want2 := (1-w1)*1 + w1*3
	if math.Abs(float64(o[2])-want2) > 1e-5 {
		t.Errorf("row1[0] = %v want %v", o[2], want2)
	}
	// Different kLen than qLen with a (Sq, Sk) mask and a batch/head mask.
	kk, _ := tensor.RandN(dtype.F32, []int{2, 3, 5, 4}, cpu)
	qq, _ := tensor.RandN(dtype.F32, []int{2, 3, 2, 4}, cpu)
	if _, err := nn.SDPA(qq, kk, kk, nil); err != nil {
		t.Fatal(err)
	}
	m4, _ := tensor.Zeros(dtype.F32, []int{2, 1, 2, 5}, cpu)
	if _, err := nn.SDPA(qq, kk, kk, m4); err != nil {
		t.Fatalf("(B,1,Sq,Sk) mask: %v", err)
	}
	bad, _ := tensor.Zeros(dtype.F32, []int{2, 3, 4, 5}, cpu)
	if _, err := nn.SDPA(qq, bad, bad, nil); err == nil {
		t.Error("expected k/q mismatch error")
	}
}

func TestSlidingWindowMask(t *testing.T) {
	cpu := device.CPU()
	m, err := nn.SlidingWindowMask(4, 4, 2, cpu)
	if err != nil {
		t.Fatal(err)
	}
	d := f32s(t, m)
	// Row 2 attends to j ∈ {1, 2}; j=0 (<= 2-2) and j=3 (> 2) are masked.
	expectMasked := map[int]bool{0: true, 3: true}
	for j := range 4 {
		masked := math.IsInf(float64(d[2*4+j]), -1)
		if masked != expectMasked[j] {
			t.Errorf("row2 j=%d masked=%v want %v", j, masked, expectMasked[j])
		}
	}
	// KV-cache alignment: qLen=1, kLen=6, window=3 → only keys 3,4,5 open.
	m2, _ := nn.SlidingWindowMask(1, 6, 3, cpu)
	d2 := f32s(t, m2)
	for j := range 6 {
		masked := math.IsInf(float64(d2[j]), -1)
		if masked != (j < 3) {
			t.Errorf("cache row j=%d masked=%v", j, masked)
		}
	}

	// The window must actually change attention: with a full causal mask,
	// query 3 sees key 0; with window=2 it must not.
	const S, D = 4, 8
	q, _ := tensor.RandN(dtype.F32, []int{1, 1, S, D}, cpu)
	k, _ := tensor.RandN(dtype.F32, []int{1, 1, S, D}, cpu)
	// Make v a one-hot per key so the output row is literally the attention weights.
	vData := make([]float32, S*D)
	for j := range S {
		vData[j*D+j] = 1
	}
	v, _ := tensor.New(vData, []int{1, 1, S, D}, cpu)
	causal, _ := nn.CausalMask(S, cpu)
	full, _ := nn.SDPA(q, k, v, causal)
	swa, _ := nn.SDPA(q, k, v, m)
	fw, sw := f32s(t, full), f32s(t, swa)
	// Row 3: causal gives weight > 0 on key 0 and 1; SWA(window=2) gives 0.
	if fw[3*D+0] <= 0 || fw[3*D+1] <= 0 {
		t.Errorf("causal row3 should attend keys 0,1: %v", fw[3*D:3*D+S])
	}
	if sw[3*D+0] != 0 || sw[3*D+1] != 0 {
		t.Errorf("SWA row3 must not attend keys 0,1: %v", sw[3*D:3*D+S])
	}
	if sw[3*D+2] <= 0 || sw[3*D+3] <= 0 {
		t.Errorf("SWA row3 should attend keys 2,3: %v", sw[3*D:3*D+S])
	}
	var sum float32
	for j := range S {
		sum += sw[3*D+j]
	}
	if math.Abs(float64(sum-1)) > 1e-5 {
		t.Errorf("SWA row3 weights sum to %v", sum)
	}
	// Row 0 is identical under both masks (window never bites).
	for j := range S {
		if math.Abs(float64(fw[j]-sw[j])) > 1e-6 {
			t.Errorf("row0 differs at %d: %v vs %v", j, fw[j], sw[j])
		}
	}
}

func TestALiBiSlopes(t *testing.T) {
	s8 := nn.ALiBiSlopes(8)
	want8 := []float64{1.0 / 2, 1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32, 1.0 / 64, 1.0 / 128, 1.0 / 256}
	for i := range want8 {
		if math.Abs(s8[i]-want8[i]) > 1e-12 {
			t.Errorf("slopes(8)[%d] = %v want %v", i, s8[i], want8[i])
		}
	}
	// 12 heads: the 8-head sequence plus every other slope of the 16-head one.
	s12 := nn.ALiBiSlopes(12)
	want12 := append(append([]float64{}, want8...),
		math.Pow(2, -0.5), math.Pow(2, -1.5), math.Pow(2, -2.5), math.Pow(2, -3.5))
	if len(s12) != 12 {
		t.Fatalf("slopes(12) len %d", len(s12))
	}
	for i := range want12 {
		if math.Abs(s12[i]-want12[i]) > 1e-12 {
			t.Errorf("slopes(12)[%d] = %v want %v", i, s12[i], want12[i])
		}
	}
	s16 := nn.ALiBiSlopes(16)
	if math.Abs(s16[0]-math.Pow(2, -0.5)) > 1e-12 || math.Abs(s16[15]-math.Pow(2, -8)) > 1e-12 {
		t.Errorf("slopes(16) = %v", s16)
	}
	if nn.ALiBiSlopes(0) != nil {
		t.Error("slopes(0) should be nil")
	}
}

func TestALiBiBias(t *testing.T) {
	cpu := device.CPU()
	b, err := nn.ALiBiBias(2, 3, 3, cpu)
	if err != nil {
		t.Fatal(err)
	}
	if b.Rank() != 3 || b.Dim(0) != 2 || b.Dim(1) != 3 || b.Dim(2) != 3 {
		t.Fatalf("shape %v", b.Shape())
	}
	d := f32s(t, b)
	// numHeads=2: slope[0] = 2^(-8/2) = 0.0625 → bias[0,2,0] = -0.0625*2 = -0.125.
	if math.Abs(float64(d[0*9+2*3+0]-(-0.125))) > 1e-6 {
		t.Errorf("bias[0,2,0] = %v want -0.125", d[0*9+2*3+0])
	}
	if d[0*9+2*3+2] != 0 {
		t.Errorf("bias on the diagonal should be 0, got %v", d[0*9+2*3+2])
	}
	if !math.IsInf(float64(d[0*9+0*3+1]), -1) {
		t.Errorf("bias[0,0,1] should be causally masked, got %v", d[0*9+0*3+1])
	}
	// Head 1 slope = 2^-8 → bias[1,2,0] = -2/256.
	if math.Abs(float64(d[1*9+2*3+0]-(-2.0/256))) > 1e-7 {
		t.Errorf("bias[1,2,0] = %v", d[1*9+2*3+0])
	}
	// Feeds straight into SDPA as a per-head mask, broadcasting over batch.
	q, _ := tensor.RandN(dtype.F32, []int{2, 2, 3, 4}, cpu)
	out, err := nn.SDPA(q, q, q, b)
	if err != nil {
		t.Fatal(err)
	}
	for _, x := range f32s(t, out) {
		if math.IsNaN(float64(x)) {
			t.Fatal("NaN in ALiBi attention output")
		}
	}
	if _, err := nn.ALiBiBias(2, 4, 3, cpu); err == nil {
		t.Error("expected kLen < qLen error")
	}
}

func TestInstanceNorm2dModule(t *testing.T) {
	cpu := device.CPU()
	const N, C, H, W = 2, 3, 4, 5
	x, _ := tensor.RandN(dtype.F32, []int{N, C, H, W}, cpu)
	m, err := nn.NewInstanceNorm2d(C, 1e-5, cpu)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Parameters()) != 2 {
		t.Errorf("params = %d", len(m.Parameters()))
	}
	y, err := m.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	// Reference: per (n, c) plane normalisation with gamma=1, beta=0.
	xs, ys := f32s(t, x), f32s(t, y)
	for n := range N {
		for c := range C {
			base := (n*C + c) * H * W
			var mean float64
			for i := range H * W {
				mean += float64(xs[base+i])
			}
			mean /= float64(H * W)
			var vr float64
			for i := range H * W {
				d := float64(xs[base+i]) - mean
				vr += d * d
			}
			vr /= float64(H * W)
			inv := 1 / math.Sqrt(vr+1e-5)
			for i := range H * W {
				want := (float64(xs[base+i]) - mean) * inv
				if math.Abs(float64(ys[base+i])-want) > 1e-4 {
					t.Fatalf("[%d,%d,%d] = %v want %v", n, c, i, ys[base+i], want)
				}
			}
		}
	}
	// Equals GroupNorm with groups = C.
	gn, _ := ops.GroupNorm(x, m.Gamma, m.Beta, C, 1e-5)
	gs := f32s(t, gn)
	for i := range gs {
		if math.Abs(float64(gs[i]-ys[i])) > 1e-6 {
			t.Fatalf("GroupNorm(groups=C) differs at %d", i)
		}
	}
	if _, err := nn.NewInstanceNorm2d(0, 1e-5, cpu); err == nil {
		t.Error("expected channels error")
	}
}
