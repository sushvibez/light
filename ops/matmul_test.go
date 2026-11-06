package ops_test

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

func TestMatMulSimple(t *testing.T) {
	// (2,3) × (3,2) → (2,2)
	// [[1,2,3],[4,5,6]] × [[1,2],[3,4],[5,6]]
	//  = [[1+6+15, 2+8+18],[4+15+30, 8+20+36]] = [[22, 28],[49, 64]]
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	b, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}, device.CPU())
	c, err := ops.MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{22, 28, 49, 64}
	if !slices.Equal(tensor.MustSlice[float32](c), want) {
		t.Errorf("got %v, want %v", tensor.MustSlice[float32](c), want)
	}
}

func TestMatMulTransposed(t *testing.T) {
	// Verify matmul works with a transposed (non-contiguous) operand.
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	b, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	bt, _ := b.Transpose(0, 1) // (3,2), non-contiguous
	c, err := ops.MatMul(a, bt)
	if err != nil {
		t.Fatal(err)
	}
	// a × b^T:
	// [[1*1+2*2+3*3, 1*4+2*5+3*6],[4*1+5*2+6*3, 4*4+5*5+6*6]]
	// = [[14, 32], [32, 77]]
	want := []float32{14, 32, 32, 77}
	if !slices.Equal(tensor.MustSlice[float32](c), want) {
		t.Errorf("got %v, want %v", tensor.MustSlice[float32](c), want)
	}
}

func TestMatMulErrors(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU()) // 1-D
	b, _ := tensor.New([]float32{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	if _, err := ops.MatMul(a, b); err == nil {
		t.Error("expected rank error")
	}

	a2, _ := tensor.New([]float32{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	b2, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}, device.CPU())
	if _, err := ops.MatMul(a2, b2); err == nil {
		t.Error("expected inner-dim mismatch")
	}

	// dtype mismatch
	a3, _ := tensor.New([]float32{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	b3, _ := tensor.New([]float64{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	if _, err := ops.MatMul(a3, b3); err == nil {
		t.Error("expected dtype mismatch")
	}
}

func TestMatMulI64(t *testing.T) {
	a, _ := tensor.New([]int64{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	b, _ := tensor.New([]int64{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	c, err := ops.MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	// [[1,2],[3,4]] × [[1,2],[3,4]] = [[7,10],[15,22]]
	want := []int64{7, 10, 15, 22}
	if !slices.Equal(tensor.MustSlice[int64](c), want) {
		t.Errorf("got %v, want %v", tensor.MustSlice[int64](c), want)
	}
}

func TestMatMulF16(t *testing.T) {
	a, err := tensor.Ones(dtype.F16, []int{2, 2}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	b, _ := tensor.Ones(dtype.F16, []int{2, 2}, device.CPU())
	c, err := ops.MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	halves, err := tensor.ToHalfSlice(c)
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range halves {
		if math.Abs(float64(h.ToF32())-2.0) > 1e-2 {
			t.Errorf("F16 matmul: %v, want 2", h.ToF32())
		}
	}
}

func TestMatMulBF16(t *testing.T) {
	a, _ := tensor.Ones(dtype.BF16, []int{2, 2}, device.CPU())
	b, _ := tensor.Ones(dtype.BF16, []int{2, 2}, device.CPU())
	c, err := ops.MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	bfs, _ := tensor.ToBF16Slice(c)
	for _, bf := range bfs {
		if math.Abs(float64(bf.ToF32())-2.0) > 1e-2 {
			t.Errorf("BF16 matmul: %v, want 2", bf.ToF32())
		}
	}
}

func TestBatchedMatMulF32(t *testing.T) {
	// (2, 2, 3) @ (2, 3, 2) → (2, 2, 2)
	a, _ := tensor.New([]float32{
		// batch 0:
		1, 2, 3,
		4, 5, 6,
		// batch 1:
		7, 8, 9,
		10, 11, 12,
	}, []int{2, 2, 3}, device.CPU())
	b, _ := tensor.New([]float32{
		// batch 0:
		1, 0,
		0, 1,
		1, 1,
		// batch 1:
		2, 0,
		0, 2,
		1, 1,
	}, []int{2, 3, 2}, device.CPU())
	c, err := ops.MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if c.Rank() != 3 || c.Dim(0) != 2 || c.Dim(1) != 2 || c.Dim(2) != 2 {
		t.Errorf("shape = %v", c.Shape())
	}
	// batch 0:
	//   [[1*1+2*0+3*1, 1*0+2*1+3*1], [4*1+5*0+6*1, 4*0+5*1+6*1]] = [[4,5],[10,11]]
	// batch 1:
	//   [[7*2+8*0+9*1, 7*0+8*2+9*1], [10*2+11*0+12*1, 10*0+11*2+12*1]] = [[23,25],[32,34]]
	want := []float32{4, 5, 10, 11, 23, 25, 32, 34}
	got := tensor.MustSlice[float32](c)
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d] got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestBatchedMatMulErrors(t *testing.T) {
	a, _ := tensor.Ones(dtype.F32, []int{2, 3, 4}, device.CPU())
	b, _ := tensor.Ones(dtype.F32, []int{3, 4, 5}, device.CPU())
	if _, err := ops.MatMul(a, b); err == nil {
		t.Error("expected batch-dim mismatch error")
	}
	b2, _ := tensor.Ones(dtype.F32, []int{2, 5, 7}, device.CPU())
	if _, err := ops.MatMul(a, b2); err == nil {
		t.Error("expected inner-dim mismatch error")
	}
	// Mixed ranks
	b3, _ := tensor.Ones(dtype.F32, []int{4, 5}, device.CPU())
	if _, err := ops.MatMul(a, b3); err == nil {
		t.Error("expected rank mismatch error")
	}
}

// ---- F16 / BF16 GEMMs on CUDA, FP8 GEMM (2026-10-28) ----

func matmulFill(t *testing.T, dims []int, seed float64, dt dtype.DType, dev device.Device) *tensor.Tensor {
	t.Helper()
	x := fillT(t, dims, seed, device.CPU())
	if dt != dtype.F32 {
		var err error
		if x, err = ops.Cast(x, dt); err != nil {
			t.Fatal(err)
		}
	}
	g, err := x.To(dev)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func relErr(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		num += float64((got[i] - want[i]) * (got[i] - want[i]))
		den += float64(want[i] * want[i])
	}
	return math.Sqrt(num / math.Max(den, 1e-20))
}

func TestMatMulHalfCUDAParity(t *testing.T) {
	gpu := cudaOrSkip(t)
	for _, dt := range []dtype.DType{dtype.F16, dtype.BF16} {
		a := matmulFill(t, []int{5, 8}, 0.1, dt, device.CPU())
		b := matmulFill(t, []int{8, 6}, 0.2, dt, device.CPU())
		w := matmulFill(t, []int{6, 8}, 0.3, dt, device.CPU())
		run := func(dev device.Device) (plain, trans, mixed []float32) {
			ad, _ := a.To(dev)
			bd, _ := b.To(dev)
			wd, _ := w.To(dev)
			p, err := ops.MatMul(ad, bd)
			if err != nil {
				t.Fatalf("%v plain on %v: %v", dt, dev, err)
			}
			wT, _ := wd.Transpose(0, 1)
			tr, err := ops.MatMul(ad, wT)
			if err != nil {
				t.Fatalf("%v transB on %v: %v", dt, dev, err)
			}
			mx, err := ops.MatMulMixed(ad, wT)
			if err != nil {
				t.Fatalf("%v mixed on %v: %v", dt, dev, err)
			}
			return f32s(t, p), f32s(t, tr), f32s(t, mx)
		}
		cp, ct, cm := run(device.CPU())
		gp, gt, gm := run(gpu)
		for name, pair := range map[string][2][]float32{"plain": {cp, gp}, "transB": {ct, gt}, "mixed": {cm, gm}} {
			if e := relErr(pair[1], pair[0]); e > 5e-3 {
				t.Errorf("%v %s: CPU/CUDA rel err %.4g", dt, name, e)
			}
		}
		// Batched (3, 4, 8) × (3, 8, 5).
		ba := matmulFill(t, []int{3, 4, 8}, 0.4, dt, device.CPU())
		bb := matmulFill(t, []int{3, 8, 5}, 0.5, dt, device.CPU())
		cb, err := ops.MatMul(ba, bb)
		if err != nil {
			t.Fatal(err)
		}
		bag, _ := ba.To(gpu)
		bbg, _ := bb.To(gpu)
		gb, err := ops.MatMul(bag, bbg)
		if err != nil {
			t.Fatalf("%v batched CUDA: %v", dt, err)
		}
		if e := relErr(f32s(t, gb), f32s(t, cb)); e > 5e-3 {
			t.Errorf("%v batched: CPU/CUDA rel err %.4g", dt, e)
		}
	}
	// F32 activation × F16 weight → F32 (mixed), against the F32 GEMM.
	a32 := fillT(t, []int{5, 8}, 0.1, device.CPU())
	w32 := fillT(t, []int{6, 8}, 0.3, device.CPU())
	w32T, _ := w32.Transpose(0, 1)
	ref, _ := ops.MatMul(a32, w32T)
	a32g, _ := a32.To(gpu)
	w16g, _ := matmulFill(t, []int{6, 8}, 0.3, dtype.F16, device.CPU()).To(gpu)
	w16T, _ := w16g.Transpose(0, 1)
	mixed, err := ops.MatMulMixed(a32g, w16T)
	if err != nil {
		t.Fatal(err)
	}
	if mixed.DType() != dtype.F32 {
		t.Fatalf("mixed F32×F16 dtype %v", mixed.DType())
	}
	if e := relErr(f32s(t, mixed), f32s(t, ref)); e > 2e-3 {
		t.Errorf("mixed F32×F16: rel err %.4g vs F32", e)
	}
}

func TestMatMulF8(t *testing.T) {
	// M=8, K=32, N=16: K multiple of 16 as cuBLASLt requires.
	a := fillT(t, []int{8, 32}, 0.7, device.CPU())
	w := fillT(t, []int{16, 32}, 0.9, device.CPU())
	wT, _ := w.Transpose(0, 1)
	ref, err := ops.MatMul(a, wT)
	if err != nil {
		t.Fatal(err)
	}
	aq, aScale, err := ops.QuantizeF8(a)
	if err != nil {
		t.Fatal(err)
	}
	wq, wScale, err := ops.QuantizeF8(w)
	if err != nil {
		t.Fatal(err)
	}
	if aq.DType() != dtype.F8E4M3 || aScale <= 0 || aScale > 448 {
		t.Fatalf("quantize: dtype %v scale %v", aq.DType(), aScale)
	}
	wqT, _ := wq.Transpose(0, 1)
	cpuOut, err := ops.MatMulF8(aq, wqT, aScale, wScale, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	refS, cpuS := f32s(t, ref), f32s(t, cpuOut)
	if e := relErr(cpuS, refS); e > 0.05 {
		t.Errorf("F8 matmul vs F32 reference: rel err %.4g", e)
	}
	// Also the (K, N) contiguous weight layout.
	wqKN, _ := wqT.Contiguous()
	cpuKN, err := ops.MatMulF8(aq, wqKN, aScale, wScale, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if e := relErr(f32s(t, cpuKN), cpuS); e > 1e-6 {
		t.Errorf("F8 (K,N) layout differs: %.4g", e)
	}

	gpu := cudaOrSkip(t)
	if major, minor, err := device.CUDAComputeCapability(gpu.Ordinal()); err != nil || major*10+minor < 89 {
		t.Skipf("CUDA FP8 GEMM needs compute capability 8.9+; GPU is %d.%d (%v)", major, minor, err)
	}
	aqg, _ := aq.To(gpu)
	wqg, _ := wq.To(gpu)
	wqgT, _ := wqg.Transpose(0, 1)
	for _, outDT := range []dtype.DType{dtype.F32, dtype.BF16, dtype.F16} {
		g, err := ops.MatMulF8(aqg, wqgT, aScale, wScale, outDT)
		if err != nil {
			t.Fatalf("cuda F8 → %v: %v", outDT, err)
		}
		c, err := ops.MatMulF8(aq, wqT, aScale, wScale, outDT)
		if err != nil {
			t.Fatal(err)
		}
		tol := 1e-4
		if outDT != dtype.F32 {
			tol = 1e-2
		}
		if e := relErr(f32s(t, g), f32s(t, c)); e > tol {
			t.Errorf("cuda F8 → %v: CPU/CUDA rel err %.4g", outDT, e)
		}
	}
	// (K, N) contiguous layout on CUDA (transposed into (N, K) internally).
	wqgKN, _ := wqgT.Contiguous()
	gKN, err := ops.MatMulF8(aqg, wqgKN, aScale, wScale, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if e := relErr(f32s(t, gKN), cpuS); e > 1e-4 {
		t.Errorf("cuda F8 (K,N) layout: rel err %.4g", e)
	}
	// Quantisation on the GPU gives the same codes and scale.
	ag, _ := a.To(gpu)
	aqg2, s2, err := ops.QuantizeF8(ag)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(s2-aScale)) > 1e-7 {
		t.Errorf("gpu scale %v vs cpu %v", s2, aScale)
	}
	back, _ := ops.Cast(aqg2, dtype.F32)
	cpuBack, _ := ops.Cast(aq, dtype.F32)
	expectRow(t, "gpu quantize codes", f32s(t, back), f32s(t, cpuBack), 0)
}

// Batched matmul must honour the strides of transposed / narrowed views on
// both devices (CPU read kᵀ views as if contiguous until 2026-10-29, which
// corrupted every CPU attention score for sequences longer than one).
func TestMatMulBatchedViewsParity(t *testing.T) {
	const B, T, HD = 3, 5, 8
	data := make([]float32, B*T*HD)
	for i := range data {
		data[i] = float32((i*37)%17) / 7
	}
	for _, dt := range []dtype.DType{dtype.F32, dtype.BF16} {
		devs := []device.Device{device.CPU()}
		if device.CUDAAvailable() {
			g, _ := device.CUDA(0)
			devs = append(devs, g)
		}
		var ref []float32
		for _, dev := range devs {
			q, _ := tensor.New(data, []int{B, T, HD}, dev)
			k, _ := tensor.New(data, []int{B, T, HD}, dev)
			if dt != dtype.F32 {
				q, _ = ops.Cast(q, dt)
				k, _ = ops.Cast(k, dt)
			}
			kT, _ := k.Transpose(1, 2) // (B, HD, T) view
			qN, _ := q.Narrow(1, 1, 3) // (B, 3, HD) view with offset
			view, err := ops.MatMul(qN, kT)
			if err != nil {
				t.Fatal(err)
			}
			kTc, _ := kT.Contiguous()
			qNc, _ := qN.Contiguous()
			want, _ := ops.MatMul(qNc, kTc)
			got := f32s(t, view)
			expectRow(t, fmt.Sprintf("%v %v view vs contiguous", dev, dt), got, f32s(t, want), 0)
			if ref == nil {
				ref = got
			} else {
				expectRow(t, fmt.Sprintf("%v cpu vs cuda", dt), got, ref, 1e-2)
			}
		}
	}
}
