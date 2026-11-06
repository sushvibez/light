//go:build !nocuda

package ops_test

import (
	"math"
	"slices"
	"testing"
	"time"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/quant"
	"light/tensor"
)

func gpuOrSkip(t testing.TB) device.Device {
	t.Helper()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	return gpu
}

func toGPU(t testing.TB, gpu device.Device, ts ...*tensor.Tensor) []*tensor.Tensor {
	t.Helper()
	out := make([]*tensor.Tensor, len(ts))
	for i, x := range ts {
		g, err := x.To(gpu)
		if err != nil {
			t.Fatal(err)
		}
		out[i] = g
	}
	return out
}

func assertParityF32(t *testing.T, name string, cpuOut, gpuOut *tensor.Tensor, tol float64) {
	t.Helper()
	if gpuOut.Device().Kind() != device.KindCUDA {
		t.Fatalf("%s: output should live on CUDA", name)
	}
	if !cpuOut.Shape().Equal(gpuOut.Shape()) {
		t.Fatalf("%s: shape %v vs %v", name, cpuOut.Shape(), gpuOut.Shape())
	}
	back, err := gpuOut.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	cs := tensor.MustSlice[float32](cpuOut)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > tol {
			t.Fatalf("%s[%d]: CPU=%v GPU=%v", name, i, cs[i], gs[i])
		}
	}
}

func TestCUDAPadFlipRollParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	a, _ := tensor.RandN(dtype.F32, []int{3, 4, 5}, device.CPU())
	aG := toGPU(t, gpu, a)[0]

	for _, c := range []struct{ dim, l, r int }{{0, 1, 2}, {1, 0, 3}, {2, 2, 0}, {-1, 1, 1}} {
		c1, err := ops.Pad(a, c.dim, c.l, c.r)
		if err != nil {
			t.Fatal(err)
		}
		g1, err := ops.Pad(aG, c.dim, c.l, c.r)
		if err != nil {
			t.Fatal(err)
		}
		assertParityF32(t, "Pad", c1, g1, 0)
	}
	for _, dims := range [][]int{{0}, {1}, {2}, {0, 2}, {0, 1, 2}, {}} {
		c1, err := ops.Flip(a, dims...)
		if err != nil {
			t.Fatal(err)
		}
		g1, err := ops.Flip(aG, dims...)
		if err != nil {
			t.Fatal(err)
		}
		assertParityF32(t, "Flip", c1, g1, 0)
	}
	for _, c := range []struct{ shift, dim int }{{1, 0}, {-1, 1}, {7, 2}, {-9, -1}, {0, 0}} {
		c1, err := ops.Roll(a, c.shift, c.dim)
		if err != nil {
			t.Fatal(err)
		}
		g1, err := ops.Roll(aG, c.shift, c.dim)
		if err != nil {
			t.Fatal(err)
		}
		assertParityF32(t, "Roll", c1, g1, 0)
	}
}

func TestCUDASortFallbackParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	a, _ := tensor.RandN(dtype.F32, []int{4, 9}, device.CPU())
	aG := toGPU(t, gpu, a)[0]
	for _, desc := range []bool{false, true} {
		cv, ci, err := ops.Sort(a, desc)
		if err != nil {
			t.Fatal(err)
		}
		gv, gi, err := ops.Sort(aG, desc)
		if err != nil {
			t.Fatal(err)
		}
		if gi.Device().Kind() != device.KindCUDA || gi.DType() != dtype.I64 {
			t.Fatalf("Sort idx on %v dtype %v", gi.Device(), gi.DType())
		}
		assertParityF32(t, "Sort", cv, gv, 0)
		giC, _ := gi.To(device.CPU())
		if !slices.Equal(tensor.MustSlice[int64](ci), tensor.MustSlice[int64](giC)) {
			t.Fatalf("Sort idx mismatch")
		}
	}
}

func TestCUDANormParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	a, _ := tensor.RandN(dtype.F32, []int{3, 4, 5}, device.CPU())
	aG := toGPU(t, gpu, a)[0]
	for _, p := range []float64{1, 2, 3} {
		for _, opts := range []ops.ReduceOptions{{}, {Axes: []int{2}}, {Axes: []int{0}, KeepDims: true}, {Axes: []int{0, 2}}} {
			c1, err := ops.Norm(a, p, opts)
			if err != nil {
				t.Fatal(err)
			}
			g1, err := ops.Norm(aG, p, opts)
			if err != nil {
				t.Fatalf("p=%v %+v: %v", p, opts, err)
			}
			assertParityF32(t, "Norm", c1, g1, 1e-4)
		}
	}
}

func TestCUDAEinsumParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	cpu := device.CPU()
	mk := func(dims ...int) *tensor.Tensor {
		x, _ := tensor.RandN(dtype.F32, dims, cpu)
		return x
	}
	cases := []struct {
		notation string
		ops      []*tensor.Tensor
	}{
		{"ij,jk->ik", []*tensor.Tensor{mk(4, 5), mk(5, 6)}},
		{"ik,jk->ij", []*tensor.Tensor{mk(4, 5), mk(6, 5)}},
		{"bij,bjk->bik", []*tensor.Tensor{mk(3, 4, 5), mk(3, 5, 2)}},
		{"bhqd,bhkd->bhqk", []*tensor.Tensor{mk(2, 3, 4, 8), mk(2, 3, 6, 8)}},
		{"ij->ji", []*tensor.Tensor{mk(4, 5)}},
		{"ijk->kji", []*tensor.Tensor{mk(2, 3, 4)}},
		{"ij->i", []*tensor.Tensor{mk(4, 5)}},
		{"ij->", []*tensor.Tensor{mk(4, 5)}},
		{"i,i->", []*tensor.Tensor{mk(7), mk(7)}},
		{"i,j->ij", []*tensor.Tensor{mk(3), mk(4)}},
		{"ij,ij->ij", []*tensor.Tensor{mk(3, 4), mk(3, 4)}},
		{"ij,i->ij", []*tensor.Tensor{mk(3, 4), mk(3)}},
		{"ijk,jl->il", []*tensor.Tensor{mk(2, 3, 4), mk(3, 5)}}, // pre-sum k, then contract j
		// Host-fallback paths on CUDA (diagonal / 3 operands).
		{"ii->i", []*tensor.Tensor{mk(5, 5)}},
		{"ij,jk,kl->il", []*tensor.Tensor{mk(2, 3), mk(3, 4), mk(4, 2)}},
	}
	for _, c := range cases {
		c1, err := ops.Einsum(c.notation, c.ops...)
		if err != nil {
			t.Fatalf("CPU %q: %v", c.notation, err)
		}
		g1, err := ops.Einsum(c.notation, toGPU(t, gpu, c.ops...)...)
		if err != nil {
			t.Fatalf("CUDA %q: %v", c.notation, err)
		}
		assertParityF32(t, "Einsum "+c.notation, c1, g1, 1e-4)
	}
}

func TestCUDAConv3dParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	cpu := device.CPU()
	in, _ := tensor.RandN(dtype.F32, []int{2, 3, 5, 6, 7}, cpu)
	w, _ := tensor.RandN(dtype.F32, []int{4, 3, 3, 3, 2}, cpu)
	b, _ := tensor.RandN(dtype.F32, []int{4}, cpu)
	g := toGPU(t, gpu, in, w, b)
	for _, c := range []struct {
		stride, pad [3]int
		bias        bool
	}{
		{[3]int{1, 1, 1}, [3]int{0, 0, 0}, false},
		{[3]int{1, 1, 1}, [3]int{1, 1, 1}, true},
		{[3]int{2, 2, 1}, [3]int{1, 0, 1}, true},
	} {
		var bc, bg *tensor.Tensor
		if c.bias {
			bc, bg = b, g[2]
		}
		c1, err := ops.Conv3d(in, w, bc, c.stride, c.pad)
		if err != nil {
			t.Fatal(err)
		}
		g1, err := ops.Conv3d(g[0], g[1], bg, c.stride, c.pad)
		if err != nil {
			t.Fatal(err)
		}
		assertParityF32(t, "Conv3d", c1, g1, 1e-4)
	}
}

func TestCUDAInstanceNorm2dParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	cpu := device.CPU()
	x, _ := tensor.RandN(dtype.F32, []int{2, 6, 5, 7}, cpu)
	gamma, _ := tensor.RandN(dtype.F32, []int{6}, cpu)
	beta, _ := tensor.RandN(dtype.F32, []int{6}, cpu)
	g := toGPU(t, gpu, x, gamma, beta)
	c1, err := ops.InstanceNorm2d(x, gamma, beta, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	g1, err := ops.InstanceNorm2d(g[0], g[1], g[2], 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	assertParityF32(t, "InstanceNorm2d", c1, g1, 1e-4)
	// nn module on GPU.
	m, err := nn.NewInstanceNorm2d(6, 1e-5, gpu)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Forward(g[0]); err != nil {
		t.Fatal(err)
	}
}

func TestCUDAFlashAttentionParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	cpu := device.CPU()
	const B, NH = 2, 2
	for _, S := range []int{1, 7, 64, 257} {
		for _, D := range []int{32, 64, 128} {
			q, _ := tensor.RandN(dtype.F32, []int{B, NH, S, D}, cpu)
			k, _ := tensor.RandN(dtype.F32, []int{B, NH, S, D}, cpu)
			v, _ := tensor.RandN(dtype.F32, []int{B, NH, S, D}, cpu)
			g := toGPU(t, gpu, q, k, v)
			for _, causal := range []bool{false, true} {
				cpuFA, err := ops.FlashAttention(q, k, v, causal)
				if err != nil {
					t.Fatal(err)
				}
				gpuFA, err := ops.FlashAttention(g[0], g[1], g[2], causal)
				if err != nil {
					t.Fatalf("S=%d D=%d causal=%v: %v", S, D, causal, err)
				}
				assertParityF32(t, "FlashAttention CPU vs CUDA", cpuFA, gpuFA, 1e-4)
				// And against SDPA evaluated on the GPU.
				gpuSDPA := sdpaRef(t, g[0], g[1], g[2], causal)
				if d := maxAbsDiff(t, gpuFA, gpuSDPA); d > 1e-4 {
					t.Errorf("S=%d D=%d causal=%v: FlashAttention(CUDA) vs SDPA(CUDA) max |diff| = %g", S, D, causal, d)
				}
			}
		}
	}
	// Generic (non-specialised D) tiled kernel path.
	q, _ := tensor.RandN(dtype.F32, []int{1, 2, 33, 48}, cpu)
	k, _ := tensor.RandN(dtype.F32, []int{1, 2, 33, 48}, cpu)
	v, _ := tensor.RandN(dtype.F32, []int{1, 2, 33, 48}, cpu)
	g := toGPU(t, gpu, q, k, v)
	c1, _ := ops.FlashAttention(q, k, v, true)
	g1, err := ops.FlashAttention(g[0], g[1], g[2], true)
	if err != nil {
		t.Fatal(err)
	}
	assertParityF32(t, "FlashAttention D=48", c1, g1, 1e-4)
}

func TestCUDAMatMulInt8Parity(t *testing.T) {
	gpu := gpuOrSkip(t)
	cpu := device.CPU()
	const M, K, N = 33, 128, 70
	x, _ := tensor.RandN(dtype.F32, []int{M, K}, cpu)
	w, _ := tensor.RandN(dtype.F32, []int{K, N}, cpu)
	wq, scales, err := quant.QuantizeInt8PerColumn(w)
	if err != nil {
		t.Fatal(err)
	}
	c1, err := ops.MatMulInt8(x, wq, scales)
	if err != nil {
		t.Fatal(err)
	}
	g := toGPU(t, gpu, x, wq, scales)
	g1, err := ops.MatMulInt8(g[0], g[1], g[2])
	if err != nil {
		t.Fatal(err)
	}
	assertParityF32(t, "MatMulInt8", c1, g1, 1e-3)
	ref, _ := ops.MatMul(x, w)
	if e := relErrNorm(t, g1, ref); e >= 0.01 {
		t.Errorf("CUDA rel-err-norm vs F32 matmul = %g, want < 0.01", e)
	}
	// Quantising a GPU-resident weight returns GPU tensors.
	wG := toGPU(t, gpu, w)[0]
	wqG, scG, err := quant.QuantizeInt8PerColumn(wG)
	if err != nil {
		t.Fatal(err)
	}
	if wqG.Device().Kind() != device.KindCUDA || scG.Device().Kind() != device.KindCUDA {
		t.Fatal("quantised weight should stay on CUDA")
	}
	g2, err := ops.MatMulInt8(g[0], wqG, scG)
	if err != nil {
		t.Fatal(err)
	}
	assertParityF32(t, "MatMulInt8 (GPU-quantised)", c1, g2, 1e-3)
}

// TestCUDAFlashAttentionSpeed reports FlashAttention vs SDPA wall time at
// S=1024, D=64 on the GPU (logged, not asserted — hardware dependent).
func TestCUDAFlashAttentionSpeed(t *testing.T) {
	if testing.Short() {
		t.Skip("timing test")
	}
	gpu := gpuOrSkip(t)
	const B, NH, S, D = 1, 8, 1024, 64
	q, _ := tensor.RandN(dtype.F32, []int{B, NH, S, D}, gpu)
	k, _ := tensor.RandN(dtype.F32, []int{B, NH, S, D}, gpu)
	v, _ := tensor.RandN(dtype.F32, []int{B, NH, S, D}, gpu)
	mask, _ := nn.CausalMask(S, gpu)
	for _, causal := range []bool{false, true} {
		var m *tensor.Tensor
		if causal {
			m = mask
		}
		timeIt := func(fn func() error) time.Duration {
			for range 3 { // warm-up (PTX JIT, cuBLAS handle)
				if err := fn(); err != nil {
					t.Fatal(err)
				}
			}
			const iters = 10
			start := time.Now()
			for range iters {
				if err := fn(); err != nil {
					t.Fatal(err)
				}
			}
			return time.Since(start) / iters
		}
		fa := timeIt(func() error { _, err := ops.FlashAttention(q, k, v, causal); return err })
		sd := timeIt(func() error { _, err := nn.SDPA(q, k, v, m); return err })
		t.Logf("S=%d D=%d NH=%d causal=%v: FlashAttention %v, SDPA %v (ratio %.2fx)", S, D, NH, causal, fa, sd, float64(sd)/float64(fa))
	}
}
