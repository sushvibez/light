//go:build !nowebgpu

package webgpu_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// gpuOrSkip returns the WebGPU device or skips the test.
func gpuOrSkip(t *testing.T) device.Device {
	t.Helper()
	gpu, err := device.WebGPU(0)
	if err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	return gpu
}

func randF32(n int, seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	out := make([]float32, n)
	for i := range out {
		out[i] = r.Float32()*4 - 2
	}
	return out
}

func mustNew(t *testing.T, data []float32, dims []int, dev device.Device) *tensor.Tensor {
	t.Helper()
	x, err := tensor.New(data, dims, dev)
	if err != nil {
		t.Fatal(err)
	}
	return x
}

func toCPUF32(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	c, err := x.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if !c.IsContiguous() || c.Offset() != 0 {
		c, err = c.Contiguous()
		if err != nil {
			t.Fatal(err)
		}
	}
	return tensor.MustSlice[float32](c)
}

func assertClose(t *testing.T, name string, want, got []float32, tol float64) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: length mismatch CPU=%d GPU=%d", name, len(want), len(got))
	}
	bad := 0
	for i := range want {
		w, g := float64(want[i]), float64(got[i])
		if math.IsNaN(w) && math.IsNaN(g) {
			continue
		}
		if math.IsInf(w, 0) || math.IsInf(g, 0) {
			if w != g {
				bad++
				if bad <= 5 {
					t.Errorf("%s[%d]: CPU=%v GPU=%v", name, i, w, g)
				}
			}
			continue
		}
		diff := math.Abs(w - g)
		if diff > tol+tol*math.Abs(w) {
			bad++
			if bad <= 5 {
				t.Errorf("%s[%d]: CPU=%v GPU=%v (diff %g)", name, i, w, g, diff)
			}
		}
	}
	if bad > 5 {
		t.Errorf("%s: %d mismatches total", name, bad)
	}
}

// parity2 runs a 2-input op on CPU and WebGPU and compares.
func parity2(t *testing.T, name string, a, b *tensor.Tensor, fn func(a, b *tensor.Tensor) (*tensor.Tensor, error), tol float64) {
	t.Helper()
	gpu := gpuOrSkip(t)
	cpuOut, err := fn(a, b)
	if err != nil {
		t.Fatalf("%s CPU: %v", name, err)
	}
	aG, err := a.To(gpu)
	if err != nil {
		t.Fatal(err)
	}
	bG, err := b.To(gpu)
	if err != nil {
		t.Fatal(err)
	}
	gpuOut, err := fn(aG, bG)
	if err != nil {
		t.Fatalf("%s WebGPU: %v", name, err)
	}
	if gpuOut.Device().Kind() != device.KindWebGPU {
		t.Fatalf("%s: output on %v, want WebGPU", name, gpuOut.Device())
	}
	if !gpuOut.Shape().Equal(cpuOut.Shape()) {
		t.Fatalf("%s: shape CPU=%v GPU=%v", name, cpuOut.Shape(), gpuOut.Shape())
	}
	assertClose(t, name, toCPUF32(t, cpuOut), toCPUF32(t, gpuOut), tol)
}

func parity1(t *testing.T, name string, a *tensor.Tensor, fn func(a *tensor.Tensor) (*tensor.Tensor, error), tol float64) {
	t.Helper()
	parity2(t, name, a, a, func(x, _ *tensor.Tensor) (*tensor.Tensor, error) { return fn(x) }, tol)
}

func TestBinaryParity(t *testing.T) {
	cpu := device.CPU()
	n := 1000
	a := mustNew(t, randF32(n, 1), []int{10, 100}, cpu)
	b := mustNew(t, randF32(n, 2), []int{10, 100}, cpu)
	pos := make([]float32, n)
	for i, v := range randF32(n, 3) {
		pos[i] = float32(math.Abs(float64(v))) + 0.1
	}
	ap := mustNew(t, pos, []int{10, 100}, cpu)
	bp := mustNew(t, randF32(n, 4), []int{10, 100}, cpu)
	for name, fn := range map[string]func(a, b *tensor.Tensor) (*tensor.Tensor, error){
		"add": ops.Add, "sub": ops.Sub, "mul": ops.Mul, "div": ops.Div,
	} {
		t.Run(name, func(t *testing.T) { parity2(t, name, a, b, fn, 1e-5) })
	}
	t.Run("pow", func(t *testing.T) { parity2(t, "pow", ap, bp, ops.Pow, 1e-4) })
}

func TestBinaryBroadcastParity(t *testing.T) {
	cpu := device.CPU()
	a := mustNew(t, randF32(2*3*4, 5), []int{2, 3, 4}, cpu)
	row := mustNew(t, randF32(4, 6), []int{4}, cpu)
	col := mustNew(t, randF32(3, 7), []int{3, 1}, cpu)
	scalar := mustNew(t, []float32{0.125}, []int{1}, cpu)
	mask := mustNew(t, randF32(3*4, 8), []int{3, 4}, cpu)
	t.Run("bias_row", func(t *testing.T) { parity2(t, "add(b,t)", a, row, ops.Add, 1e-5) })
	t.Run("col", func(t *testing.T) { parity2(t, "mul(col)", a, col, ops.Mul, 1e-5) })
	t.Run("scalar", func(t *testing.T) { parity2(t, "mul(scalar)", a, scalar, ops.Mul, 1e-5) })
	t.Run("mask", func(t *testing.T) { parity2(t, "add(mask)", a, mask, ops.Add, 1e-5) })
	t.Run("scalar_lhs", func(t *testing.T) { parity2(t, "sub(scalar,t)", scalar, a, ops.Sub, 1e-5) })
	// Strided (transposed view) operand.
	t.Run("transposed", func(t *testing.T) {
		gpu := gpuOrSkip(t)
		m := mustNew(t, randF32(6*7, 9), []int{6, 7}, cpu)
		mT, _ := m.Transpose(0, 1)
		o := mustNew(t, randF32(7*6, 10), []int{7, 6}, cpu)
		cpuOut, err := ops.Add(mT, o)
		if err != nil {
			t.Fatal(err)
		}
		mG, _ := m.To(gpu)
		mTG, _ := mG.Transpose(0, 1)
		oG, _ := o.To(gpu)
		gpuOut, err := ops.Add(mTG, oG)
		if err != nil {
			t.Fatal(err)
		}
		assertClose(t, "add(T)", toCPUF32(t, cpuOut), toCPUF32(t, gpuOut), 1e-5)
	})
	// Offset (narrowed) operand.
	t.Run("narrowed", func(t *testing.T) {
		gpu := gpuOrSkip(t)
		m := mustNew(t, randF32(8*5, 11), []int{8, 5}, cpu)
		mN, _ := m.Narrow(0, 3, 4)
		o := mustNew(t, randF32(4*5, 12), []int{4, 5}, cpu)
		cpuOut, _ := ops.Mul(mN, o)
		mG, _ := m.To(gpu)
		mNG, _ := mG.Narrow(0, 3, 4)
		oG, _ := o.To(gpu)
		gpuOut, err := ops.Mul(mNG, oG)
		if err != nil {
			t.Fatal(err)
		}
		assertClose(t, "mul(narrow)", toCPUF32(t, cpuOut), toCPUF32(t, gpuOut), 1e-5)
	})
}

func TestUnaryParity(t *testing.T) {
	cpu := device.CPU()
	n := 777
	a := mustNew(t, randF32(n, 20), []int{7, 111}, cpu)
	pos := make([]float32, n)
	for i, v := range randF32(n, 21) {
		pos[i] = float32(math.Abs(float64(v))) + 0.05
	}
	ap := mustNew(t, pos, []int{7, 111}, cpu)
	for name, fn := range map[string]func(*tensor.Tensor) (*tensor.Tensor, error){
		"neg": ops.Neg, "exp": ops.Exp, "relu": ops.Relu, "sigmoid": ops.Sigmoid,
		"tanh": ops.Tanh, "silu": ops.SiLU, "gelu": ops.Gelu, "abs": ops.Abs,
		"sin": ops.Sin, "cos": ops.Cos, "floor": ops.Floor, "ceil": ops.Ceil, "round": ops.Round,
	} {
		t.Run(name, func(t *testing.T) { parity1(t, name, a, fn, 1e-4) })
	}
	t.Run("log", func(t *testing.T) { parity1(t, "log", ap, ops.Log, 1e-4) })
	t.Run("sqrt", func(t *testing.T) { parity1(t, "sqrt", ap, ops.Sqrt, 1e-5) })
	// tan is ill-conditioned near poles; keep inputs small.
	small := make([]float32, n)
	for i, v := range randF32(n, 22) {
		small[i] = v / 4
	}
	as := mustNew(t, small, []int{n}, cpu)
	t.Run("tan", func(t *testing.T) { parity1(t, "tan", as, ops.Tan, 1e-3) })
}

func TestMatMulParity(t *testing.T) {
	cpu := device.CPU()
	for _, sz := range [][3]int{{4, 5, 6}, {33, 17, 65}, {128, 64, 256}, {1, 768, 50}} {
		M, K, N := sz[0], sz[1], sz[2]
		a := mustNew(t, randF32(M*K, 30), []int{M, K}, cpu)
		b := mustNew(t, randF32(K*N, 31), []int{K, N}, cpu)
		t.Run("plain", func(t *testing.T) { parity2(t, "matmul", a, b, ops.MatMul, 1e-3) })
		// nn.Linear layout: W is (N, K), used as W^T view.
		w := mustNew(t, randF32(N*K, 32), []int{N, K}, cpu)
		t.Run("transB", func(t *testing.T) {
			gpu := gpuOrSkip(t)
			wT, _ := w.Transpose(0, 1)
			cpuOut, err := ops.MatMul(a, wT)
			if err != nil {
				t.Fatal(err)
			}
			aG, _ := a.To(gpu)
			wG, _ := w.To(gpu)
			wTG, _ := wG.Transpose(0, 1)
			gpuOut, err := ops.MatMul(aG, wTG)
			if err != nil {
				t.Fatal(err)
			}
			assertClose(t, "matmul(transB)", toCPUF32(t, cpuOut), toCPUF32(t, gpuOut), 1e-3)
		})
	}
}

func TestBatchedMatMulParity(t *testing.T) {
	cpu := device.CPU()
	B, M, K, N := 6, 9, 20, 11
	a := mustNew(t, randF32(B*M*K, 40), []int{B, M, K}, cpu)
	b := mustNew(t, randF32(B*K*N, 41), []int{B, K, N}, cpu)
	parity2(t, "matmul_batched", a, b, ops.MatMul, 1e-3)
}

func TestSoftmaxParity(t *testing.T) {
	cpu := device.CPU()
	a := mustNew(t, randF32(3*5*300, 50), []int{3, 5, 300}, cpu)
	for _, axis := range []int{-1, 1, 0} {
		t.Run("softmax", func(t *testing.T) {
			parity1(t, "softmax", a, func(x *tensor.Tensor) (*tensor.Tensor, error) { return ops.Softmax(x, axis) }, 1e-5)
		})
	}
	t.Run("logsoftmax", func(t *testing.T) {
		parity1(t, "logsoftmax", a, func(x *tensor.Tensor) (*tensor.Tensor, error) { return ops.LogSoftmax(x, -1) }, 1e-4)
	})
}

func TestNormParity(t *testing.T) {
	cpu := device.CPU()
	rows, cols := 37, 768
	x := mustNew(t, randF32(rows*cols, 60), []int{rows, cols}, cpu)
	g := mustNew(t, randF32(cols, 61), []int{cols}, cpu)
	b := mustNew(t, randF32(cols, 62), []int{cols}, cpu)
	t.Run("layernorm", func(t *testing.T) {
		gpu := gpuOrSkip(t)
		cpuOut, err := ops.LayerNorm(x, g, b, 1e-5)
		if err != nil {
			t.Fatal(err)
		}
		xG, _ := x.To(gpu)
		gG, _ := g.To(gpu)
		bG, _ := b.To(gpu)
		gpuOut, err := ops.LayerNorm(xG, gG, bG, 1e-5)
		if err != nil {
			t.Fatal(err)
		}
		assertClose(t, "layernorm", toCPUF32(t, cpuOut), toCPUF32(t, gpuOut), 1e-4)
	})
	t.Run("rmsnorm", func(t *testing.T) {
		parity2(t, "rmsnorm", x, g, func(a, w *tensor.Tensor) (*tensor.Tensor, error) { return ops.RMSNorm(a, w, 1e-6) }, 1e-4)
	})
}

func TestReduceParity(t *testing.T) {
	cpu := device.CPU()
	a := mustNew(t, randF32(4*6*300, 70), []int{4, 6, 300}, cpu)
	fns := map[string]func(*tensor.Tensor, ops.ReduceOptions) (*tensor.Tensor, error){
		"sum": ops.Sum, "mean": ops.Mean, "max": ops.Max, "min": ops.Min,
	}
	cases := []ops.ReduceOptions{
		{},                  // all
		{Axes: []int{-1}},   // last
		{Axes: []int{0}},    // first (permute path)
		{Axes: []int{1}},    // middle
		{Axes: []int{0, 2}}, // mixed
		{Axes: []int{1}, KeepDims: true},
		{Axes: []int{2}, KeepDims: true},
	}
	for name, fn := range fns {
		for _, opts := range cases {
			t.Run(name, func(t *testing.T) {
				parity1(t, name, a, func(x *tensor.Tensor) (*tensor.Tensor, error) { return fn(x, opts) }, 1e-4)
			})
		}
	}
	// prod on values near 1 so the product stays finite.
	near := make([]float32, 5*7)
	for i, v := range randF32(len(near), 71) {
		near[i] = 1 + v/10
	}
	p := mustNew(t, near, []int{5, 7}, cpu)
	t.Run("prod", func(t *testing.T) {
		parity1(t, "prod", p, func(x *tensor.Tensor) (*tensor.Tensor, error) { return ops.Prod(x, ops.ReduceOptions{Axes: []int{1}}) }, 1e-4)
	})
}

func TestIndexSelectParity(t *testing.T) {
	cpu := device.CPU()
	gpu := gpuOrSkip(t)
	table := mustNew(t, randF32(50*16, 80), []int{50, 16}, cpu)
	idx, _ := tensor.New([]int64{3, 0, 49, 7, 7, 12}, []int{6}, cpu)
	cpuOut, err := ops.IndexSelect(table, idx)
	if err != nil {
		t.Fatal(err)
	}
	tG, _ := table.To(gpu)
	iG, _ := idx.To(gpu)
	gpuOut, err := ops.IndexSelect(tG, iG)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "index_select", toCPUF32(t, cpuOut), toCPUF32(t, gpuOut), 0)

	// I64 table (8-byte rows).
	src64, _ := tensor.New([]int64{10, 11, 20, 21, 30, 31, 40, 41}, []int{4, 2}, cpu)
	cpu64, _ := ops.IndexSelect(src64, idx64(t, cpu, 2, 0, 3))
	s64G, _ := src64.To(gpu)
	gpu64, err := ops.IndexSelect(s64G, idx64(t, gpu, 2, 0, 3))
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpu64.To(cpu)
	w := tensor.MustSlice[int64](cpu64)
	g := tensor.MustSlice[int64](back)
	for i := range w {
		if w[i] != g[i] {
			t.Fatalf("index_select i64: CPU=%v GPU=%v", w, g)
		}
	}
}

func idx64(t *testing.T, dev device.Device, vals ...int64) *tensor.Tensor {
	t.Helper()
	x, err := tensor.New(vals, []int{len(vals)}, dev)
	if err != nil {
		t.Fatal(err)
	}
	return x
}

func TestConcatParity(t *testing.T) {
	cpu := device.CPU()
	gpu := gpuOrSkip(t)
	a := mustNew(t, randF32(2*3*4, 90), []int{2, 3, 4}, cpu)
	b := mustNew(t, randF32(2*5*4, 91), []int{2, 5, 4}, cpu)
	c := mustNew(t, randF32(2*3*7, 92), []int{2, 3, 7}, cpu)
	aG, _ := a.To(gpu)
	bG, _ := b.To(gpu)
	cG, _ := c.To(gpu)
	for _, tc := range []struct {
		name string
		cpu  []*tensor.Tensor
		gpu  []*tensor.Tensor
		axis int
	}{
		{"axis1", []*tensor.Tensor{a, b}, []*tensor.Tensor{aG, bG}, 1},
		{"axis2", []*tensor.Tensor{a, c}, []*tensor.Tensor{aG, cG}, 2},
		{"axis0", []*tensor.Tensor{a, a}, []*tensor.Tensor{aG, aG}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cpuOut, err := ops.Concat(tc.cpu, tc.axis)
			if err != nil {
				t.Fatal(err)
			}
			gpuOut, err := ops.Concat(tc.gpu, tc.axis)
			if err != nil {
				t.Fatal(err)
			}
			assertClose(t, "concat", toCPUF32(t, cpuOut), toCPUF32(t, gpuOut), 0)
		})
	}
}

func TestContiguousParity(t *testing.T) {
	cpu := device.CPU()
	gpu := gpuOrSkip(t)
	// 4-D permute (the MHA head split), transpose, narrow+offset, I64 (8-byte).
	x := mustNew(t, randF32(2*5*3*4, 100), []int{2, 5, 3, 4}, cpu)
	xG, _ := x.To(gpu)
	t.Run("permute4d", func(t *testing.T) {
		pC, _ := x.Permute(0, 2, 1, 3)
		cpuOut, _ := pC.Contiguous()
		pG, _ := xG.Permute(0, 2, 1, 3)
		gpuOut, err := pG.Contiguous()
		if err != nil {
			t.Fatal(err)
		}
		if gpuOut.Device().Kind() != device.KindWebGPU || !gpuOut.IsContiguous() {
			t.Fatal("Contiguous result must be a contiguous WebGPU tensor")
		}
		assertClose(t, "permute", toCPUF32(t, cpuOut), toCPUF32(t, gpuOut), 0)
	})
	t.Run("transpose_to_cpu", func(t *testing.T) {
		m := mustNew(t, randF32(7*9, 101), []int{7, 9}, cpu)
		mT, _ := m.Transpose(0, 1)
		cpuOut, _ := mT.Contiguous()
		mG, _ := m.To(gpu)
		mTG, _ := mG.Transpose(0, 1)
		// To(CPU) must contiguify on device first.
		assertClose(t, "transpose", tensor.MustSlice[float32](cpuOut), toCPUF32(t, mTG), 0)
	})
	t.Run("narrow_offset", func(t *testing.T) {
		n, _ := x.Narrow(1, 2, 2)
		cpuOut, _ := n.Contiguous()
		nG, _ := xG.Narrow(1, 2, 2)
		gpuOut, err := nG.Contiguous()
		if err != nil {
			t.Fatal(err)
		}
		assertClose(t, "narrow", toCPUF32(t, cpuOut), toCPUF32(t, gpuOut), 0)
	})
	t.Run("i64_transpose", func(t *testing.T) {
		v := make([]int64, 6*5)
		for i := range v {
			v[i] = int64(i)*1000000007 - 5
		}
		m, _ := tensor.New(v, []int{6, 5}, cpu)
		mT, _ := m.Transpose(0, 1)
		cpuOut, _ := mT.Contiguous()
		mG, _ := m.To(gpu)
		mTG, _ := mG.Transpose(0, 1)
		gpuOut, err := mTG.Contiguous()
		if err != nil {
			t.Fatal(err)
		}
		back, _ := gpuOut.To(cpu)
		w := tensor.MustSlice[int64](cpuOut)
		g := tensor.MustSlice[int64](back)
		for i := range w {
			if w[i] != g[i] {
				t.Fatalf("i64 contiguous mismatch at %d: %d vs %d", i, w[i], g[i])
			}
		}
	})
}

func TestConv2dParity(t *testing.T) {
	cpu := device.CPU()
	gpu := gpuOrSkip(t)
	const N, Cin, H, W, Cout, kH, kW = 2, 3, 9, 7, 4, 3, 3
	in := mustNew(t, randF32(N*Cin*H*W, 110), []int{N, Cin, H, W}, cpu)
	w := mustNew(t, randF32(Cout*Cin*kH*kW, 111), []int{Cout, Cin, kH, kW}, cpu)
	bias := mustNew(t, randF32(Cout, 112), []int{Cout}, cpu)
	inG, _ := in.To(gpu)
	wG, _ := w.To(gpu)
	bG, _ := bias.To(gpu)
	for _, tc := range []struct {
		name    string
		stride  [2]int
		padding [2]int
		bias    bool
	}{
		{"s1p1_bias", [2]int{1, 1}, [2]int{1, 1}, true},
		{"s2p0", [2]int{2, 2}, [2]int{0, 0}, false},
		{"s2p1_bias", [2]int{2, 1}, [2]int{1, 2}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var bc, bg *tensor.Tensor
			if tc.bias {
				bc, bg = bias, bG
			}
			cpuOut, err := ops.Conv2d(in, w, bc, tc.stride, tc.padding)
			if err != nil {
				t.Fatal(err)
			}
			gpuOut, err := ops.Conv2d(inG, wG, bg, tc.stride, tc.padding)
			if err != nil {
				t.Fatal(err)
			}
			assertClose(t, "conv2d", toCPUF32(t, cpuOut), toCPUF32(t, gpuOut), 1e-4)
		})
	}
}

func TestPoolParity(t *testing.T) {
	cpu := device.CPU()
	gpu := gpuOrSkip(t)
	in := mustNew(t, randF32(2*3*9*8, 120), []int{2, 3, 9, 8}, cpu)
	inG, _ := in.To(gpu)
	for name, fn := range map[string]func(*tensor.Tensor, [2]int, [2]int, [2]int) (*tensor.Tensor, error){
		"max": ops.MaxPool2d, "avg": ops.AvgPool2d,
	} {
		for _, tc := range []struct {
			k, s, p [2]int
		}{
			{[2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0}},
			{[2]int{3, 3}, [2]int{2, 2}, [2]int{1, 1}},
			{[2]int{3, 2}, [2]int{1, 2}, [2]int{1, 0}},
		} {
			t.Run(name, func(t *testing.T) {
				cpuOut, err := fn(in, tc.k, tc.s, tc.p)
				if err != nil {
					t.Fatal(err)
				}
				gpuOut, err := fn(inG, tc.k, tc.s, tc.p)
				if err != nil {
					t.Fatal(err)
				}
				assertClose(t, name+"pool", toCPUF32(t, cpuOut), toCPUF32(t, gpuOut), 1e-5)
			})
		}
	}
	t.Run("adaptive", func(t *testing.T) {
		for _, o := range [][2]int{{1, 1}, {3, 3}, {4, 5}, {9, 8}} {
			cpuOut, err := ops.AdaptiveAvgPool2d(in, o[0], o[1])
			if err != nil {
				t.Fatal(err)
			}
			gpuOut, err := ops.AdaptiveAvgPool2d(inG, o[0], o[1])
			if err != nil {
				t.Fatal(err)
			}
			if gpuOut.Device().Kind() != device.KindWebGPU {
				t.Fatal("adaptive pool output should stay on WebGPU")
			}
			assertClose(t, "adaptive", toCPUF32(t, cpuOut), toCPUF32(t, gpuOut), 1e-5)
		}
	})
}

func TestBatchNormEvalParity(t *testing.T) {
	cpu := device.CPU()
	gpu := gpuOrSkip(t)
	const N, C, H, W = 2, 5, 4, 3
	in := mustNew(t, randF32(N*C*H*W, 130), []int{N, C, H, W}, cpu)
	g := mustNew(t, randF32(C, 131), []int{C}, cpu)
	b := mustNew(t, randF32(C, 132), []int{C}, cpu)
	rm := mustNew(t, randF32(C, 133), []int{C}, cpu)
	rvRaw := randF32(C, 134)
	for i := range rvRaw {
		rvRaw[i] = float32(math.Abs(float64(rvRaw[i]))) + 0.2
	}
	rv := mustNew(t, rvRaw, []int{C}, cpu)
	opts := ops.BatchNormOptions{Train: false, Eps: 1e-5}
	cpuRes, err := ops.BatchNorm2d(in, g, b, rm, rv, opts)
	if err != nil {
		t.Fatal(err)
	}
	inG, _ := in.To(gpu)
	gG, _ := g.To(gpu)
	bG, _ := b.To(gpu)
	rmG, _ := rm.To(gpu)
	rvG, _ := rv.To(gpu)
	gpuRes, err := ops.BatchNorm2d(inG, gG, bG, rmG, rvG, opts)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "bn.out", toCPUF32(t, cpuRes.Out), toCPUF32(t, gpuRes.Out), 1e-4)
	assertClose(t, "bn.xhat", toCPUF32(t, cpuRes.XHat), toCPUF32(t, gpuRes.XHat), 1e-4)
	assertClose(t, "bn.sigma", toCPUF32(t, cpuRes.Sigma), toCPUF32(t, gpuRes.Sigma), 1e-5)
	assertClose(t, "bn.mean", toCPUF32(t, cpuRes.Mean), toCPUF32(t, gpuRes.Mean), 0)
}

func TestCastParity(t *testing.T) {
	cpu := device.CPU()
	gpu := gpuOrSkip(t)
	vals := []float32{-3.7, -1, -0.5, 0, 0.5, 1, 2.9, 100000.5, -123456}
	f := mustNew(t, vals, []int{len(vals)}, cpu)
	fG, _ := f.To(gpu)
	for _, to := range []dtype.DType{dtype.I32, dtype.I64} {
		cpuOut, err := ops.Cast(f, to)
		if err != nil {
			t.Fatal(err)
		}
		gpuOut, err := ops.Cast(fG, to)
		if err != nil {
			t.Fatalf("cast F32→%v: %v", to, err)
		}
		back, _ := gpuOut.To(cpu)
		// Round-trip back to F32 on both sides so one comparison path serves
		// both integer widths.
		cpuF, _ := ops.Cast(cpuOut, dtype.F32)
		gpuF, err := ops.Cast(gpuOut, dtype.F32)
		if err != nil {
			t.Fatalf("cast %v→F32: %v", to, err)
		}
		assertClose(t, "cast "+to.String(), toCPUF32(t, cpuF), toCPUF32(t, gpuF), 0)
		if to == dtype.I64 {
			w := tensor.MustSlice[int64](cpuOut)
			g := tensor.MustSlice[int64](back)
			for i := range w {
				if w[i] != g[i] {
					t.Fatalf("cast I64 raw mismatch at %d: %d vs %d", i, w[i], g[i])
				}
			}
		}
	}
	// Same-dtype cast is a copy.
	same, err := ops.Cast(fG, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "cast same", vals, toCPUF32(t, same), 0)
}

func TestTransferAndCopyFrom(t *testing.T) {
	cpu := device.CPU()
	gpu := gpuOrSkip(t)
	data := randF32(1001, 140) // odd length: exercises 4-byte padding paths
	x := mustNew(t, data, []int{1001}, cpu)
	xG, err := x.To(gpu)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "roundtrip", data, toCPUF32(t, xG), 0)

	// Zeros really are zero even when the pool hands back a used buffer.
	z, err := tensor.Zeros(dtype.F32, []int{1001}, gpu)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range toCPUF32(t, z) {
		if v != 0 {
			t.Fatalf("Zeros[%d] = %v", i, v)
		}
	}
	// CopyFrom host → device in place, and device → device.
	dst, _ := tensor.Zeros(dtype.F32, []int{1001}, gpu)
	if err := dst.CopyFrom(x); err != nil {
		t.Fatal(err)
	}
	assertClose(t, "copyfrom host", data, toCPUF32(t, dst), 0)
	dst2, _ := tensor.Zeros(dtype.F32, []int{1001}, gpu)
	if err := dst2.CopyFrom(dst); err != nil {
		t.Fatal(err)
	}
	assertClose(t, "copyfrom device", data, toCPUF32(t, dst2), 0)

	// U8 round trip (no kernels, but storage/transfer must work byte-exact).
	u8, _ := tensor.New([]uint8{1, 2, 3, 4, 5}, []int{5}, cpu)
	u8G, err := u8.To(gpu)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := u8G.To(cpu)
	got := tensor.MustSlice[uint8](back)
	for i, v := range []uint8{1, 2, 3, 4, 5} {
		if got[i] != v {
			t.Fatalf("u8 roundtrip: %v", got)
		}
	}
}

func TestLargeDispatch(t *testing.T) {
	// > 65535*256 elements forces the 2-D workgroup grid.
	cpu := device.CPU()
	gpu := gpuOrSkip(t)
	n := 65535*256 + 12345
	a := mustNew(t, randF32(n, 150), []int{n}, cpu)
	aG, _ := a.To(gpu)
	gpuOut, err := ops.Relu(aG)
	if err != nil {
		t.Fatal(err)
	}
	cpuOut, _ := ops.Relu(a)
	assertClose(t, "relu large", toCPUF32(t, cpuOut), toCPUF32(t, gpuOut), 0)
}

func TestMissingKernelErrors(t *testing.T) {
	gpu := gpuOrSkip(t)
	x, _ := tensor.Ones(dtype.F64, []int{4}, gpu)
	if _, err := ops.Exp(x); err == nil {
		t.Fatal("F64 exp on WebGPU should fail with kernel-not-found, not silently fall back")
	}
}
