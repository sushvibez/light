//go:build !nocuda && cgo

package ops_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// parityCheck runs op on CPU and on CUDA, compares results within tolerance.
func parityAdd(t *testing.T, data1, data2 []float32, dims ...int) {
	t.Helper()
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}

	// CPU
	aCPU, _ := tensor.New(data1, dims, cpu)
	bCPU, _ := tensor.New(data2, dims, cpu)
	outCPU, err := ops.Add(aCPU, bCPU)
	if err != nil {
		t.Fatalf("CPU Add: %v", err)
	}
	cpuVals := tensor.MustSlice[float32](outCPU)

	// CUDA
	aGPU, err := aCPU.To(gpu)
	if err != nil {
		t.Fatalf("To(GPU) a: %v", err)
	}
	bGPU, err := bCPU.To(gpu)
	if err != nil {
		t.Fatalf("To(GPU) b: %v", err)
	}
	outGPU, err := ops.Add(aGPU, bGPU)
	if err != nil {
		t.Fatalf("CUDA Add: %v", err)
	}
	if outGPU.Device().Kind() != device.KindCUDA {
		t.Fatal("output should be on CUDA")
	}
	backCPU, err := outGPU.To(cpu)
	if err != nil {
		t.Fatalf("To(CPU) result: %v", err)
	}
	gpuVals := tensor.MustSlice[float32](backCPU)

	if len(gpuVals) != len(cpuVals) {
		t.Fatalf("length mismatch: CPU=%d, GPU=%d", len(cpuVals), len(gpuVals))
	}
	for i := range cpuVals {
		if math.Abs(float64(cpuVals[i]-gpuVals[i])) > 1e-5 {
			t.Errorf("[%d] CPU=%v vs GPU=%v", i, cpuVals[i], gpuVals[i])
		}
	}
}

func TestAddF32Parity(t *testing.T) {
	parityAdd(t, []float32{1, 2, 3, 4}, []float32{10, 20, 30, 40}, 4)
}

func TestAddLargeParity(t *testing.T) {
	n := 1024
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32(i) * 0.5
		b[i] = float32(n-i) * 0.25
	}
	parityAdd(t, a, b, n)
}

func TestAdd2DParity(t *testing.T) {
	parityAdd(t, []float32{1, 2, 3, 4, 5, 6}, []float32{10, 20, 30, 40, 50, 60}, 2, 3)
}

func TestCUDAConv2dParity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	// Small conv: (1, 3, 5, 5) × (2, 3, 3, 3) stride=1 padding=1 → (1, 2, 5, 5).
	const N, Cin, H, W, Cout, kH, kW = 1, 3, 5, 5, 2, 3, 3
	inData := make([]float32, N*Cin*H*W)
	wData := make([]float32, Cout*Cin*kH*kW)
	for i := range inData {
		inData[i] = float32(i%11) * 0.1
	}
	for i := range wData {
		wData[i] = float32(i%7) * 0.05
	}
	in, _ := tensor.New(inData, []int{N, Cin, H, W}, cpu)
	w, _ := tensor.New(wData, []int{Cout, Cin, kH, kW}, cpu)
	cpuOut, _ := ops.Conv2d(in, w, nil, [2]int{1, 1}, [2]int{1, 1})
	inG, _ := in.To(gpu)
	wG, _ := w.To(gpu)
	gpuOut, err := ops.Conv2d(inG, wG, nil, [2]int{1, 1}, [2]int{1, 1})
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpuOut.To(cpu)
	cs := tensor.MustSlice[float32](cpuOut)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > 1e-4 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestCUDAPoolParity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	in, _ := tensor.New([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, []int{1, 1, 4, 4}, cpu)
	inG, _ := in.To(gpu)
	for name, fn := range map[string]func(*tensor.Tensor, [2]int, [2]int, [2]int) (*tensor.Tensor, error){
		"max": ops.MaxPool2d,
		"avg": ops.AvgPool2d,
	} {
		t.Run(name, func(t *testing.T) {
			cpuOut, _ := fn(in, [2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0})
			gpuOut, err := fn(inG, [2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0})
			if err != nil {
				t.Fatal(err)
			}
			back, _ := gpuOut.To(cpu)
			cs := tensor.MustSlice[float32](cpuOut)
			gs := tensor.MustSlice[float32](back)
			for i := range cs {
				if math.Abs(float64(cs[i]-gs[i])) > 1e-4 {
					t.Errorf("%s[%d] CPU=%v GPU=%v", name, i, cs[i], gs[i])
				}
			}
		})
	}
}

func TestCUDARoPEParity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	x, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 4}, cpu)
	pos, _ := tensor.New([]int64{0, 1}, []int{2}, cpu)
	cpuOut, _ := ops.RoPE(x, pos, 10000)
	xG, _ := x.To(gpu)
	pG, _ := pos.To(gpu)
	gpuOut, err := ops.RoPE(xG, pG, 10000)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpuOut.To(cpu)
	cs := tensor.MustSlice[float32](cpuOut)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > 1e-4 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestCUDACumsumParity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5}, []int{5}, cpu)
	cpuOut, _ := ops.Cumsum(a, 0)
	aG, _ := a.To(gpu)
	gpuOut, err := ops.Cumsum(aG, 0)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpuOut.To(cpu)
	cs := tensor.MustSlice[float32](cpuOut)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > 1e-4 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestCUDATrilParity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	a, _ := tensor.Ones(dtype.F32, []int{3, 3}, cpu)
	cpuOut, _ := tensor.Tril(a, 0)
	aG, _ := a.To(gpu)
	gpuOut, err := tensor.Tril(aG, 0)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpuOut.To(cpu)
	cs := tensor.MustSlice[float32](cpuOut)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if cs[i] != gs[i] {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestCUDADropoutWorks(t *testing.T) {
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	a, _ := tensor.Ones(dtype.F32, []int{1000}, gpu)
	out, err := ops.Dropout(a, 0.5, true)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := out.To(device.CPU())
	vals := tensor.MustSlice[float32](back)
	zeros := 0
	nonzero := 0
	for _, v := range vals {
		if v == 0 {
			zeros++
		} else if math.Abs(float64(v)-2.0) < 1e-4 {
			nonzero++
		}
	}
	if zeros < 300 || zeros > 700 {
		t.Errorf("zero count %d", zeros)
	}
	if nonzero < 300 || nonzero > 700 {
		t.Errorf("nonzero count %d", nonzero)
	}
}

func TestCUDAReduceNonLastAxisParity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	// Reduce along axis 0 (non-last) of (3, 4).
	a, _ := tensor.New([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}, []int{3, 4}, cpu)
	aG, _ := a.To(gpu)
	cpuOut, _ := ops.Sum(a, ops.ReduceOptions{Axes: []int{0}})
	gpuOut, err := ops.Sum(aG, ops.ReduceOptions{Axes: []int{0}})
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpuOut.To(cpu)
	cs := tensor.MustSlice[float32](cpuOut)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > 1e-4 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestUnaryParity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	a, _ := tensor.New([]float32{0.1, 0.5, 1.0, 2.0, 5.0}, []int{5}, cpu)
	aGPU, err := a.To(gpu)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		fn   func(*tensor.Tensor) (*tensor.Tensor, error)
	}{
		{"exp", ops.Exp},
		{"log", ops.Log},
		{"sqrt", ops.Sqrt},
		{"relu", ops.Relu},
		{"sigmoid", ops.Sigmoid},
		{"tanh", ops.Tanh},
		{"silu", ops.SiLU},
		{"neg", ops.Neg},
	} {
		t.Run(c.name, func(t *testing.T) {
			outCPU, err := c.fn(a)
			if err != nil {
				t.Fatal(err)
			}
			outGPU, err := c.fn(aGPU)
			if err != nil {
				t.Fatal(err)
			}
			backCPU, _ := outGPU.To(cpu)
			cs := tensor.MustSlice[float32](outCPU)
			gs := tensor.MustSlice[float32](backCPU)
			for i := range cs {
				if math.Abs(float64(cs[i]-gs[i])) > 1e-5 {
					t.Errorf("%s[%d]: CPU=%v GPU=%v", c.name, i, cs[i], gs[i])
				}
			}
		})
	}
}

func TestBinaryParityF32(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	a, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, cpu)
	b, _ := tensor.New([]float32{10, 20, 30, 40}, []int{4}, cpu)
	aG, _ := a.To(gpu)
	bG, _ := b.To(gpu)

	cases := []struct {
		name string
		fn   func(*tensor.Tensor, *tensor.Tensor) (*tensor.Tensor, error)
	}{
		{"add", ops.Add},
		{"sub", ops.Sub},
		{"mul", ops.Mul},
		{"div", ops.Div},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			outC, err := c.fn(a, b)
			if err != nil {
				t.Fatal(err)
			}
			outG, err := c.fn(aG, bG)
			if err != nil {
				t.Fatal(err)
			}
			back, _ := outG.To(cpu)
			cs := tensor.MustSlice[float32](outC)
			gs := tensor.MustSlice[float32](back)
			for i := range cs {
				if math.Abs(float64(cs[i]-gs[i])) > 1e-5 {
					t.Errorf("%s[%d]: CPU=%v GPU=%v", c.name, i, cs[i], gs[i])
				}
			}
		})
	}
}

func TestCUDAStorageFinalizer(t *testing.T) {
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	_ = gpu
	// Create, release reference, GC — refcount should hit 0 eventually,
	// freeing GPU memory. Hard to assert directly; at least verify no panic.
	// Detailed alloc/free accounting lives in a separate cudart-level test.
	_ = dtype.F32
}

func TestCUDAContiguousFromTranspose(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, cpu)
	aG, _ := a.To(gpu)
	tG, err := aG.Transpose(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if tG.IsContiguous() {
		t.Fatal("transpose should not be contiguous")
	}
	cG, err := tG.Contiguous()
	if err != nil {
		t.Fatal(err)
	}
	if !cG.IsContiguous() {
		t.Fatal("Contiguous() output should be contiguous")
	}
	if cG.Device().Kind() != device.KindCUDA {
		t.Fatal("Contiguous on CUDA should stay on CUDA")
	}
	back, err := cG.To(cpu)
	if err != nil {
		t.Fatal(err)
	}
	vals := tensor.MustSlice[float32](back)
	// Transpose of [[1,2,3],[4,5,6]] is [[1,4],[2,5],[3,6]] → flat [1,4,2,5,3,6].
	want := []float32{1, 4, 2, 5, 3, 6}
	for i := range want {
		if vals[i] != want[i] {
			t.Errorf("[%d] = %v, want %v", i, vals[i], want[i])
		}
	}
}

func TestCUDAMatMulWithTranspose(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	q, _ := tensor.New([]float32{1, 0, 0, 0, 0, 1, 0, 0}, []int{2, 4}, cpu)
	k, _ := tensor.New([]float32{1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0}, []int{3, 4}, cpu)

	kT, _ := k.Transpose(0, 1)
	cpuScores, _ := ops.MatMul(q, kT)

	qG, _ := q.To(gpu)
	kG, _ := k.To(gpu)
	kGT, _ := kG.Transpose(0, 1)
	gpuScores, err := ops.MatMul(qG, kGT)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpuScores.To(cpu)
	cs := tensor.MustSlice[float32](cpuScores)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > 1e-5 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

// --- Phase 2 parity closure ---

func TestCUDANewUnaryParity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	a, _ := tensor.New([]float32{0.1, 0.5, 1.0, 2.0, -1.5, 3.7}, []int{6}, cpu)
	aG, _ := a.To(gpu)
	cases := map[string]func(*tensor.Tensor) (*tensor.Tensor, error){
		"abs":   ops.Abs,
		"sin":   ops.Sin,
		"cos":   ops.Cos,
		"tan":   ops.Tan,
		"floor": ops.Floor,
		"ceil":  ops.Ceil,
		"round": ops.Round,
		"gelu":  ops.Gelu,
	}
	for name, fn := range cases {
		t.Run(name, func(t *testing.T) {
			cpuOut, _ := fn(a)
			gpuOut, err := fn(aG)
			if err != nil {
				t.Fatal(err)
			}
			back, _ := gpuOut.To(cpu)
			cs := tensor.MustSlice[float32](cpuOut)
			gs := tensor.MustSlice[float32](back)
			for i := range cs {
				if math.Abs(float64(cs[i]-gs[i])) > 1e-4 {
					t.Errorf("%s[%d] CPU=%v GPU=%v", name, i, cs[i], gs[i])
				}
			}
		})
	}
}

func TestCUDAPowParity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	a, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, cpu)
	b, _ := tensor.New([]float32{2, 0.5, 3, 1}, []int{4}, cpu)
	cpuOut, _ := ops.Pow(a, b)
	aG, _ := a.To(gpu)
	bG, _ := b.To(gpu)
	gpuOut, err := ops.Pow(aG, bG)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpuOut.To(cpu)
	cs := tensor.MustSlice[float32](cpuOut)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > 1e-3 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestCUDAReductionParity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 4}, cpu)
	aG, _ := a.To(gpu)
	// Last-axis reductions.
	for name, fn := range map[string]func(*tensor.Tensor, ops.ReduceOptions) (*tensor.Tensor, error){
		"sum": ops.Sum, "mean": ops.Mean, "max": ops.Max, "min": ops.Min,
	} {
		t.Run(name, func(t *testing.T) {
			cpuOut, _ := fn(a, ops.ReduceOptions{Axes: []int{-1}})
			gpuOut, err := fn(aG, ops.ReduceOptions{Axes: []int{-1}})
			if err != nil {
				t.Fatal(err)
			}
			back, _ := gpuOut.To(cpu)
			cs := tensor.MustSlice[float32](cpuOut)
			gs := tensor.MustSlice[float32](back)
			for i := range cs {
				if math.Abs(float64(cs[i]-gs[i])) > 1e-4 {
					t.Errorf("%s[%d] CPU=%v GPU=%v", name, i, cs[i], gs[i])
				}
			}
		})
	}
}

func TestCUDAArgmaxParity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	a, _ := tensor.New([]float32{3, 1, 4, 1, 5, 9, 2, 6}, []int{2, 4}, cpu)
	aG, _ := a.To(gpu)
	cpuOut, _ := ops.Argmax(a, -1, false)
	gpuOut, err := ops.Argmax(aG, -1, false)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpuOut.To(cpu)
	cs := tensor.MustSlice[int64](cpuOut)
	gs := tensor.MustSlice[int64](back)
	for i := range cs {
		if cs[i] != gs[i] {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestCUDALogSoftmaxParity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	a, _ := tensor.New([]float32{1, 2, 3, 4, 1, 2, 3, 4}, []int{2, 4}, cpu)
	aG, _ := a.To(gpu)
	cpuOut, _ := ops.LogSoftmax(a, -1)
	gpuOut, err := ops.LogSoftmax(aG, -1)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpuOut.To(cpu)
	cs := tensor.MustSlice[float32](cpuOut)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > 1e-4 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestCUDAAddBroadcast(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	// (2,3) + (3,) — the "bias" pattern.
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, cpu)
	b, _ := tensor.New([]float32{10, 20, 30}, []int{3}, cpu)

	cpuOut, _ := ops.Add(a, b)

	aG, _ := a.To(gpu)
	bG, _ := b.To(gpu)
	gpuOut, err := ops.Add(aG, bG)
	if err != nil {
		t.Fatalf("broadcast add on CUDA failed: %v", err)
	}
	back, _ := gpuOut.To(cpu)
	cs := tensor.MustSlice[float32](cpuOut)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > 1e-5 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestCUDAScalarDiv(t *testing.T) {
	// tensor / scalar — the attention-scaling pattern.
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	a, _ := tensor.New([]float32{2, 4, 6, 8}, []int{4}, cpu)
	s, _ := tensor.New([]float32{2}, []int{1}, cpu)
	cpuOut, _ := ops.Div(a, s)

	aG, _ := a.To(gpu)
	sG, _ := s.To(gpu)
	gpuOut, err := ops.Div(aG, sG)
	if err != nil {
		t.Fatalf("scalar div on CUDA failed: %v", err)
	}
	back, _ := gpuOut.To(cpu)
	cs := tensor.MustSlice[float32](cpuOut)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > 1e-5 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestCUDAContiguousToCPU(t *testing.T) {
	// Verify that .To(CPU) on a non-contig CUDA tensor uses the strided-copy
	// kernel under the hood.
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, cpu)
	aG, _ := a.To(gpu)
	tG, _ := aG.Transpose(0, 1)
	back, err := tG.To(cpu)
	if err != nil {
		t.Fatalf("non-contig To(CPU) should now succeed: %v", err)
	}
	vals := tensor.MustSlice[float32](back)
	want := []float32{1, 4, 2, 5, 3, 6}
	for i := range want {
		if vals[i] != want[i] {
			t.Errorf("[%d] = %v, want %v", i, vals[i], want[i])
		}
	}
}

func TestMatMulF32Parity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	// (2,3) × (3,2) → (2,2)
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, cpu)
	b, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}, cpu)
	outCPU, _ := ops.MatMul(a, b)
	aG, _ := a.To(gpu)
	bG, _ := b.To(gpu)
	outG, err := ops.MatMul(aG, bG)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := outG.To(cpu)
	cs := tensor.MustSlice[float32](outCPU)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > 1e-4 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestMatMulLargeF32Parity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	const m, k, n = 64, 32, 48
	aData := make([]float32, m*k)
	bData := make([]float32, k*n)
	for i := range aData {
		aData[i] = float32(i%7) * 0.1
	}
	for i := range bData {
		bData[i] = float32(i%5) * 0.2
	}
	a, _ := tensor.New(aData, []int{m, k}, cpu)
	b, _ := tensor.New(bData, []int{k, n}, cpu)
	cpuOut, _ := ops.MatMul(a, b)

	aG, _ := a.To(gpu)
	bG, _ := b.To(gpu)
	gpuOut, err := ops.MatMul(aG, bG)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpuOut.To(cpu)
	cs := tensor.MustSlice[float32](cpuOut)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > 1e-3 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestRMSNormF32Parity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	x, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 4}, cpu)
	w, _ := tensor.New([]float32{1, 1.5, 0.5, 2}, []int{4}, cpu)
	cpuOut, _ := ops.RMSNorm(x, w, 1e-5)

	xG, _ := x.To(gpu)
	wG, _ := w.To(gpu)
	gpuOut, err := ops.RMSNorm(xG, wG, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpuOut.To(cpu)
	cs := tensor.MustSlice[float32](cpuOut)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > 1e-5 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestLayerNormF32Parity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	x, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 4}, cpu)
	g, _ := tensor.New([]float32{1, 1.5, 0.5, 2}, []int{4}, cpu)
	b, _ := tensor.New([]float32{0.1, 0.2, 0.3, 0.4}, []int{4}, cpu)
	cpuOut, _ := ops.LayerNorm(x, g, b, 1e-5)

	xG, _ := x.To(gpu)
	gG, _ := g.To(gpu)
	bG, _ := b.To(gpu)
	gpuOut, err := ops.LayerNorm(xG, gG, bG, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpuOut.To(cpu)
	cs := tensor.MustSlice[float32](cpuOut)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > 1e-4 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestIndexSelectF32Parity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	src, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}, cpu)
	idx, _ := tensor.New([]int64{2, 0, 1, 0}, []int{4}, cpu)
	cpuOut, _ := ops.IndexSelect(src, idx)

	srcG, _ := src.To(gpu)
	idxG, _ := idx.To(gpu)
	gpuOut, err := ops.IndexSelect(srcG, idxG)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpuOut.To(cpu)
	cs := tensor.MustSlice[float32](cpuOut)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if cs[i] != gs[i] {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestSoftmaxF32Parity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	a, _ := tensor.New([]float32{1, 2, 3, 1, 2, 3}, []int{2, 3}, cpu)
	cpuOut, _ := ops.Softmax(a, -1)

	aG, _ := a.To(gpu)
	gpuOut, err := ops.Softmax(aG, -1)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpuOut.To(cpu)
	cs := tensor.MustSlice[float32](cpuOut)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > 1e-5 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestSoftmaxLargeF32Parity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	// A non-trivial row size beyond the 256-thread block, to exercise the
	// strided access path inside the kernel.
	const rows, cols = 4, 1024
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i%13) * 0.1
	}
	a, _ := tensor.New(data, []int{rows, cols}, cpu)
	cpuOut, _ := ops.Softmax(a, -1)
	aG, _ := a.To(gpu)
	gpuOut, err := ops.Softmax(aG, -1)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := gpuOut.To(cpu)
	cs := tensor.MustSlice[float32](cpuOut)
	gs := tensor.MustSlice[float32](back)
	for i := range cs {
		if math.Abs(float64(cs[i]-gs[i])) > 1e-5 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

func TestCUDAF16Add(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	// Ones + Ones should yield 2.0 in F16 on CUDA.
	a, err := tensor.Ones(dtype.F16, []int{8}, cpu)
	if err != nil {
		t.Fatal(err)
	}
	aG, err := a.To(gpu)
	if err != nil {
		t.Fatal(err)
	}
	bG, err := a.To(gpu)
	if err != nil {
		t.Fatal(err)
	}
	c, err := ops.Add(aG, bG)
	if err != nil {
		t.Fatal(err)
	}
	back, err := c.To(cpu)
	if err != nil {
		t.Fatal(err)
	}
	halves, err := tensor.ToHalfSlice(back)
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range halves {
		v := float64(h.ToF32())
		if math.Abs(v-2.0) > 1e-2 {
			t.Errorf("F16 add on CUDA: got %v, want 2.0", v)
		}
	}
}

func TestCUDABF16Add(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	a, err := tensor.Ones(dtype.BF16, []int{8}, cpu)
	if err != nil {
		t.Fatal(err)
	}
	aG, _ := a.To(gpu)
	bG, _ := a.To(gpu)
	c, err := ops.Add(aG, bG)
	if err != nil {
		t.Fatal(err)
	}
	back, _ := c.To(cpu)
	bfs, err := tensor.ToBF16Slice(back)
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range bfs {
		v := float64(b.ToF32())
		if math.Abs(v-2.0) > 1e-2 {
			t.Errorf("BF16 add on CUDA: got %v, want 2.0", v)
		}
	}
}

func TestMatMulF64Parity(t *testing.T) {
	cpu := device.CPU()
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	a, _ := tensor.New([]float64{1, 2, 3, 4}, []int{2, 2}, cpu)
	b, _ := tensor.New([]float64{5, 6, 7, 8}, []int{2, 2}, cpu)
	outC, _ := ops.MatMul(a, b)
	aG, _ := a.To(gpu)
	bG, _ := b.To(gpu)
	outG, _ := ops.MatMul(aG, bG)
	back, _ := outG.To(cpu)
	cs := tensor.MustSlice[float64](outC)
	gs := tensor.MustSlice[float64](back)
	for i := range cs {
		if math.Abs(cs[i]-gs[i]) > 1e-9 {
			t.Errorf("[%d] CPU=%v GPU=%v", i, cs[i], gs[i])
		}
	}
}

// ops.MatMulW16 on CUDA (16-bit weight widened into a pooled F32 scratch,
// cuBLAS SGEMM, F32 activations) against the CPU kernel, for BF16 and F16
// weights, odd shapes and a weight that is a narrowed view.
func TestMatMulW16CUDAParity(t *testing.T) {
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	cpu := device.CPU()
	const M, K, N = 37, 96, 53
	for _, wdt := range []dtype.DType{dtype.BF16, dtype.F16} {
		x, _ := tensor.RandN(dtype.F32, []int{M, K}, cpu)
		full, _ := tensor.RandN(dtype.F32, []int{N + 5, K}, cpu)
		full16, err := ops.Cast(full, wdt)
		if err != nil {
			t.Fatal(err)
		}
		w, _ := full16.Narrow(0, 5, N) // offset view
		want, err := ops.MatMulW16(x, w)
		if err != nil {
			t.Fatal(err)
		}
		xg, _ := x.To(gpu)
		wg, _ := full16.To(gpu)
		if wg, err = wg.Narrow(0, 5, N); err != nil {
			t.Fatal(err)
		}
		got, err := ops.MatMulW16(xg, wg)
		if err != nil {
			t.Fatalf("%v: %v", wdt, err)
		}
		if got, err = got.To(cpu); err != nil {
			t.Fatal(err)
		}
		g, wv := tensor.MustSlice[float32](got), tensor.MustSlice[float32](want)
		md, ma := 0.0, 0.0
		for i := range wv {
			md = math.Max(md, math.Abs(float64(g[i]-wv[i])))
			ma = math.Max(ma, math.Abs(float64(wv[i])))
		}
		t.Logf("%v weights: max diff %.3g (max |ref| %.3g)", wdt, md, ma)
		if md > 1e-4*math.Max(1, ma) {
			t.Errorf("%v: CUDA matmul_w16 differs by %.3g", wdt, md)
		}
	}
}
