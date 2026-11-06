//go:build !nocuda

package ops_test

import (
	"math/rand/v2"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// Sort must be bit-identical between CPU (stable sort) and the CUDA bitonic
// kernels, including tie order, for both paths (shared ≤1024, global
// beyond) and all four dtypes. 151936 is a Qwen vocab row.
func TestCUDASortKernelParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	rng := rand.New(rand.NewPCG(11, 13))
	for _, cols := range []int{1, 2, 9, 1000, 1024, 1025, 4097, 151936} {
		rows := 3
		if cols > 10000 {
			rows = 1
		}
		// Small value range so ties are common.
		f := make([]float32, rows*cols)
		for i := range f {
			f[i] = float32(rng.IntN(cols/4 + 2))
		}
		a, _ := tensor.New(f, []int{rows, cols}, device.CPU())
		aG, _ := a.To(gpu)
		for _, desc := range []bool{false, true} {
			cv, ci, err := ops.Sort(a, desc)
			if err != nil {
				t.Fatal(err)
			}
			gv, gi, err := ops.Sort(aG, desc)
			if err != nil {
				t.Fatalf("cols=%d desc=%v: %v", cols, desc, err)
			}
			if !device.IsCUDA(gv.Device()) || !device.IsCUDA(gi.Device()) {
				t.Fatal("Sort outputs left the device")
			}
			gvC, _ := gv.To(device.CPU())
			giC, _ := gi.To(device.CPU())
			if !slices.Equal(tensor.MustSlice[float32](cv), tensor.MustSlice[float32](gvC)) {
				t.Errorf("cols=%d desc=%v: values differ", cols, desc)
			}
			if !slices.Equal(tensor.MustSlice[int64](ci), tensor.MustSlice[int64](giC)) {
				t.Errorf("cols=%d desc=%v: indices differ (tie order)", cols, desc)
			}
		}
	}
	// Integer and F64 dtypes on both paths.
	for _, cols := range []int{37, 3000} {
		i64 := make([]int64, 2*cols)
		for i := range i64 {
			i64[i] = int64(rng.IntN(50) - 25)
		}
		for _, dt := range []dtype.DType{dtype.I64, dtype.I32, dtype.F64} {
			var a *tensor.Tensor
			switch dt {
			case dtype.I64:
				a, _ = tensor.New(i64, []int{2, cols}, device.CPU())
			case dtype.I32:
				v := make([]int32, len(i64))
				for i, x := range i64 {
					v[i] = int32(x)
				}
				a, _ = tensor.New(v, []int{2, cols}, device.CPU())
			case dtype.F64:
				v := make([]float64, len(i64))
				for i, x := range i64 {
					v[i] = float64(x) / 3
				}
				a, _ = tensor.New(v, []int{2, cols}, device.CPU())
			}
			aG, _ := a.To(gpu)
			_, ci, _ := ops.Sort(a, true)
			_, gi, err := ops.Sort(aG, true)
			if err != nil {
				t.Fatalf("%v cols=%d: %v", dt, cols, err)
			}
			giC, _ := gi.To(device.CPU())
			if !slices.Equal(tensor.MustSlice[int64](ci), tensor.MustSlice[int64](giC)) {
				t.Errorf("%v cols=%d: indices differ", dt, cols)
			}
		}
	}
}

func TestCUDATopKParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	rng := rand.New(rand.NewPCG(2, 4))
	for _, cols := range []int{16, 5000} {
		f := make([]float32, 2*cols)
		for i := range f {
			f[i] = float32(rng.IntN(100))
		}
		a, _ := tensor.New(f, []int{2, cols}, device.CPU())
		aG, _ := a.To(gpu)
		for _, largest := range []bool{true, false} {
			cv, ci, _ := ops.TopK(a, 5, largest)
			gv, gi, err := ops.TopK(aG, 5, largest)
			if err != nil {
				t.Fatal(err)
			}
			if !device.IsCUDA(gv.Device()) {
				t.Fatal("TopK left the device")
			}
			gvC, _ := gv.To(device.CPU())
			giC, _ := gi.To(device.CPU())
			if !slices.Equal(tensor.MustSlice[float32](cv), tensor.MustSlice[float32](gvC)) ||
				!slices.Equal(tensor.MustSlice[int64](ci), tensor.MustSlice[int64](giC)) {
				t.Errorf("cols=%d largest=%v: TopK mismatch", cols, largest)
			}
		}
	}
}

// Three-operand and diagonal expressions now run on the device via pairwise
// contraction and AsStrided diagonals; results must match the CPU kernel.
func TestCUDAEinsumGeneralParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	mk := func(dims ...int) (*tensor.Tensor, *tensor.Tensor) {
		a, _ := tensor.RandN(dtype.F32, dims, device.CPU())
		g, _ := a.To(gpu)
		return a, g
	}
	a, aG := mk(3, 4)
	b, bG := mk(4, 5)
	c, cG := mk(5, 6)
	sq, sqG := mk(4, 4)
	cube, cubeG := mk(3, 4, 4)
	cases := []struct {
		n   string
		cpu []*tensor.Tensor
		gpu []*tensor.Tensor
	}{
		{"ij,jk,kl->il", []*tensor.Tensor{a, b, c}, []*tensor.Tensor{aG, bG, cG}},
		{"ij,jk,kl->l", []*tensor.Tensor{a, b, c}, []*tensor.Tensor{aG, bG, cG}},
		{"ii->i", []*tensor.Tensor{sq}, []*tensor.Tensor{sqG}},
		{"ii->", []*tensor.Tensor{sq}, []*tensor.Tensor{sqG}},
		{"bii->bi", []*tensor.Tensor{cube}, []*tensor.Tensor{cubeG}},
		{"bii,ij->bj", []*tensor.Tensor{cube, sq}, []*tensor.Tensor{cubeG, sqG}},
		{"ij,jk,kl,ii->il", []*tensor.Tensor{a, b, c, a}, []*tensor.Tensor{aG, bG, cG, aG}},
	}
	for _, cs := range cases {
		if cs.n == "ij,jk,kl,ii->il" {
			// 'ii' needs a square operand; reuse sq with letters jj instead.
			cs.n = "ij,jk,kl,jj->il"
			cs.cpu[3], cs.gpu[3] = sq, sqG
		}
		want, err := ops.Einsum(cs.n, cs.cpu...)
		if err != nil {
			t.Fatalf("%s cpu: %v", cs.n, err)
		}
		got, err := ops.Einsum(cs.n, cs.gpu...)
		if err != nil {
			t.Fatalf("%s cuda: %v", cs.n, err)
		}
		assertParityF32(t, "Einsum "+cs.n, want, got, 1e-4)
	}
}
