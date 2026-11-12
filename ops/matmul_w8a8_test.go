//go:build !nocuda && cgo

package ops_test

import (
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/ops"
	"light/tensor"
)

// The CUDA W8A8 path (row quantiser → cublasGemmEx int8 → scale epilogue)
// must reproduce the Go reference bit for bit in int32 (the float epilogue
// then differs only by rounding), and both must track the F32 product to
// int8 precision.
func TestMatMulW8A8CUDAParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	rng := rand.New(rand.NewPCG(3, 4))
	for _, tc := range []struct{ M, K, N int }{{1, 256, 512}, {7, 1024, 1000}, {64, 2048, 2048}, {130, 512, 4}} {
		xd := make([]float32, tc.M*tc.K)
		for i := range xd {
			xd[i] = float32(rng.NormFloat64())
		}
		wd := make([]int8, tc.N*tc.K)
		for i := range wd {
			wd[i] = int8(rng.IntN(255) - 127)
		}
		sd := make([]float32, tc.N)
		for i := range sd {
			sd[i] = float32(0.005 + 0.02*rng.Float64())
		}
		x, _ := tensor.New(xd, []int{tc.M, tc.K}, device.CPU())
		wq, _ := tensor.New(wd, []int{tc.N, tc.K}, device.CPU())
		ws, _ := tensor.New(sd, []int{tc.N}, device.CPU())
		want, err := ops.MatMulW8A8(x, wq, ws)
		if err != nil {
			t.Fatal(err)
		}
		xG, _ := x.To(gpu)
		wG, _ := wq.To(gpu)
		sG, _ := ws.To(gpu)
		got, err := ops.MatMulW8A8(xG, wG, sG)
		if err != nil {
			t.Fatal(err)
		}
		gc, _ := got.To(device.CPU())
		w, g := tensor.MustSlice[float32](want), tensor.MustSlice[float32](gc)
		var maxRel float64
		for i := range w {
			d := math.Abs(float64(g[i]-w[i])) / math.Max(1e-6, math.Abs(float64(w[i])))
			if d > maxRel {
				maxRel = d
			}
		}
		if maxRel > 1e-4 {
			t.Fatalf("%dx%dx%d: CUDA vs reference max relative error %.3g", tc.M, tc.K, tc.N, maxRel)
		}
		// Against the exact F32 product of the dequantised operands: the
		// activation's int8 rounding bounds the error.
		wf := make([]float32, len(wd))
		for i, v := range wd {
			wf[i] = float32(v) * sd[i/tc.K]
		}
		var num, den float64
		for m := 0; m < tc.M; m++ {
			for n := 0; n < tc.N; n++ {
				var acc float64
				for k := 0; k < tc.K; k++ {
					acc += float64(xd[m*tc.K+k]) * float64(wf[n*tc.K+k])
				}
				num += math.Abs(float64(g[m*tc.N+n]) - acc)
				den += math.Abs(acc)
			}
		}
		t.Logf("%dx%dx%d: CUDA vs Go reference max rel %.2g; vs F32 product %.3f%% mean", tc.M, tc.K, tc.N, maxRel, 100*num/den)
		if num/den > 0.02 {
			t.Fatalf("%dx%dx%d: %.3f%% off the F32 product", tc.M, tc.K, tc.N, 100*num/den)
		}
	}
}

// TestMatMulW8A8Bench prints W8A8 against BF16 cuBLAS at prefill shapes
// (LIGHT_HEAVY_TESTS not needed; a few hundred ms).
func TestMatMulW8A8Bench(t *testing.T) {
	gpu := gpuOrSkip(t)
	if testing.Short() {
		t.Skip()
	}
	rng := rand.New(rand.NewPCG(5, 6))
	const K, N = 4096, 4096
	wd := make([]int8, N*K)
	for i := range wd {
		wd[i] = int8(rng.IntN(255) - 127)
	}
	sd := make([]float32, N)
	for i := range sd {
		sd[i] = 0.01
	}
	wq, _ := tensor.New(wd, []int{N, K}, device.CPU())
	ws, _ := tensor.New(sd, []int{N}, device.CPU())
	wG, _ := wq.To(gpu)
	sG, _ := ws.To(gpu)
	wbf, _ := tensor.New(make([]float32, N*K), []int{N, K}, device.CPU())
	wbfG, _ := wbf.To(gpu)
	wbfG, _ = ops.Cast(wbfG, dtype.BF16)
	wT, _ := wbfG.Transpose(0, 1)
	for _, M := range []int{16, 128, 512, 2048} {
		xd := make([]float32, M*K)
		for i := range xd {
			xd[i] = float32(rng.NormFloat64())
		}
		x, _ := tensor.New(xd, []int{M, K}, device.CPU())
		xG, _ := x.To(gpu)
		time1 := func(f func()) float64 {
			f()
			syncCUDA(t)
			const n = 20
			t0 := time.Now()
			for i := 0; i < n; i++ {
				f()
			}
			syncCUDA(t)
			return float64(time.Since(t0).Microseconds()) / n / 1000
		}
		msI8 := time1(func() { _, _ = ops.MatMulW8A8(xG, wG, sG) })
		msBF := time1(func() { _, _ = ops.MatMulMixed(xG, wT) })
		flops := 2 * float64(M) * K * N
		t.Logf("M=%-5d W8A8 %.3f ms (%.1f TOPS)   BF16 cuBLAS (F32 act, BF16 weight) %.3f ms (%.1f TFLOPS)", M, msI8, flops/msI8/1e9, msBF, flops/msBF/1e9)
	}
}

func syncCUDA(t testing.TB) {
	t.Helper()
	if err := cudart.DeviceSynchronize(); err != nil {
		t.Fatal(err)
	}
}
