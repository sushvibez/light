//go:build !nocuda && cgo

package ops_test

import (
	"fmt"
	"math/rand/v2"
	"os"
	"testing"
	"time"

	"light/dtype"
	"light/internal/cudart"
	"light/ops"
	"light/tensor"
)

// TestGEMVBandwidth reports the effective weight bandwidth of the decode
// GEMVs (one row against an nn.Linear weight, read as its transpose
// view) at the shapes of Llama-3.2-1B and Qwen3.5-0.8B.
//
//	LIGHT_HEAVY_TESTS=1 go test ./ops -run TestGEMVBandwidth -v
func TestGEMVBandwidth(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1")
	}
	gpu := gpuOrSkip(t)
	rng := rand.New(rand.NewPCG(1, 2))
	for _, tc := range []struct {
		name string
		K, N int
		dt   dtype.DType
	}{
		{"llama q/o 2048→2048", 2048, 2048, dtype.BF16},
		{"llama gate/up 2048→8192", 2048, 8192, dtype.BF16},
		{"llama down 8192→2048", 8192, 2048, dtype.BF16},
		{"llama lm_head 2048→128256", 2048, 128256, dtype.BF16},
		{"qwen3.5 lm_head 1024→248320", 1024, 248320, dtype.BF16},
		{"qwen3.5 up 1024→3072", 1024, 3072, dtype.BF16},
		{"f32 gate/up 2048→8192", 2048, 8192, dtype.F32},
	} {
		w := randTensor(t, rng, []int{tc.N, tc.K}, 1, gpu, tc.dt) // nn.Linear weight (out, in)
		x := randTensor(t, rng, []int{1, tc.K}, 1, gpu, tc.dt)
		wT, _ := w.Transpose(0, 1)
		for i := 0; i < 5; i++ {
			if _, err := ops.MatMul(x, wT); err != nil {
				t.Fatal(err)
			}
		}
		_ = cudart.DeviceSynchronize()
		const iters = 50
		t0 := time.Now()
		for i := 0; i < iters; i++ {
			if _, err := ops.MatMul(x, wT); err != nil {
				t.Fatal(err)
			}
		}
		_ = cudart.DeviceSynchronize()
		el := time.Since(t0) / iters
		bytes := float64(tc.N*tc.K) * float64(tc.dt.Size())
		fused := ""
		if tc.dt == dtype.BF16 {
			xr, _ := x.Reshape(1, 1, tc.K)
			for i := 0; i < 5; i++ {
				if _, err := ops.LinearMulti(xr, []*tensor.Tensor{w}, dtype.BF16); err != nil {
					t.Fatal(err)
				}
			}
			_ = cudart.DeviceSynchronize()
			t1 := time.Now()
			for i := 0; i < iters; i++ {
				if _, err := ops.LinearMulti(xr, []*tensor.Tensor{w}, dtype.BF16); err != nil {
					t.Fatal(err)
				}
			}
			_ = cudart.DeviceSynchronize()
			e2 := time.Since(t1) / iters
			fused = fmt.Sprintf("   gemv_bf16_multi %7.1f µs %6.1f GB/s", float64(e2.Nanoseconds())/1000, bytes/e2.Seconds()/1e9)
		}
		t.Logf("%-28s cuBLAS %7.1f µs %6.1f GB/s%s", tc.name, float64(el.Nanoseconds())/1000, bytes/el.Seconds()/1e9, fused)
	}
	// The gated pair at Llama-1B's and Qwen3.5-0.8B's MLP shapes against
	// two cuBLAS GEMVs plus SiLU and the product.
	for _, tc := range []struct{ K, N int }{{2048, 8192}, {1024, 3584}} {
		wg := randTensor(t, rng, []int{tc.N, tc.K}, 0.05, gpu, dtype.BF16)
		wu := randTensor(t, rng, []int{tc.N, tc.K}, 0.05, gpu, dtype.BF16)
		x := randTensor(t, rng, []int{1, 1, tc.K}, 1, gpu, dtype.BF16)
		x2, _ := x.Reshape(1, tc.K)
		wgT, _ := wg.Transpose(0, 1)
		wuT, _ := wu.Transpose(0, 1)
		unfused := func() {
			g, _ := ops.MatMulMixed(x2, wgT)
			g, _ = ops.SiLU(g)
			u, _ := ops.MatMulMixed(x2, wuT)
			_, _ = ops.Mul(g, u)
		}
		fusedFn := func() { _, _ = ops.GatedLinear(x, wg, wu, false) }
		time1 := func(f func()) float64 {
			for i := 0; i < 5; i++ {
				f()
			}
			_ = cudart.DeviceSynchronize()
			t0 := time.Now()
			for i := 0; i < 50; i++ {
				f()
			}
			_ = cudart.DeviceSynchronize()
			return float64(time.Since(t0).Microseconds()) / 50
		}
		bytes := 2 * float64(tc.N*tc.K) * 2
		a, b := time1(unfused), time1(fusedFn)
		t.Logf("gated %dx%d: unfused %7.1f µs %6.1f GB/s   gemv_bf16_gated %7.1f µs %6.1f GB/s", tc.N, tc.K, a, bytes/a/1e3, b, bytes/b/1e3)
	}
}
