//go:build !nocuda && cgo

package ops_test

import (
	"math/rand/v2"
	"os"
	"testing"
	"time"

	"light/dtype"
	"light/internal/cudart"
	"light/ops"
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
		t.Logf("%-28s %7.1f µs  %6.1f GB/s", tc.name, float64(el.Nanoseconds())/1000, bytes/el.Seconds()/1e9)
	}
}
