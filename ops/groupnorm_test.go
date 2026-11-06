package ops_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// referenceGroupNorm computes GroupNorm in plain Go for parity comparison.
func referenceGroupNorm(x []float32, gamma, beta []float32, n, c, h, w, g int, eps float64) []float32 {
	cg := c / g
	hw := h * w
	m := cg * hw
	out := make([]float32, len(x))
	for ni := 0; ni < n; ni++ {
		for gi := 0; gi < g; gi++ {
			var sum, sumSq float64
			for ci := 0; ci < cg; ci++ {
				ch := gi*cg + ci
				rowOff := (ni*c + ch) * hw
				for k := 0; k < hw; k++ {
					v := float64(x[rowOff+k])
					sum += v
					sumSq += v * v
				}
			}
			invM := 1.0 / float64(m)
			mean := sum * invM
			variance := sumSq*invM - mean*mean
			if variance < 0 {
				variance = 0
			}
			invSigma := 1.0 / math.Sqrt(variance+eps)
			for ci := 0; ci < cg; ci++ {
				ch := gi*cg + ci
				rowOff := (ni*c + ch) * hw
				for k := 0; k < hw; k++ {
					xh := (float64(x[rowOff+k]) - mean) * invSigma
					out[rowOff+k] = float32(xh*float64(gamma[ch]) + float64(beta[ch]))
				}
			}
		}
	}
	return out
}

func TestGroupNormCPUParity(t *testing.T) {
	N, C, H, W, G := 2, 8, 4, 4, 4
	xs := make([]float32, N*C*H*W)
	for i := range xs {
		xs[i] = float32(i)*0.13 - 5
	}
	gamma := make([]float32, C)
	beta := make([]float32, C)
	for i := range gamma {
		gamma[i] = 1 + float32(i)*0.01
		beta[i] = float32(i) * -0.02
	}
	expect := referenceGroupNorm(xs, gamma, beta, N, C, H, W, G, 1e-5)

	cpu := device.CPU()
	x, _ := tensor.New(xs, []int{N, C, H, W}, cpu)
	g, _ := tensor.New(gamma, []int{C}, cpu)
	b, _ := tensor.New(beta, []int{C}, cpu)
	got, err := ops.GroupNorm(x, g, b, G, 1e-5)
	if err != nil {
		t.Fatalf("CPU GroupNorm: %v", err)
	}
	gotS := tensor.MustSlice[float32](got)
	for i := range gotS {
		if math.Abs(float64(gotS[i]-expect[i])) > 1e-4 {
			t.Fatalf("CPU mismatch at %d: got %v, want %v", i, gotS[i], expect[i])
		}
	}
}

func TestGroupNormCUDAParity(t *testing.T) {
	if !device.CUDAAvailable() {
		t.Skip("CUDA unavailable")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Skipf("CUDA(0): %v", err)
	}
	N, C, H, W, G := 1, 32, 8, 8, 32
	xs := make([]float32, N*C*H*W)
	for i := range xs {
		xs[i] = float32(i)*0.07 - 3
	}
	gamma := make([]float32, C)
	beta := make([]float32, C)
	for i := range gamma {
		gamma[i] = 1 + float32(i)*0.01
		beta[i] = float32(i) * -0.02
	}
	expect := referenceGroupNorm(xs, gamma, beta, N, C, H, W, G, 1e-5)

	x, _ := tensor.New(xs, []int{N, C, H, W}, dev)
	g, _ := tensor.New(gamma, []int{C}, dev)
	b, _ := tensor.New(beta, []int{C}, dev)
	got, err := ops.GroupNorm(x, g, b, G, 1e-5)
	if err != nil {
		t.Fatalf("CUDA GroupNorm: %v", err)
	}
	gotCPU, _ := got.To(device.CPU())
	gotS := tensor.MustSlice[float32](gotCPU)
	// 5e-3 absolute tolerance accommodates `-use_fast_math` rsqrtf drift on
	// the GPU vs the F64-reduction reference. Same threshold used for our
	// BatchNorm CUDA parity tests.
	for i := range gotS {
		if math.Abs(float64(gotS[i]-expect[i])) > 5e-3 {
			t.Fatalf("CUDA mismatch at %d: got %v, want %v", i, gotS[i], expect[i])
		}
	}
	_ = dtype.F32
}
