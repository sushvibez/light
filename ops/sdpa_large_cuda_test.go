//go:build !nocuda && cgo

package ops_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/kernels/cuda"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// Attention at FLUX.1's geometry (24 heads, 1280 tokens, 128 wide), F32:
// CUDA against the CPU.
func TestSDPAFluxGeometryParity(t *testing.T) {
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skip(err)
	}
	rng := rand.New(rand.NewPCG(7, 8))
	rnd := func(shape ...int) []float32 {
		n := 1
		for _, d := range shape {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	B, H, S, D := 1, 24, 1280, 128
	q, k, v := rnd(B, H, S, D), rnd(B, H, S, D), rnd(B, H, S, D)
	run := func(dev device.Device) []float32 {
		mk := func(x []float32) *tensor.Tensor {
			t_, err := tensor.New(x, []int{B, H, S, D}, dev)
			if err != nil {
				t.Fatal(err)
			}
			return t_
		}
		o, err := nn.SDPA(mk(q), mk(k), mk(v), nil)
		if err != nil {
			t.Fatal(err)
		}
		if o.DType() != dtype.F32 {
			t.Fatalf("dtype %v", o.DType())
		}
		c, _ := o.To(device.CPU())
		return tensor.MustSlice[float32](c)
	}
	a, b := run(device.CPU()), run(gpu)
	md := 0.0
	for i := range a {
		md = math.Max(md, math.Abs(float64(a[i]-b[i])))
	}
	t.Logf("SDPA (%d, %d, %d, %d): CPU vs CUDA max diff %.3g", B, H, S, D, md)
	if md > 1e-4 {
		t.Errorf("differ by %.3g", md)
	}
	// The batched-GEMM attention, whole and in several query chunks.
	mk := func(x []float32) *tensor.Tensor {
		t_, _ := tensor.New(x, []int{B, H, S, D}, gpu)
		return t_
	}
	for _, budget := range []int64{1 << 30, 64 << 20} {
		restore := cuda.SetAttentionGEMMBudget(budget)
		o, err := ops.AttentionGEMM(mk(q), mk(k), mk(v))
		restore()
		if err != nil {
			t.Fatal(err)
		}
		c, _ := o.To(device.CPU())
		g := tensor.MustSlice[float32](c)
		md := 0.0
		for i := range a {
			md = math.Max(md, math.Abs(float64(a[i]-g[i])))
		}
		t.Logf("AttentionGEMM (budget %d MB): CPU vs CUDA max diff %.3g", budget>>20, md)
		if md > 1e-4 {
			t.Errorf("AttentionGEMM differs by %.3g", md)
		}
	}
}

// Large-magnitude q / k (FLUX's first block has |q| up to ~40 per
// element): the attention must still match a float64 reference, on the
// CPU and on CUDA.
func TestSDPALargeLogits(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	B, H, S, D := 1, 2, 256, 128
	scale := float32(30)
	rnd := func(n int, sc float32) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = sc * float32(rng.NormFloat64())
		}
		return v
	}
	q, k, v := rnd(B*H*S*D, scale), rnd(B*H*S*D, scale), rnd(B*H*S*D, 1)
	// float64 reference
	want := make([]float64, B*H*S*D)
	inv := 1 / math.Sqrt(float64(D))
	for h := range H {
		for i := range S {
			logits := make([]float64, S)
			mx := math.Inf(-1)
			for j := range S {
				var dot float64
				for d := range D {
					dot += float64(q[(h*S+i)*D+d]) * float64(k[(h*S+j)*D+d])
				}
				logits[j] = dot * inv
				mx = math.Max(mx, logits[j])
			}
			var sum float64
			for j := range S {
				logits[j] = math.Exp(logits[j] - mx)
				sum += logits[j]
			}
			for d := range D {
				var acc float64
				for j := range S {
					acc += logits[j] / sum * float64(v[(h*S+j)*D+d])
				}
				want[(h*S+i)*D+d] = acc
			}
		}
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.CUDA(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		mk := func(x []float32) *tensor.Tensor {
			t_, err := tensor.New(x, []int{B, H, S, D}, dev)
			if err != nil {
				t.Fatal(err)
			}
			return t_
		}
		o, err := nn.SDPA(mk(q), mk(k), mk(v), nil)
		if err != nil {
			t.Fatal(err)
		}
		c, _ := o.To(device.CPU())
		got := tensor.MustSlice[float32](c)
		md, nan := 0.0, 0
		for i := range got {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
				nan++
				continue
			}
			md = math.Max(md, math.Abs(float64(got[i])-want[i]))
		}
		t.Logf("%s: max diff vs float64 %.3g, %d non-finite", dev, md, nan)
		if md > 1e-3 || nan > 0 {
			t.Errorf("%s: differ by %.3g, %d non-finite", dev, md, nan)
		}
	}
}
