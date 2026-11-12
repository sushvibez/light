//go:build !nocuda && cgo

package ops_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// The fused decode GEMVs must match the unfused BF16 path (cuBLAS GemmEx
// on CUDA, the CPU matmul there) to BF16 rounding of a differently
// ordered F32 sum.
func TestGemvCUDAParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	rng := rand.New(rand.NewPCG(11, 13))
	K := 1024
	x := randTensor(t, rng, []int{1, 1, K}, 1, device.CPU(), dtype.BF16)
	ws := []*tensor.Tensor{
		randTensor(t, rng, []int{6144, K}, 0.05, device.CPU(), dtype.BF16),
		randTensor(t, rng, []int{2048, K}, 0.05, device.CPU(), dtype.BF16),
		randTensor(t, rng, []int{16, K}, 0.05, device.CPU(), dtype.BF16),
		randTensor(t, rng, []int{16, K}, 0.05, device.CPU(), dtype.BF16),
	}
	to := func(x *tensor.Tensor) *tensor.Tensor {
		y, err := x.To(gpu)
		if err != nil {
			t.Fatal(err)
		}
		return y
	}
	check := func(name string, got, want *tensor.Tensor, tol float64) {
		g, w := f32(t, got), f32(t, want)
		if len(g) != len(w) {
			t.Fatalf("%s: %d vs %d elements", name, len(g), len(w))
		}
		for i := range g {
			if d := math.Abs(float64(g[i] - w[i])); d > tol*math.Max(0.05, math.Abs(float64(w[i]))) {
				t.Fatalf("%s[%d] = %v, want %v", name, i, g[i], w[i])
			}
		}
	}
	for _, outDT := range []dtype.DType{dtype.BF16, dtype.F32} {
		cpu, err := ops.LinearMulti(x, ws, outDT)
		if err != nil {
			t.Fatal(err)
		}
		gws := make([]*tensor.Tensor, len(ws))
		for i := range ws {
			gws[i] = to(ws[i])
		}
		got, err := ops.LinearMulti(to(x), gws, outDT)
		if err != nil {
			t.Fatal(err)
		}
		for i := range ws {
			if !got[i].Shape().Equal(cpu[i].Shape()) || got[i].DType() != outDT {
				t.Fatalf("out %d: %v %v", i, got[i].Shape(), got[i].DType())
			}
			check("multi", got[i], cpu[i], 2e-2)
		}
	}
	wg := randTensor(t, rng, []int{3584, K}, 0.05, device.CPU(), dtype.BF16)
	wu := randTensor(t, rng, []int{3584, K}, 0.05, device.CPU(), dtype.BF16)
	for _, gelu := range []bool{false, true} {
		cpu, err := ops.GatedLinear(x, wg, wu, gelu)
		if err != nil {
			t.Fatal(err)
		}
		got, err := ops.GatedLinear(to(x), to(wg), to(wu), gelu)
		if err != nil {
			t.Fatal(err)
		}
		check("gated", got, cpu, 3e-2)
	}
	// Timing of the fused pair against the unfused one at Llama-1B's MLP.
	wg2 := to(randTensor(t, rng, []int{8192, 2048}, 0.05, device.CPU(), dtype.BF16))
	wu2 := to(randTensor(t, rng, []int{8192, 2048}, 0.05, device.CPU(), dtype.BF16))
	x2 := to(randTensor(t, rng, []int{1, 2048}, 1, device.CPU(), dtype.BF16))
	if _, err := ops.GatedLinear(x2, wg2, wu2, false); err != nil {
		t.Fatal(err)
	}
}
