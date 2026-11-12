//go:build !nocuda && cgo

package ops_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
)

// The fused CUDA silu(g)·u → BF16 matches the unfused SiLU, Mul and Cast on
// the CPU to one BF16 rounding (expf differs from Go's exp in the last
// bits), at a length that is not a multiple of the kernel's 4-wide loads.
func TestSiLUMulBF16MatchesUnfused(t *testing.T) {
	gpu := gpuOrSkip(t)
	rng := rand.New(rand.NewPCG(5, 6))
	dims := []int{3, 1001}
	g := randTensor(t, rng, dims, 3, device.CPU(), dtype.F32)
	u := randTensor(t, rng, dims, 2, device.CPU(), dtype.F32)
	want, err := ops.SiLUMulBF16(g, u) // CPU: the unfused fallback
	if err != nil {
		t.Fatal(err)
	}
	gg, _ := g.To(gpu)
	ug, _ := u.To(gpu)
	got, err := ops.SiLUMulBF16(gg, ug)
	if err != nil {
		t.Fatal(err)
	}
	if got.DType() != dtype.BF16 || !got.Shape().Equal(want.Shape()) {
		t.Fatalf("got %v %v, want %v %v", got.DType(), got.Shape(), want.DType(), want.Shape())
	}
	a, b := f32(t, got), f32(t, want)
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > math.Abs(float64(b[i]))/128+1e-30 {
			t.Fatalf("element %d: %g, unfused %g", i, a[i], b[i])
		}
	}
}
