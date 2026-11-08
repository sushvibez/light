//go:build !nowebgpu

package webgpu_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/ops"
	"light/tensor"
)

// RoPE on WebGPU (rotate-half, interleaved, and with caller inverse
// frequencies) against the CPU kernels, at positions up to 32k where the
// f32 angle (pos × invFreq on the GPU, f64 on the CPU) matters most.
func TestRoPEParity(t *testing.T) {
	g, err := device.WebGPU(0)
	if err != nil {
		t.Skip(err)
	}
	cpu := device.CPU()
	rng := rand.New(rand.NewPCG(5, 6))
	const B, H, D = 1, 3, 64
	for _, positions := range [][]int64{{0, 1, 2, 3, 4, 5, 6, 7}, {41}, {1000, 4095, 8191, 32767}} {
		T := len(positions)
		xs := make([]float32, B*H*T*D)
		for i := range xs {
			xs[i] = float32(rng.NormFloat64())
		}
		x, _ := tensor.New(xs, []int{B, H, T, D}, cpu)
		p, _ := tensor.New(positions, []int{T}, cpu)
		xg, _ := x.To(g)
		pg, _ := p.To(g)
		inv := make([]float32, D/2)
		for i := range inv {
			inv[i] = float32(math.Pow(500000, -float64(2*i)/D)) * 0.25
		}
		ic, _ := tensor.New(inv, []int{D / 2}, cpu)
		ig, _ := ic.To(g)
		for _, c := range []struct {
			name string
			f    func(x, p, inv *tensor.Tensor) (*tensor.Tensor, error)
		}{
			{"rope", func(x, p, _ *tensor.Tensor) (*tensor.Tensor, error) { return ops.RoPE(x, p, 10000) }},
			{"rope_i", func(x, p, _ *tensor.Tensor) (*tensor.Tensor, error) { return ops.RoPEInterleaved(x, p, 10000) }},
			{"rope_scaled", func(x, p, inv *tensor.Tensor) (*tensor.Tensor, error) { return ops.RoPEWithInvFreq(x, p, inv) }},
		} {
			want, err := c.f(x, p, ic)
			if err != nil {
				t.Fatal(err)
			}
			got, err := c.f(xg, pg, ig)
			if err != nil {
				t.Fatal(err)
			}
			got, _ = got.To(cpu)
			a, b := tensor.MustSlice[float32](want), tensor.MustSlice[float32](got)
			var md float64
			for i := range a {
				md = max(md, math.Abs(float64(a[i]-b[i])))
			}
			// The angle's f32 rounding is ~pos·6e-8 rad: 2e-3 at 32k.
			tol := 1e-5 + 1e-7*float64(positions[len(positions)-1])
			if md > tol {
				t.Errorf("%s positions %v: max |diff| %.3g > %.3g", c.name, positions, md, tol)
			} else {
				t.Logf("%s positions %v: max |diff| %.3g", c.name, positions, md)
			}
		}
	}
}
