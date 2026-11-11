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

func randTensor(t *testing.T, rng *rand.Rand, dims []int, scale float32, dev device.Device, dt dtype.DType) *tensor.Tensor {
	t.Helper()
	n := 1
	for _, d := range dims {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(rng.NormFloat64()) * scale
	}
	x, err := tensor.New(data, dims, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if dt != dtype.F32 {
		if x, err = ops.Cast(x, dt); err != nil {
			t.Fatal(err)
		}
	}
	if x, err = x.To(dev); err != nil {
		t.Fatal(err)
	}
	return x
}

func hostF32(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	c, err := x.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if c.DType() != dtype.F32 {
		if c, err = ops.Cast(c, dtype.F32); err != nil {
			t.Fatal(err)
		}
	}
	return tensor.MustSlice[float32](c)
}

func maxRelDiff(a, b []float32) float64 {
	var worst float64
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if s := math.Abs(float64(b[i])); s > 1 {
			d /= s
		}
		worst = max(worst, d)
	}
	return worst
}

// The fused CUDA delta rule (in-place head split, gates computed in the
// kernel) must match the op-by-op path, which is what the CPU runs:
// interleaved and tiled key-head mapping, F32 and BF16 gates, with and
// without an initial state, one token (fused) and several (unfused on both).
func TestGatedDeltaRuleFusedCUDAParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	rng := rand.New(rand.NewPCG(3, 9))
	const B, Hk, Hv, dk, dv = 2, 2, 4, 32, 48
	C := 2*Hk*dk + Hv*dv
	for _, tc := range []struct {
		T     int
		tiled bool
		gates dtype.DType
		state bool
	}{{1, false, dtype.BF16, true}, {1, true, dtype.F32, true}, {1, false, dtype.F32, false}, {3, false, dtype.BF16, true}} {
		mk := func(dev device.Device) (*tensor.Tensor, *tensor.Tensor) {
			r := rand.New(rand.NewPCG(3, 9))
			qkv := randTensor(t, r, []int{B, tc.T, C}, 1, dev, dtype.F32)
			a := randTensor(t, r, []int{B, tc.T, Hv}, 1, dev, tc.gates)
			b := randTensor(t, r, []int{B, tc.T, Hv}, 1, dev, tc.gates)
			dtBias := randTensor(t, r, []int{Hv}, 0.5, dev, dtype.F32)
			negA := randTensor(t, r, []int{Hv}, 0.5, dev, dtype.F32)
			negAH := hostF32(t, negA)
			for i := range negAH {
				negAH[i] = -float32(math.Abs(float64(negAH[i])))
			}
			var err error
			if negA, err = tensor.New(negAH, []int{Hv}, device.CPU()); err != nil {
				t.Fatal(err)
			}
			if negA, err = negA.To(dev); err != nil {
				t.Fatal(err)
			}
			var state *tensor.Tensor
			if tc.state {
				state = randTensor(t, r, []int{B, Hv, dk, dv}, 0.3, dev, dtype.F32)
			}
			out, final, err := ops.GatedDeltaRuleFused(qkv, a, b, dtBias, negA, state, Hk, dk, Hv, dv, tc.tiled, ops.GatedDeltaOptions{QKL2Norm: true})
			if err != nil {
				t.Fatalf("%+v: %v", tc, err)
			}
			return out, final
		}
		co, cf := mk(device.CPU())
		go_, gf := mk(gpu)
		if d := maxRelDiff(hostF32(t, go_), hostF32(t, co)); d > 2e-4 {
			t.Errorf("%+v: output differs by %.3g", tc, d)
		}
		if d := maxRelDiff(hostF32(t, gf), hostF32(t, cf)); d > 2e-4 {
			t.Errorf("%+v: final state differs by %.3g", tc, d)
		}
	}
	_ = rng
}

// GatedRMSNormSiLU on CUDA (one kernel) against the CPU composition, for
// BF16 / F32 gates and outputs.
func TestGatedRMSNormSiLUCUDAParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	for _, tc := range []struct{ z, out dtype.DType }{{dtype.BF16, dtype.BF16}, {dtype.F32, dtype.F32}, {dtype.BF16, dtype.F32}} {
		mk := func(dev device.Device) *tensor.Tensor {
			r := rand.New(rand.NewPCG(5, 1))
			x := randTensor(t, r, []int{2, 3, 4, 128}, 2, dev, dtype.F32)
			z := randTensor(t, r, []int{2, 3, 4, 128}, 1, dev, tc.z)
			w := randTensor(t, r, []int{128}, 1, dev, dtype.F32)
			out, err := ops.GatedRMSNormSiLU(x, z, w, 1e-6, tc.out)
			if err != nil {
				t.Fatal(err)
			}
			return out
		}
		tol := 1e-5
		if tc.out == dtype.BF16 {
			tol = 1e-2
		}
		if d := maxRelDiff(hostF32(t, mk(gpu)), hostF32(t, mk(device.CPU()))); d > tol {
			t.Errorf("z %v out %v: differs by %.3g", tc.z, tc.out, d)
		}
	}
}

// RoPEPartial on CUDA (one kernel) against the CPU narrow + RoPE + concat.
func TestRoPEPartialCUDAParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	for _, dt := range []dtype.DType{dtype.F32, dtype.BF16} {
		for _, rot := range []int{32, 64, 128} {
			mk := func(dev device.Device) *tensor.Tensor {
				r := rand.New(rand.NewPCG(8, 2))
				x := randTensor(t, r, []int{2, 4, 5, 128}, 1, dev, dt)
				pos, err := tensor.New([]int64{100, 101, 102, 103, 104}, []int{5}, dev)
				if err != nil {
					t.Fatal(err)
				}
				out, err := ops.RoPEPartial(x, pos, 10000, rot)
				if err != nil {
					t.Fatal(err)
				}
				return out
			}
			tol := 1e-4 // fast-math sincos on the device
			if dt == dtype.BF16 {
				tol = 1e-2
			}
			if d := maxRelDiff(hostF32(t, mk(gpu)), hostF32(t, mk(device.CPU()))); d > tol {
				t.Errorf("%v rot %d: differs by %.3g", dt, rot, d)
			}
		}
	}
}
