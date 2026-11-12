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

// FlashAttentionBF16 against an F64 softmax(q·kᵀ/√D)·v on the same
// BF16-rounded inputs: lengths off the 64 / 128 tiles, causal or not. The
// kernel rounds the probabilities to BF16 for the P·V product, so it is
// held to BF16-level error relative to the output's scale.
func TestFlashAttentionBF16MatchesReference(t *testing.T) {
	gpu := gpuOrSkip(t)
	rng := rand.New(rand.NewPCG(7, 8))
	const D = 128
	for _, c := range []struct {
		B, NH, S int
		causal   bool
	}{{1, 2, 64, false}, {2, 3, 200, false}, {1, 2, 333, true}, {1, 1, 1, false}, {1, 4, 1031, false}} {
		n := c.B * c.NH * c.S * D
		mk := func(scale float32) ([]float64, *tensor.Tensor) {
			x := randTensor(t, rng, []int{c.B, c.NH, c.S, D}, scale, device.CPU(), dtype.BF16)
			ref := f32(t, x) // the BF16 values, exactly
			r := make([]float64, n)
			for i, v := range ref {
				r[i] = float64(v)
			}
			g, err := x.To(gpu)
			if err != nil {
				t.Fatal(err)
			}
			return r, g
		}
		qr, q := mk(2) // scores spread wide enough to exercise the running max
		kr, k := mk(1)
		vr, v := mk(1)
		out, err := ops.FlashAttentionBF16(q, k, v, c.causal)
		if err != nil {
			t.Fatal(err)
		}
		got := f32(t, out)
		scale := 1 / math.Sqrt(D)
		maxErr, maxRef := 0.0, 0.0
		for bh := 0; bh < c.B*c.NH; bh++ {
			base := bh * c.S * D
			for i := 0; i < c.S; i++ {
				jEnd := c.S
				if c.causal {
					jEnd = i + 1
				}
				s := make([]float64, jEnd)
				m := math.Inf(-1)
				for j := range jEnd {
					dot := 0.0
					for d := range D {
						dot += qr[base+i*D+d] * kr[base+j*D+d]
					}
					s[j] = dot * scale
					m = math.Max(m, s[j])
				}
				l := 0.0
				for j := range s {
					s[j] = math.Exp(s[j] - m)
					l += s[j]
				}
				for d := range D {
					o := 0.0
					for j := range s {
						o += s[j] * vr[base+j*D+d]
					}
					o /= l
					maxRef = math.Max(maxRef, math.Abs(o))
					maxErr = math.Max(maxErr, math.Abs(o-float64(got[base+i*D+d])))
				}
			}
		}
		t.Logf("B=%d NH=%d S=%d causal=%v: max |err| %.3g (max |out| %.3g)", c.B, c.NH, c.S, c.causal, maxErr, maxRef)
		if maxErr > 1e-2*math.Max(maxRef, 1e-3) {
			t.Errorf("S=%d causal=%v: max error %.3g", c.S, c.causal, maxErr)
		}
	}
}

// BenchmarkFlashAttentionBF16 is one Z-Image 1024² attention: 30 heads over
// 4160 tokens (and the F32 kernel it replaces, for comparison).
func BenchmarkFlashAttentionBF16(b *testing.B) {
	gpu := gpuOrSkip(b)
	rng := rand.New(rand.NewPCG(1, 1))
	dims := []int{1, 30, 4160, 128}
	for _, dt := range []dtype.DType{dtype.BF16, dtype.F32} {
		var x [3]*tensor.Tensor
		for i := range x {
			c := randTensorB(b, rng, dims, dt)
			g, err := c.To(gpu)
			if err != nil {
				b.Fatal(err)
			}
			x[i] = g
		}
		b.Run(dt.String(), func(b *testing.B) {
			for b.Loop() {
				var o *tensor.Tensor
				var err error
				if dt == dtype.BF16 {
					o, err = ops.FlashAttentionBF16(x[0], x[1], x[2], false)
				} else {
					o, err = ops.FlashAttention(x[0], x[1], x[2], false)
				}
				if err != nil {
					b.Fatal(err)
				}
				syncGPU(b, o)
			}
			flops := 4.0 * 30 * 4160 * 4160 * 128
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOPS")
		})
	}
}

func randTensorB(b *testing.B, rng *rand.Rand, dims []int, dt dtype.DType) *tensor.Tensor {
	n := 1
	for _, d := range dims {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(rng.NormFloat64())
	}
	x, err := tensor.New(data, dims, device.CPU())
	if err != nil {
		b.Fatal(err)
	}
	if dt != dtype.F32 {
		if x, err = ops.Cast(x, dt); err != nil {
			b.Fatal(err)
		}
	}
	return x
}

// syncGPU waits for x by reading one element of it back.
func syncGPU(b *testing.B, x *tensor.Tensor) {
	one := x
	for d := range x.Rank() {
		var err error
		if one, err = one.Narrow(d, 0, 1); err != nil {
			b.Fatal(err)
		}
	}
	if _, err := one.To(device.CPU()); err != nil {
		b.Fatal(err)
	}
}
