package ops_test

import (
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// The in-place forms must leave the state the out-of-place forms return,
// and produce the same output, on every device with the ops.
func TestDeltaNetInPlaceState(t *testing.T) {
	devs := []device.Device{device.CPU()}
	if device.CUDAAvailable() {
		if gpu, err := device.CUDA(0); err == nil {
			devs = append(devs, gpu)
		}
	}
	rng := rand.New(rand.NewPCG(2, 4))
	B, Hk, dk, Hv, dv, K := 2, 2, 32, 4, 32, 4
	C := 2*Hk*dk + Hv*dv
	for _, dev := range devs {
		qkvConv := randTensor(t, rng, []int{B, C, 1}, 1, device.CPU(), dtype.F32) // (B, C, T=1) for the conv
		w := randTensor(t, rng, []int{C, K}, 0.5, device.CPU(), dtype.F32)
		conv0 := randTensor(t, rng, []int{B, C, K - 1}, 1, device.CPU(), dtype.F32)
		a := randTensor(t, rng, []int{B, 1, Hv}, 1, device.CPU(), dtype.F32)
		b := randTensor(t, rng, []int{B, 1, Hv}, 1, device.CPU(), dtype.F32)
		dtBias := randTensor(t, rng, []int{Hv}, 0.1, device.CPU(), dtype.F32)
		negA := randTensor(t, rng, []int{Hv}, 0.1, device.CPU(), dtype.F32)
		rec0 := randTensor(t, rng, []int{B, Hv, dk, dv}, 0.3, device.CPU(), dtype.F32)
		to := func(x *tensor.Tensor) *tensor.Tensor {
			y, err := x.To(dev)
			if err != nil {
				t.Fatal(err)
			}
			c, _ := y.Contiguous()
			return c
		}
		// Reference: out-of-place.
		xc, wc, sc := to(qkvConv), to(w), to(conv0)
		outRef, convRef, err := ops.CausalConv1dDepthwise(xc, wc, sc, true)
		if err != nil {
			t.Fatal(err)
		}
		// In place on a fresh copy of the state.
		s2 := to(conv0)
		outIn, err := ops.CausalConv1dDepthwiseInPlace(xc, wc, s2, true)
		if err != nil {
			t.Fatal(err)
		}
		same(t, dev, "conv out", outIn, outRef)
		same(t, dev, "conv state", s2, convRef)

		// Delta rule over (B, T=1, C) with the conv output.
		qkv, _ := outRef.Transpose(1, 2)
		qkv, _ = qkv.Contiguous()
		opts := ops.GatedDeltaOptions{Mode: ops.GatedDeltaRecurrent, QKL2Norm: true}
		r1 := to(rec0)
		oRef, fRef, err := ops.GatedDeltaRuleFused(qkv, to(a), to(b), to(dtBias), to(negA), r1, Hk, dk, Hv, dv, false, opts)
		if err != nil {
			t.Fatal(err)
		}
		r2 := to(rec0)
		opts.InPlace = true
		oIn, fIn, err := ops.GatedDeltaRuleFused(qkv, to(a), to(b), to(dtBias), to(negA), r2, Hk, dk, Hv, dv, false, opts)
		if err != nil {
			t.Fatal(err)
		}
		if fIn != r2 {
			t.Fatalf("%v: InPlace returned a new state tensor", dev)
		}
		same(t, dev, "delta out", oIn, oRef)
		same(t, dev, "delta state", r2, fRef)
	}
}

func same(t *testing.T, dev device.Device, what string, got, want *tensor.Tensor) {
	t.Helper()
	g, w := f32(t, got), f32(t, want)
	if len(g) != len(w) {
		t.Fatalf("%v %s: %d vs %d elements", dev, what, len(g), len(w))
	}
	for i := range g {
		if g[i] != w[i] {
			t.Fatalf("%v %s: [%d] = %v, want %v", dev, what, i, g[i], w[i])
		}
	}
}
