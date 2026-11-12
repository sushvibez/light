//go:build !nocuda && cgo

package ops_test

import (
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// The single-block CUDA top-k and the block-parallel argmax must agree with
// the CPU kernels on logits-sized rows, including ties (lowest index wins).
func TestTopKArgmaxCUDAParity(t *testing.T) {
	gpu := gpuOrSkip(t)
	rng := rand.New(rand.NewPCG(7, 11))
	for _, tc := range []struct{ rows, cols, k int }{{1, 250_000, 40}, {3, 128_256, 64}, {2, 1000, 1}, {1, 50, 50}, {4, 777, 8}} {
		data := make([]float32, tc.rows*tc.cols)
		for i := range data {
			data[i] = float32(rng.NormFloat64())
			if i%97 == 0 { // planted ties
				data[i] = 3.5
			}
		}
		cpu, err := tensor.New(data, []int{tc.rows, tc.cols}, device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		dev, err := cpu.To(gpu)
		if err != nil {
			t.Fatal(err)
		}
		for _, largest := range []bool{true, false} {
			wv, wi, err := ops.TopK(cpu, tc.k, largest)
			if err != nil {
				t.Fatal(err)
			}
			gv, gi, err := ops.TopK(dev, tc.k, largest)
			if err != nil {
				t.Fatal(err)
			}
			gvC, _ := gv.To(device.CPU())
			giC, _ := gi.To(device.CPU())
			wvs, gvs := tensor.MustSlice[float32](wv), tensor.MustSlice[float32](gvC)
			wis, gis := tensor.MustSlice[int64](wi), tensor.MustSlice[int64](giC)
			for i := range wvs {
				if wvs[i] != gvs[i] || wis[i] != gis[i] {
					t.Fatalf("%dx%d k=%d largest=%v: entry %d cpu (%g, %d) cuda (%g, %d)", tc.rows, tc.cols, tc.k, largest, i, wvs[i], wis[i], gvs[i], gis[i])
				}
			}
		}
		wa, err := ops.Argmax(cpu, -1, false)
		if err != nil {
			t.Fatal(err)
		}
		ga, err := ops.Argmax(dev, -1, false)
		if err != nil {
			t.Fatal(err)
		}
		gaC, _ := ga.To(device.CPU())
		if ga.DType() != dtype.I64 {
			t.Fatalf("argmax dtype %v", ga.DType())
		}
		was, gas := tensor.MustSlice[int64](wa), tensor.MustSlice[int64](gaC)
		for i := range was {
			if was[i] != gas[i] {
				t.Fatalf("%dx%d argmax row %d: cpu %d cuda %d", tc.rows, tc.cols, i, was[i], gas[i])
			}
		}
	}
}
