package ops_test

import (
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

func f32(t *testing.T, x *tensor.Tensor) []float32 {
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
