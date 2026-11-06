package quant_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/quant"
	"light/tensor"
)

func TestQuantizeRoundTrip(t *testing.T) {
	n := 32 * 24
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(math.Sin(0.37*float64(i))) * float32(1+i%7)
	}
	x, _ := tensor.New(xs, []int{24, 32}, device.CPU())
	for _, tc := range []struct {
		dt  dtype.DType
		tol float64
	}{{dtype.Q8_0, 0.01}, {dtype.Q5_0, 0.05}, {dtype.Q4_0, 0.12}} {
		q, err := quant.Quantize(x, tc.dt)
		if err != nil {
			t.Fatal(err)
		}
		if q.DType() != tc.dt || q.Dim(0) != 24 || q.Dim(1) != 32 {
			t.Fatalf("%v: got %v %v", tc.dt, q.DType(), q.Shape())
		}
		back, err := quant.Dequantize(q)
		if err != nil {
			t.Fatal(err)
		}
		bs := tensor.MustSlice[float32](back)
		var num, den float64
		for i := range xs {
			num += float64((bs[i] - xs[i]) * (bs[i] - xs[i]))
			den += float64(xs[i] * xs[i])
		}
		if e := math.Sqrt(num / den); e > tc.tol {
			t.Errorf("%v round trip rel err %.4g > %g", tc.dt, e, tc.tol)
		}
	}
}
