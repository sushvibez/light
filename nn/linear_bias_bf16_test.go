package nn

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// A BF16 Linear whose product nearly cancels a large bias (the Qwen2 /
// MiMo k_proj situation): x·Wᵀ ≈ −49.9 against b = 50. Rounding x·Wᵀ to
// BF16 first (steps of 0.25 near 50) gives 0.0; one rounding of the F32 sum
// gives ≈ 0.1, as transformers' cuBLAS bias epilogue does.
func TestLinearBF16BiasSingleRounding(t *testing.T) {
	devs := []device.Device{device.CPU()}
	if device.CUDAAvailable() {
		if g, err := device.CUDA(0); err == nil {
			devs = append(devs, g)
		}
	}
	for _, dev := range devs {
		w32, _ := tensor.New([]float32{-12.5, -12.5, -12.5, -12.4}, []int{1, 4}, dev)
		b32, _ := tensor.New([]float32{50}, []int{1}, dev)
		x32, _ := tensor.New([]float32{1, 1, 1, 1}, []int{1, 4}, dev)
		w, _ := ops.Cast(w32, dtype.BF16)
		b, _ := ops.Cast(b32, dtype.BF16)
		x, _ := ops.Cast(x32, dtype.BF16)
		// The exact result from the BF16 values themselves.
		wc, _ := ops.Cast(w, dtype.F32)
		wc, _ = wc.To(device.CPU())
		want := 50.0
		for _, v := range tensor.MustSlice[float32](wc) {
			want += float64(v)
		}
		l := &Linear{W: w, B: b, inFeatures: 4, outFeatures: 1}
		y, err := l.Forward(x)
		if err != nil {
			t.Fatal(err)
		}
		if y.DType() != dtype.BF16 {
			t.Fatalf("%v: output %v, want BF16", dev, y.DType())
		}
		y32, _ := ops.Cast(y, dtype.F32)
		y32, _ = y32.To(device.CPU())
		got := float64(tensor.MustSlice[float32](y32)[0])
		if math.Abs(got-want) > 2e-3 {
			t.Errorf("%v: x·Wᵀ + b = %g, want %g (one rounding)", dev, got, want)
		}
	}
}
