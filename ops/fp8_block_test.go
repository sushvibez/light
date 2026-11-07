package ops_test

import (
	"math/rand"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// DequantizeF8Block against a per-element reference on a 3 × 4 grid of
// 4 × 8 blocks (every block has its own scale), F32 and BF16 out, CPU and
// CUDA.
func TestDequantizeF8Block(t *testing.T) {
	const R, C, br, bc = 12, 32, 4, 8
	rng := rand.New(rand.NewSource(1))
	codes := make([]float32, R*C)
	for i := range codes {
		codes[i] = dtype.F8FromF32(float32(rng.NormFloat64() * 50)).ToF32()
	}
	scales := make([]float32, (R/br)*(C/bc))
	for i := range scales {
		scales[i] = float32(rng.Float64()*0.01 + 1e-4)
	}
	want := make([]float32, R*C)
	for i := range R {
		for j := range C {
			want[i*C+j] = codes[i*C+j] * scales[(i/br)*(C/bc)+j/bc]
		}
	}
	devs := []device.Device{device.CPU()}
	if device.CUDAAvailable() {
		if g, err := device.CUDA(0); err == nil {
			devs = append(devs, g)
		}
	}
	for _, dev := range devs {
		x, _ := tensor.New(codes, []int{R, C}, dev)
		w, err := ops.Cast(x, dtype.F8E4M3)
		if err != nil {
			t.Fatal(err)
		}
		s, _ := tensor.New(scales, []int{R / br, C / bc}, dev)
		for _, dt := range []dtype.DType{dtype.F32, dtype.BF16} {
			out, err := ops.DequantizeF8Block(w, s, br, bc, dt)
			if err != nil {
				t.Fatal(err)
			}
			if out.DType() != dt || out.Dim(0) != R || out.Dim(1) != C {
				t.Fatalf("%v %v: got %v %v", dev, dt, out.DType(), out.Shape())
			}
			o32, _ := ops.Cast(out, dtype.F32)
			o32, _ = o32.To(device.CPU())
			got := tensor.MustSlice[float32](o32)
			for i := range want {
				ref := want[i]
				if dt == dtype.BF16 {
					ref = dtype.BF16FromF32(ref).ToF32()
				}
				if got[i] != ref {
					t.Fatalf("%v %v: [%d, %d] = %g, want %g", dev, dt, i/C, i%C, got[i], ref)
				}
			}
		}
		if _, err := ops.DequantizeF8Block(w, s, 5, bc, dtype.F32); err == nil {
			t.Errorf("%v: block 5 on 12 rows accepted", dev)
		}
	}
}
