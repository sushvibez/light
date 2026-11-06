package ops_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// RoPEInterleaved: pairs (2i, 2i+1) rotate by pos·base^(-2i/D). Check
// against a hand computation and against CUDA.
func TestRoPEInterleaved(t *testing.T) {
	const T, D = 3, 4
	data := make([]float32, 2*T*D) // (2, T, D)
	for i := range data {
		data[i] = float32(i%7) - 3
	}
	x, _ := tensor.New(data, []int{2, T, D}, device.CPU())
	pos, _ := tensor.New([]int64{0, 1, 2}, []int{T}, device.CPU())
	y, err := ops.RoPEInterleaved(x, pos, 10000)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float32, len(data))
	for b := range 2 {
		for s := range T {
			for i := range D / 2 {
				theta := float64(s) * math.Pow(10000, -float64(2*i)/D)
				c, sn := math.Cos(theta), math.Sin(theta)
				off := (b*T+s)*D + 2*i
				x0, x1 := float64(data[off]), float64(data[off+1])
				want[off] = float32(x0*c - x1*sn)
				want[off+1] = float32(x0*sn + x1*c)
			}
		}
	}
	expectRow(t, "rope_i", tensor.MustSlice[float32](y), want, 1e-5)
	// Interleaved differs from rotate-half for D=4 at pos>0.
	rh, _ := ops.RoPE(x, pos, 10000)
	same := true
	for i, v := range tensor.MustSlice[float32](rh) {
		if math.Abs(float64(v-want[i])) > 1e-4 {
			same = false
		}
	}
	if same {
		t.Error("rope_i should differ from rotate-half rope")
	}
	gpu := cudaOrSkip(t)
	for _, dt := range []dtype.DType{dtype.F32, dtype.F64, dtype.BF16} {
		xc, _ := ops.Cast(x, dt)
		yc, _ := ops.RoPEInterleaved(xc, pos, 10000)
		xg, _ := xc.To(gpu)
		pg, _ := pos.To(gpu)
		yg, err := ops.RoPEInterleaved(xg, pg, 10000)
		if err != nil {
			t.Fatalf("%v: %v", dt, err)
		}
		tol := 1e-5
		if dt == dtype.BF16 {
			tol = 3e-2
		}
		expectRow(t, "rope_i cuda "+dt.String(), f32s(t, yg), f32s(t, yc), tol)
	}
}

func TestRoPETHD(t *testing.T) {
	const B, T, H, D = 1, 3, 2, 4
	data := make([]float32, B*T*H*D)
	for i := range data {
		data[i] = float32(i%5) - 2
	}
	x, _ := tensor.New(data, []int{B, T, H, D}, device.CPU()) // (B, T, H, D)
	pos, _ := tensor.New([]int64{0, 1, 2}, []int{T}, device.CPU())
	y, err := ops.RoPETHD(x, pos, 10000)
	if err != nil {
		t.Fatal(err)
	}
	// Reference: permute to (B, H, T, D), RoPE, permute back.
	p, _ := x.Permute(0, 2, 1, 3)
	pc, _ := p.Contiguous()
	r, _ := ops.RoPE(pc, pos, 10000)
	rb, _ := r.Permute(0, 2, 1, 3)
	rc, _ := rb.Contiguous()
	expectRow(t, "rope_thd", tensor.MustSlice[float32](y), tensor.MustSlice[float32](rc), 1e-6)
	if !y.Shape().Equal(x.Shape()) {
		t.Errorf("shape %v", y.Shape())
	}
}
