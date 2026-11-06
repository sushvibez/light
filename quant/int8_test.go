package quant_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/quant"
	"light/tensor"
)

func TestQuantizeInt8PerColumn(t *testing.T) {
	cpu := device.CPU()
	// Column 0: max |w| = 254 → scale 2, codes ±127, 0.
	// Column 1: max |w| = 1.27 → scale 0.01.
	// Column 2: all zero → scale 1, codes 0.
	w, _ := tensor.New([]float32{
		254, 1.27, 0,
		-254, -0.64, 0,
		0, 0.006, 0,
	}, []int{3, 3}, cpu)
	q, sc, err := quant.QuantizeInt8PerColumn(w)
	if err != nil {
		t.Fatal(err)
	}
	if q.DType() != dtype.I8 || sc.DType() != dtype.F32 {
		t.Fatalf("dtypes %v %v", q.DType(), sc.DType())
	}
	if q.Dim(0) != 3 || q.Dim(1) != 3 || sc.Dim(0) != 3 {
		t.Fatalf("shapes %v %v", q.Shape(), sc.Shape())
	}
	scales := tensor.MustSlice[float32](sc)
	for i, want := range []float32{2, 0.01, 1} {
		if math.Abs(float64(scales[i]-want)) > 1e-6 {
			t.Errorf("scale[%d] = %v want %v", i, scales[i], want)
		}
	}
	codes := tensor.MustSlice[int8](q)
	wantCodes := []int8{127, 127, 0, -127, -64, 0, 0, 1, 0} // 0.006/0.01 = 0.6 → 1
	for i := range wantCodes {
		if codes[i] != wantCodes[i] {
			t.Errorf("code[%d] = %d want %d", i, codes[i], wantCodes[i])
		}
	}
	// Dequantised values are within half a step of the original.
	wd, err := quant.DequantizeInt8PerColumn(q, sc)
	if err != nil {
		t.Fatal(err)
	}
	ws := tensor.MustSlice[float32](w)
	ds := tensor.MustSlice[float32](wd)
	for i := range ws {
		col := i % 3
		if math.Abs(float64(ws[i]-ds[i])) > float64(scales[col])/2+1e-6 {
			t.Errorf("[%d] dequant %v vs %v (scale %v)", i, ds[i], ws[i], scales[col])
		}
	}
	// Random matrix: relative reconstruction error is small, and the code
	// range is fully used (some column hits ±127).
	r, _ := tensor.RandN(dtype.F32, []int{64, 32}, cpu)
	rq, rs, _ := quant.QuantizeInt8PerColumn(r)
	rd, _ := quant.DequantizeInt8PerColumn(rq, rs)
	rsrc, rdq := tensor.MustSlice[float32](r), tensor.MustSlice[float32](rd)
	var num, den float64
	for i := range rsrc {
		d := float64(rsrc[i] - rdq[i])
		num += d * d
		den += float64(rsrc[i]) * float64(rsrc[i])
	}
	if rel := math.Sqrt(num / den); rel > 0.01 {
		t.Errorf("random reconstruction rel err %g", rel)
	}
	hit := false
	for _, c := range tensor.MustSlice[int8](rq) {
		if c == 127 || c == -127 {
			hit = true
			break
		}
	}
	if !hit {
		t.Error("expected at least one saturated code")
	}
	// Errors.
	bad, _ := tensor.Zeros(dtype.F32, []int{2, 2, 2}, cpu)
	if _, _, err := quant.QuantizeInt8PerColumn(bad); err == nil {
		t.Error("expected rank error")
	}
	bad64, _ := tensor.Zeros(dtype.F64, []int{2, 2}, cpu)
	if _, _, err := quant.QuantizeInt8PerColumn(bad64); err == nil {
		t.Error("expected dtype error")
	}
	// Non-contiguous input (transposed view) is handled.
	wt, _ := w.Transpose(0, 1)
	qt, st, err := quant.QuantizeInt8PerColumn(wt)
	if err != nil {
		t.Fatal(err)
	}
	// Row 0 of w becomes column 0 of wt: max |.| = 254 → scale 2.
	if s := tensor.MustSlice[float32](st)[0]; math.Abs(float64(s-2)) > 1e-6 {
		t.Errorf("transposed scale[0] = %v", s)
	}
	if c := tensor.MustSlice[int8](qt)[0]; c != 127 {
		t.Errorf("transposed code[0] = %d", c)
	}
}
