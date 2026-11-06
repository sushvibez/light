package ops_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/quant"
	"light/tensor"
)

// relErrNorm = ||a - b||₂ / ||b||₂ over CPU copies.
func relErrNorm(t *testing.T, a, b *tensor.Tensor) float64 {
	t.Helper()
	ac, _ := a.To(device.CPU())
	bc, _ := b.To(device.CPU())
	as := tensor.MustSlice[float32](ac)
	bs := tensor.MustSlice[float32](bc)
	var num, den float64
	for i := range as {
		d := float64(as[i] - bs[i])
		num += d * d
		den += float64(bs[i]) * float64(bs[i])
	}
	return math.Sqrt(num) / math.Sqrt(den)
}

func TestMatMulInt8CPU(t *testing.T) {
	cpu := device.CPU()
	const M, K, N = 5, 64, 48
	x, _ := tensor.RandN(dtype.F32, []int{M, K}, cpu)
	w, _ := tensor.RandN(dtype.F32, []int{K, N}, cpu)
	wq, scales, err := quant.QuantizeInt8PerColumn(w)
	if err != nil {
		t.Fatal(err)
	}
	if wq.DType() != dtype.I8 || wq.Dim(0) != K || wq.Dim(1) != N || scales.Dim(0) != N {
		t.Fatalf("quantized shapes/dtypes off: %v %v %v", wq.DType(), wq.Shape(), scales.Shape())
	}
	out, err := ops.MatMulInt8(x, wq, scales)
	if err != nil {
		t.Fatal(err)
	}
	if out.Rank() != 2 || out.Dim(0) != M || out.Dim(1) != N || out.DType() != dtype.F32 {
		t.Fatalf("out shape %v dtype %v", out.Shape(), out.DType())
	}
	// Exact (up to fp order) against MatMul with the dequantised weight.
	wd, err := quant.DequantizeInt8PerColumn(wq, scales)
	if err != nil {
		t.Fatal(err)
	}
	exact, _ := ops.MatMul(x, wd)
	if d := maxAbsDiff(t, out, exact); d > 1e-3 {
		t.Errorf("vs dequantised matmul: max |diff| = %g", d)
	}
	// Accuracy against the un-quantised F32 product.
	ref, _ := ops.MatMul(x, w)
	if e := relErrNorm(t, out, ref); e >= 0.01 {
		t.Errorf("rel-err-norm vs F32 matmul = %g, want < 0.01", e)
	}
	// Errors.
	if _, err := ops.MatMulInt8(x, w, scales); err == nil {
		t.Error("expected dtype error for F32 weight")
	}
	badS, _ := tensor.Zeros(dtype.F32, []int{N + 1}, cpu)
	if _, err := ops.MatMulInt8(x, wq, badS); err == nil {
		t.Error("expected scales-size error")
	}
}
