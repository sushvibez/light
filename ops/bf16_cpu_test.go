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

// The CPU BF16 kernels for rmsnorm, rope and softmax compute in F32 and
// round each output to BF16 once, so they must equal the F32 kernel run on
// the same (BF16-representable) inputs, rounded to BF16.

func bf16Pair(t *testing.T, dims []int, seed uint64) (f32, bf16 *tensor.Tensor) {
	t.Helper()
	n := 1
	for _, d := range dims {
		n *= d
	}
	r := rand.New(rand.NewPCG(seed, 7))
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = dtype.BF16FromF32(float32(r.NormFloat64())).ToF32() // exactly representable in BF16
	}
	f32, err := tensor.New(vals, dims, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if bf16, err = ops.Cast(f32, dtype.BF16); err != nil {
		t.Fatal(err)
	}
	return f32, bf16
}

// sameAsRoundedF32 checks got (BF16) == round_bf16(want (F32)) element-wise.
func sameAsRoundedF32(t *testing.T, name string, got, want *tensor.Tensor) {
	t.Helper()
	g, err := ops.Cast(got, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	gs, ws := tensor.MustSlice[float32](g), tensor.MustSlice[float32](want)
	bad := 0
	for i := range ws {
		if gs[i] != dtype.BF16FromF32(ws[i]).ToF32() {
			bad++
			if bad <= 3 {
				t.Errorf("%s[%d] = %v, want round_bf16(%v) = %v", name, i, gs[i], ws[i], dtype.BF16FromF32(ws[i]).ToF32())
			}
		}
	}
	if bad > 0 {
		t.Errorf("%s: %d of %d elements differ", name, bad, len(ws))
	}
}

func TestCPUBF16RMSNorm(t *testing.T) {
	xf, xb := bf16Pair(t, []int{3, 5, 16}, 1)
	wf, wb := bf16Pair(t, []int{16}, 2)
	want, err := ops.RMSNorm(xf, wf, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ops.RMSNorm(xb, wb, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	// F32 kernel accumulates in float32, BF16 in float64: allow the rare
	// one-ulp tie difference by comparing with a BF16-step tolerance.
	g, _ := ops.Cast(got, dtype.F32)
	gs, ws := tensor.MustSlice[float32](g), tensor.MustSlice[float32](want)
	for i := range ws {
		if d := math.Abs(float64(gs[i] - ws[i])); d > math.Abs(float64(ws[i]))/128+1e-6 {
			t.Fatalf("rmsnorm[%d] = %v, F32 %v", i, gs[i], ws[i])
		}
	}
}

func TestCPUBF16RoPE(t *testing.T) {
	xf, xb := bf16Pair(t, []int{2, 3, 7, 8}, 3)
	pos, _ := tensor.New([]int64{0, 1, 2, 5, 9, 10, 40}, []int{7}, device.CPU())
	want, err := ops.RoPE(xf, pos, 10000)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ops.RoPE(xb, pos, 10000)
	if err != nil {
		t.Fatal(err)
	}
	sameAsRoundedF32(t, "rope", got, want)
}

func TestCPUBF16Softmax(t *testing.T) {
	xf, xb := bf16Pair(t, []int{4, 6, 9}, 4)
	for _, axis := range []int{-1, 1} {
		want, err := ops.Softmax(xf, axis)
		if err != nil {
			t.Fatal(err)
		}
		got, err := ops.Softmax(xb, axis)
		if err != nil {
			t.Fatal(err)
		}
		sameAsRoundedF32(t, "softmax", got, want)
	}
	// A strided (transposed) BF16 input goes through the widening copy.
	xtf, _ := xf.Transpose(0, 2)
	xtb, _ := xb.Transpose(0, 2)
	want, err := ops.Softmax(xtf, -1)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ops.Softmax(xtb, -1)
	if err != nil {
		t.Fatal(err)
	}
	sameAsRoundedF32(t, "softmax transposed", got, want)
}
