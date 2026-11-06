package ops_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// sdpaRef runs nn.SDPA on the same device as q with an optional causal mask.
func sdpaRef(t *testing.T, q, k, v *tensor.Tensor, causal bool) *tensor.Tensor {
	t.Helper()
	var mask *tensor.Tensor
	if causal {
		var err error
		if mask, err = nn.CausalMask(q.Dim(2), q.Device()); err != nil {
			t.Fatal(err)
		}
	}
	out, err := nn.SDPA(q, k, v, mask)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func maxAbsDiff(t *testing.T, a, b *tensor.Tensor) float64 {
	t.Helper()
	ac, err := a.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	bc, err := b.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	as := tensor.MustSlice[float32](ac)
	bs := tensor.MustSlice[float32](bc)
	if len(as) != len(bs) {
		t.Fatalf("length mismatch %d vs %d", len(as), len(bs))
	}
	var m float64
	for i := range as {
		if d := math.Abs(float64(as[i] - bs[i])); d > m {
			m = d
		}
	}
	return m
}

func TestFlashAttentionCPUMatchesSDPA(t *testing.T) {
	cpu := device.CPU()
	const B, NH = 2, 3
	for _, S := range []int{1, 7, 64, 257} {
		for _, D := range []int{32, 64, 128} {
			if testing.Short() && S == 257 && D == 128 {
				continue
			}
			q, _ := tensor.RandN(dtype.F32, []int{B, NH, S, D}, cpu)
			k, _ := tensor.RandN(dtype.F32, []int{B, NH, S, D}, cpu)
			v, _ := tensor.RandN(dtype.F32, []int{B, NH, S, D}, cpu)
			for _, causal := range []bool{false, true} {
				got, err := ops.FlashAttention(q, k, v, causal)
				if err != nil {
					t.Fatalf("S=%d D=%d causal=%v: %v", S, D, causal, err)
				}
				if got.Rank() != 4 || got.Dim(0) != B || got.Dim(1) != NH || got.Dim(2) != S || got.Dim(3) != D {
					t.Fatalf("shape %v", got.Shape())
				}
				want := sdpaRef(t, q, k, v, causal)
				if d := maxAbsDiff(t, got, want); d > 1e-4 {
					t.Errorf("S=%d D=%d causal=%v: max |diff| = %g", S, D, causal, d)
				}
			}
		}
	}
}

func TestFlashAttentionF64AndErrors(t *testing.T) {
	cpu := device.CPU()
	q, _ := tensor.RandN(dtype.F64, []int{1, 2, 5, 8}, cpu)
	k, _ := tensor.RandN(dtype.F64, []int{1, 2, 5, 8}, cpu)
	v, _ := tensor.RandN(dtype.F64, []int{1, 2, 5, 8}, cpu)
	out, err := ops.FlashAttention(q, k, v, true)
	if err != nil {
		t.Fatal(err)
	}
	// Row 0 of each head attends only to key 0 → equals v[.., 0, :].
	o := tensor.MustSlice[float64](out)
	vs := tensor.MustSlice[float64](v)
	for h := range 2 {
		for d := range 8 {
			if math.Abs(o[(h*5)*8+d]-vs[(h*5)*8+d]) > 1e-12 {
				t.Errorf("causal row0 head %d d %d: %v vs %v", h, d, o[(h*5)*8+d], vs[(h*5)*8+d])
			}
		}
	}
	bad, _ := tensor.RandN(dtype.F64, []int{1, 2, 6, 8}, cpu)
	if _, err := ops.FlashAttention(q, bad, v, false); err == nil {
		t.Error("expected shape-mismatch error")
	}
	r3, _ := tensor.RandN(dtype.F64, []int{2, 5, 8}, cpu)
	if _, err := ops.FlashAttention(r3, r3, r3, false); err == nil {
		t.Error("expected rank error")
	}
}
