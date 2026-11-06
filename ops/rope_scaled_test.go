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

func baseInvFreq(headDim int, base float64) []float32 {
	out := make([]float32, headDim/2)
	for i := range out {
		out[i] = float32(math.Pow(base, -float64(2*i)/float64(headDim)))
	}
	return out
}

// With the plain base^(-2i/d) table, RoPEWithInvFreq must equal RoPE.
func TestRoPEWithInvFreqMatchesRoPE(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	const B, S, D = 3, 5, 8
	data := make([]float32, B*S*D)
	for i := range data {
		data[i] = rng.Float32()*2 - 1
	}
	x, _ := tensor.New(data, []int{B, S, D}, device.CPU())
	pos, _ := tensor.New([]int64{0, 1, 2, 3, 4}, []int{S}, device.CPU())
	inv, _ := tensor.New(baseInvFreq(D, 10000), []int{D / 2}, device.CPU())

	want, err := ops.RoPE(x, pos, 10000)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ops.RoPEWithInvFreq(x, pos, inv)
	if err != nil {
		t.Fatal(err)
	}
	w := tensor.MustSlice[float32](want)
	g := tensor.MustSlice[float32](got)
	for i := range w {
		if math.Abs(float64(w[i]-g[i])) > 1e-5 {
			t.Fatalf("[%d] = %v, want %v", i, g[i], w[i])
		}
	}
}

// A scaled table changes only the pairs whose frequency changed.
func TestRoPEWithInvFreqScaling(t *testing.T) {
	const S, D = 2, 4
	x, _ := tensor.New([]float32{1, 0, 1, 0, 1, 0, 1, 0}, []int{S, D}, device.CPU())
	pos, _ := tensor.New([]int64{0, 1}, []int{S}, device.CPU())
	inv, _ := tensor.New([]float32{1, 0.5}, []int{2}, device.CPU())
	out, err := ops.RoPEWithInvFreq(x, pos, inv)
	if err != nil {
		t.Fatal(err)
	}
	v := tensor.MustSlice[float32](out)
	// Rotate-half pairing: pair 0 = dims (0, 2), rotated by 1 rad at pos 1;
	// pair 1 = dims (1, 3), rotated by 0.5 rad but both inputs are 0.
	check := func(i int, want float64) {
		if math.Abs(float64(v[i])-want) > 1e-6 {
			t.Errorf("[%d] = %v, want %v", i, v[i], want)
		}
	}
	check(4, math.Cos(1)-math.Sin(1))
	check(6, math.Cos(1)+math.Sin(1))
	check(5, 0)
	check(7, 0)
}

func TestRoPEWithInvFreqErrors(t *testing.T) {
	x, _ := tensor.Zeros(dtype.F32, []int{2, 4}, device.CPU())
	pos, _ := tensor.New([]int64{0, 1}, []int{2}, device.CPU())
	bad, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	if _, err := ops.RoPEWithInvFreq(x, pos, bad); err == nil {
		t.Error("expected invFreq length error")
	}
}
