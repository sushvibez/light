package quant_test

import (
	"math"
	"math/rand"
	"testing"

	"light/device"
	"light/quant"
	"light/tensor"
)

// mlxQuantize is mlx.core.quantize (affine), as in mlx/ops.cpp: per group
// the scale spans [min, max] over 2^bits − 1 steps, signed so the larger-
// magnitude edge is exact, and the bias is that edge.
func mlxQuantize(w []float32, N, K, group, bits int) (codes []uint32, scales, biases []float32) {
	nBins := float32(int(1)<<bits - 1)
	per := 32 / bits
	G := K / group
	codes = make([]uint32, N*K/per)
	scales, biases = make([]float32, N*G), make([]float32, N*G)
	for n := range N {
		for g := range G {
			seg := w[n*K+g*group : n*K+(g+1)*group]
			lo, hi := seg[0], seg[0]
			for _, v := range seg {
				lo, hi = min(lo, v), max(hi, v)
			}
			s := max((hi-lo)/nBins, 1e-7)
			edge := hi
			if float32(math.Abs(float64(lo))) > float32(math.Abs(float64(hi))) {
				edge = lo
			} else {
				s = -s
			}
			q0 := float32(math.RoundToEven(float64(edge / s)))
			b := edge
			if q0 != 0 {
				s = edge / q0
			} else {
				b = 0
			}
			scales[n*G+g], biases[n*G+g] = s, b
			for i, v := range seg {
				q := min(max(float32(math.RoundToEven(float64((v-b)/s))), 0), nBins)
				k := g*group + i
				codes[(n*K+k)/per] |= uint32(q) << (bits * (k % per))
			}
		}
	}
	return codes, scales, biases
}

func TestMLXDequant(t *testing.T) {
	const N, K, G = 6, 128, 64
	r := rand.New(rand.NewSource(1))
	w := make([]float32, N*K)
	for i := range w {
		w[i] = float32(r.NormFloat64())
	}
	for _, bits := range []int{2, 4, 8} {
		codes, sc, bi := mlxQuantize(w, N, K, G, bits)
		ct, _ := tensor.New(codes, []int{N, K * bits / 32}, device.CPU())
		st, _ := tensor.New(sc, []int{N, K / G}, device.CPU())
		bt, _ := tensor.New(bi, []int{N, K / G}, device.CPU())
		dq, err := quant.DequantMLX(ct, st, bt, G, bits)
		if err != nil {
			t.Fatal(err)
		}
		got := tensor.MustSlice[float32](dq)
		step := 0.0
		for i := range got {
			step = max(step, math.Abs(float64(got[i]-w[i])))
		}
		// The pinned edge can push the far end past the last code: within
		// one step of the original.
		maxStep := 0.0
		for _, s := range sc {
			maxStep = max(maxStep, math.Abs(float64(s)))
		}
		if step > maxStep*1.001 {
			t.Errorf("%d-bit: max |dq − w| %g > step %g", bits, step, maxStep)
		}
		// Exact against s · q + b with q read straight from the words.
		per := 32 / bits
		for i := range got {
			n, k := i/K, i%K
			q := codes[i/per] >> (bits * (k % per)) & (1<<bits - 1)
			g := n*(K/G) + k/G
			if want := sc[g]*float32(q) + bi[g]; got[i] != want {
				t.Fatalf("%d-bit: element %d = %g, want %g", bits, i, got[i], want)
			}
		}
		if bits != 4 {
			continue
		}
		q4, err := quant.FromMLX(ct, st, bt, G)
		if err != nil {
			t.Fatal(err)
		}
		via, err := q4.Dequant()
		if err != nil {
			t.Fatal(err)
		}
		diff := 0.0
		for i := range via {
			diff = max(diff, math.Abs(float64(via[i]-got[i])))
		}
		if diff > 1e-6 {
			t.Errorf("Int4 vs MLX dequant: max diff %g", diff)
		}
		t.Logf("4-bit: max |dq − w| %.4f, Int4 vs MLX %.2g", step, diff)
	}
}
