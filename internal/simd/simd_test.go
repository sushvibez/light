package simd

import (
	"math"
	"math/rand/v2"
	"testing"
)

// Dot and Axpy (the AVX2 path where available) agree with float64 sums
// for every length around the 8- and 32-lane boundaries.
func TestDotAxpy(t *testing.T) {
	t.Logf("accelerated: %v", Accelerated())
	rng := rand.New(rand.NewPCG(1, 2))
	for n := 0; n < 200; n++ {
		a, b, y := make([]float32, n+3), make([]float32, n+3), make([]float32, n+3)
		for i := range a {
			a[i], b[i], y[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64()), float32(rng.NormFloat64())
		}
		var want, mag float64
		for i := range n {
			want += float64(a[i]) * float64(b[i])
			mag += math.Abs(float64(a[i]) * float64(b[i]))
		}
		if got := Dot(a[:n], b); math.Abs(float64(got)-want) > 1e-5*(1+mag) {
			t.Fatalf("Dot n=%d: %g, want %g", n, got, want)
		}
		y0 := append([]float32(nil), y...)
		Axpy(0.75, a[:n], y)
		for i := range y {
			w := y0[i]
			if i < n {
				w = float32(float64(y0[i]) + 0.75*float64(a[i]))
			}
			if math.Abs(float64(y[i]-w)) > 1e-5*(1+math.Abs(float64(w))) {
				t.Fatalf("Axpy n=%d [%d]: %g, want %g", n, i, y[i], w)
			}
		}
	}
}

func BenchmarkDot4096(b *testing.B) {
	x, y := make([]float32, 4096), make([]float32, 4096)
	for i := range x {
		x[i], y[i] = float32(i%7), float32(i%5)
	}
	b.SetBytes(2 * 4 * 4096)
	for b.Loop() {
		_ = Dot(x, y)
	}
}
