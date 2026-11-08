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

// MatTB (4 × 3 AVX2 tiles plus Dot edges) agrees with float64 sums for
// shapes around the tile and k % 8 boundaries, with padded strides.
func TestMatTB(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, sh := range [][3]int{{1, 1, 8}, {4, 3, 8}, {5, 4, 16}, {9, 7, 64}, {8, 6, 13}, {13, 11, 896}, {3, 2, 40}} {
		m, n, k := sh[0], sh[1], sh[2]
		lda, ldw, ldo := k+5, k+3, n+2
		a, w := make([]float32, m*lda), make([]float32, n*ldw)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		out := make([]float32, m*ldo)
		MatTB(a, lda, w, ldw, out, ldo, m, n, k)
		for i := range m {
			for j := range n {
				var want, mag float64
				for t := range k {
					p := float64(a[i*lda+t]) * float64(w[j*ldw+t])
					want += p
					mag += math.Abs(p)
				}
				if d := math.Abs(float64(out[i*ldo+j]) - want); d > 1e-5*(1+mag) {
					t.Fatalf("%v [%d,%d] = %g, want %g", sh, i, j, out[i*ldo+j], want)
				}
			}
		}
	}
}

func BenchmarkMatTB(b *testing.B) {
	const m, n, k = 64, 96, 896
	a, w, out := make([]float32, m*k), make([]float32, n*k), make([]float32, m*n)
	for i := range a {
		a[i] = float32(i % 7)
	}
	for i := range w {
		w[i] = float32(i % 5)
	}
	b.Run("tiled", func(b *testing.B) {
		b.SetBytes(int64(2 * m * n * k)) // FLOPs, reported as "MB/s" = MFLOP/s
		for b.Loop() {
			MatTB(a, k, w, k, out, n, m, n, k)
		}
	})
	b.Run("dots", func(b *testing.B) {
		b.SetBytes(int64(2 * m * n * k))
		for b.Loop() {
			for i := range m {
				for j := range n {
					out[i*n+j] = Dot(a[i*k:i*k+k], w[j*k:j*k+k])
				}
			}
		}
	})
}
