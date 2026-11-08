// Package simd holds the CPU vector kernels the hot loops use: Dot (Σ a·b)
// and Axpy (y += α·x) on float32. On amd64 with AVX2 and FMA they run Go
// assembly (8 lanes, 4 independent accumulators); elsewhere, or when the
// CPU lacks them, a pure Go loop. No cgo.
package simd

// Dot returns Σ a[i]·b[i] over len(a) elements (b must be at least as long).
func Dot(a, b []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	_ = b[len(a)-1]
	if useAVX2 {
		return dotAVX2(&a[0], &b[0], len(a))
	}
	return dotGo(a, b)
}

// Axpy sets y[i] += alpha·x[i] over len(x) elements (y at least as long).
func Axpy(alpha float32, x, y []float32) {
	if len(x) == 0 {
		return
	}
	_ = y[len(x)-1]
	if useAVX2 {
		axpyAVX2(alpha, &x[0], &y[0], len(x))
		return
	}
	axpyGo(alpha, x, y)
}

// Accelerated reports whether the vector path is in use.
func Accelerated() bool { return useAVX2 }

func dotGo(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func axpyGo(alpha float32, x, y []float32) {
	for i, v := range x {
		y[i] += alpha * v
	}
}
