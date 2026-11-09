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

// F16C reports AVX2 plus the F16C half-precision conversions (VCVTPH2PS).
func F16C() bool { return hasF16C }

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

// MatTB computes out[i·ldo + j] = Σ_t a[i·lda + t] · w[j·ldw + t] for
// i < m, j < n, t < k: a block of x · Wᵀ (rows of A against rows of W).
// On AVX2 it runs 4 × 3 register tiles, loading each A and W vector once
// per 12 multiply-adds; edges and k % 8 tails use Dot.
func MatTB(a []float32, lda int, w []float32, ldw int, out []float32, ldo, m, n, k int) {
	i := 0
	if useAVX2 && k%8 == 0 && k > 0 {
		var t [12]float32
		for ; i+4 <= m; i += 4 {
			j := 0
			for ; j+3 <= n; j += 3 {
				_ = a[(i+3)*lda+k-1]
				_ = w[(j+2)*ldw+k-1]
				dot4x3AVX2(&a[i*lda], lda, &w[j*ldw], ldw, k, &t[0])
				for r := range 4 {
					o := out[(i+r)*ldo+j:]
					o[0], o[1], o[2] = t[3*r], t[3*r+1], t[3*r+2]
				}
			}
			for ; j < n; j++ {
				for r := range 4 {
					out[(i+r)*ldo+j] = Dot(a[(i+r)*lda:(i+r)*lda+k], w[j*ldw:])
				}
			}
		}
	}
	for ; i < m; i++ {
		for j := range n {
			out[i*ldo+j] = Dot(a[i*lda:i*lda+k], w[j*ldw:])
		}
	}
}
