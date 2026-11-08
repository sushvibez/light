package cpu

import (
	"light/internal/parallel"
	"light/internal/simd"
)

// matmulF32Fast handles the F32 layouts models produce, in parallel over
// output columns with SIMD inner loops; it reports false for any other
// stride pattern (the caller then runs the generic loop).
//
//   - b is a transposed row-major weight (b[k, j] at k + j·K: nn.Linear's
//     x · Wᵀ): every output is a dot product of an A row with a W row.
//   - b is row-major (b[k, j] at k·N + j): output rows accumulate
//     a[i, k] · b[k, :] (Axpy).
//
// A's rows must be contiguous (stride 1 along K).
func matmulF32Fast(aS []float32, aOff int, aStr []int, bS []float32, bOff int, bStr []int, out []float32, M, K, N int) bool {
	if M == 0 || N == 0 {
		return true
	}
	if aStr[1] != 1 && K > 1 {
		return false
	}
	aRow := func(i int) []float32 { return aS[aOff+i*aStr[0]:][:K] }
	switch {
	case (bStr[0] == 1 || K == 1) && (bStr[1] == K || N == 1):
		// out[i, j] = a_i · w_j with w_j = b[:, j], contiguous.
		grain := max(1, 32768/(K*M+1))
		parallel.For(N, grain, func(lo, hi int) {
			for j := lo; j < hi; j++ {
				w := bS[bOff+j*bStr[1]:][:K]
				for i := range M {
					out[i*N+j] = simd.Dot(aRow(i), w)
				}
			}
		})
		return true
	case bStr[1] == 1 || N == 1:
		// out[i, j0:j1] = Σ_k a[i, k] · b[k, j0:j1]: column blocks in parallel.
		const block = 256
		blocks := (N + block - 1) / block
		parallel.For(blocks*M, max(1, 65536/(K*block+1)), func(lo, hi int) {
			for t := lo; t < hi; t++ {
				i, j0 := t/blocks, (t%blocks)*block
				j1 := min(j0+block, N)
				o := out[i*N+j0 : i*N+j1]
				clear(o)
				a := aRow(i)
				for k, av := range a {
					if av != 0 {
						simd.Axpy(av, bS[bOff+k*bStr[0]+j0:][:j1-j0], o)
					}
				}
			}
		})
		return true
	}
	return false
}
