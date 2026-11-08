//go:build !amd64

package simd

const useAVX2 = false

func dotAVX2(a, b *float32, n int) float32         { panic("unreachable") }
func axpyAVX2(alpha float32, x, y *float32, n int) { panic("unreachable") }
func dot4x3AVX2(a *float32, lda int, w *float32, ldw int, k int, out *float32) {
	panic("unreachable")
}
