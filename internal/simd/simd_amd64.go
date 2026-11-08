//go:build amd64

package simd

//go:noescape
func dotAVX2(a, b *float32, n int) float32

//go:noescape
func axpyAVX2(alpha float32, x, y *float32, n int)

func cpuid(leaf, sub uint32) (eax, ebx, ecx, edx uint32)
func xgetbv() (eax, edx uint32)

// useAVX2: the CPU has AVX2 and FMA, and the OS saves YMM state.
var useAVX2 = func() bool {
	max, _, _, _ := cpuid(0, 0)
	if max < 7 {
		return false
	}
	_, _, c1, _ := cpuid(1, 0)
	fma, osxsave, avx := c1&(1<<12) != 0, c1&(1<<27) != 0, c1&(1<<28) != 0
	if !fma || !osxsave || !avx {
		return false
	}
	if xcr0, _ := xgetbv(); xcr0&6 != 6 { // XMM and YMM state enabled
		return false
	}
	_, b7, _, _ := cpuid(7, 0)
	return b7&(1<<5) != 0 // AVX2
}()
