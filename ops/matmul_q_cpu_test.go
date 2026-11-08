package ops_test

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/quant"
	"light/tensor"
)

// The CPU matmul_q kernels (row decode + SIMD dot, per GGUF format) equal
// x · dequant(W)ᵀ computed from light/quant's decoders, for decode (M = 1)
// and prefill (M = 9) shapes.
func TestMatMulQCPUKernels(t *testing.T) {
	cpu := device.CPU()
	rng := rand.New(rand.NewPCG(9, 10))
	const N, K = 37, 512
	for _, dt := range []dtype.DType{dtype.Q4_0, dtype.Q5_0, dtype.Q8_0, dtype.Q4_K, dtype.Q6_K} {
		if !ops.MatMulQFusedOn(device.KindCPU, dt) {
			t.Fatalf("%v: no CPU matmul_q kernel", dt)
		}
		be := dt.BlockElems()
		bb := dt.ByteSize(be)
		raw := make([]byte, N*(K/be)*bb)
		for i := range raw {
			raw[i] = byte(rng.Uint32())
		}
		f16 := func(off int, v float32) { binary.LittleEndian.PutUint16(raw[off:], uint16(dtype.HalfFromF32(v))) }
		for b := 0; b < len(raw)/bb; b++ {
			o := b * bb
			switch dt {
			case dtype.Q4_K:
				f16(o, 0.003)
				f16(o+2, 0.002+0.002*rng.Float32())
			case dtype.Q6_K:
				f16(o+208, 0.0007)
			default:
				f16(o, 0.01+0.02*rng.Float32())
			}
		}
		w, err := tensor.FromBytes(raw, dt, []int{N, K}, cpu)
		if err != nil {
			t.Fatal(err)
		}
		wd, err := quant.Dequantize(w)
		if err != nil {
			t.Fatal(err)
		}
		wf := tensor.MustSlice[float32](wd)
		for _, M := range []int{1, 9} {
			xs := make([]float32, M*K)
			for i := range xs {
				xs[i] = float32(rng.NormFloat64())
			}
			x, _ := tensor.New(xs, []int{M, K}, cpu)
			y, err := ops.MatMulQ(x, w)
			if err != nil {
				t.Fatal(err)
			}
			ys := tensor.MustSlice[float32](y)
			for m := range M {
				for n := range N {
					var want, mag float64
					for k := range K {
						p := float64(xs[m*K+k]) * float64(wf[n*K+k])
						want += p
						mag += math.Abs(p)
					}
					if d := math.Abs(float64(ys[m*N+n]) - want); d > 1e-5*(1+mag) {
						t.Fatalf("%v M=%d [%d,%d] = %g, want %g", dt, M, m, n, ys[m*N+n], want)
					}
				}
			}
		}
	}
}
