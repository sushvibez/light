//go:build !nowebgpu

package webgpu_test

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

// randomBlocks returns N rows of K elements of random GGUF blocks of dt:
// random bytes everywhere, with the f16 scale fields set to small finite
// values so every element is finite.
func randomBlocks(rng *rand.Rand, dt dtype.DType, N, K int) []byte {
	be := dt.BlockElems()
	bb := dt.ByteSize(be)
	b := make([]byte, N*(K/be)*bb)
	for i := range b {
		b[i] = byte(rng.Uint32())
	}
	f16 := func(off int, v float32) {
		binary.LittleEndian.PutUint16(b[off:], uint16(dtype.HalfFromF32(v)))
	}
	for blk := 0; blk < len(b)/bb; blk++ {
		o := blk * bb
		switch dt {
		case dtype.Q4_0, dtype.Q5_0, dtype.Q8_0:
			f16(o, 0.01+0.02*rng.Float32())
		case dtype.Q4_K:
			f16(o, 0.002+0.004*rng.Float32())
			f16(o+2, 0.002+0.004*rng.Float32())
		case dtype.Q6_K:
			f16(o+208, 0.0005+0.001*rng.Float32())
		}
	}
	return b
}

// The GGUF kernels on WebGPU (dequant, the fused decode GEMV for M ≤ 8, the
// dequant + matmul path for larger M, and the quantized row gather) equal
// the CPU decoders in light/quant, per format.
func TestGGUFKernelsParity(t *testing.T) {
	g, err := device.WebGPU(0)
	if err != nil {
		t.Skip(err)
	}
	cpu := device.CPU()
	rng := rand.New(rand.NewPCG(7, 8))
	const N, K = 40, 512
	for _, dt := range []dtype.DType{dtype.Q4_0, dtype.Q5_0, dtype.Q8_0, dtype.Q4_K, dtype.Q6_K} {
		blocks := randomBlocks(rng, dt, N, K)
		wc, err := tensor.FromBytes(blocks, dt, []int{N, K}, cpu)
		if err != nil {
			t.Fatal(err)
		}
		wg, err := wc.To(g)
		if err != nil {
			t.Fatal(err)
		}
		ref, err := quant.Dequantize(wc)
		if err != nil {
			t.Fatal(err)
		}
		wd := tensor.MustSlice[float32](ref)
		// Dequant.
		got, err := quant.Dequantize(wg)
		if err != nil {
			t.Fatalf("%v dequant: %v", dt, err)
		}
		got, _ = got.To(cpu)
		for i, v := range tensor.MustSlice[float32](got) {
			if d := math.Abs(float64(v - wd[i])); d > 1e-6*max(1, math.Abs(float64(wd[i]))) {
				t.Fatalf("%v dequant [%d] = %g, cpu %g", dt, i, v, wd[i])
			}
		}
		// MatMulQ.
		for _, M := range []int{1, 3, 8, 9} {
			xs := make([]float32, M*K)
			for i := range xs {
				xs[i] = float32(rng.NormFloat64())
			}
			x, _ := tensor.New(xs, []int{M, K}, g)
			y, err := ops.MatMulQ(x, wg)
			if err != nil {
				t.Fatalf("%v M=%d: %v", dt, M, err)
			}
			y, _ = y.To(cpu)
			ys := tensor.MustSlice[float32](y)
			for m := range M {
				for n := range N {
					var want, mag float64
					for k := range K {
						p := float64(xs[m*K+k]) * float64(wd[n*K+k])
						want += p
						mag += math.Abs(p)
					}
					if d := math.Abs(float64(ys[m*N+n]) - want); d > 1e-5*max(1, mag) {
						t.Fatalf("%v M=%d [%d,%d] = %g, want %g", dt, M, m, n, ys[m*N+n], want)
					}
				}
			}
		}
		// Quantized row gather (embedding lookup), then dequant.
		ids := []int64{3, 0, 39, 3, 17}
		ic, _ := tensor.New(ids, []int{len(ids)}, g)
		rows, err := ops.IndexSelect(wg, ic)
		if err != nil {
			t.Fatalf("%v gather: %v", dt, err)
		}
		rd, err := quant.Dequantize(rows)
		if err != nil {
			t.Fatal(err)
		}
		rd, _ = rd.To(cpu)
		rs := tensor.MustSlice[float32](rd)
		for r, id := range ids {
			for k := range K {
				if rs[r*K+k] != wd[int(id)*K+k] {
					t.Fatalf("%v gather row %d (id %d) [%d] = %g, want %g", dt, r, id, k, rs[r*K+k], wd[int(id)*K+k])
				}
			}
		}
	}
}
