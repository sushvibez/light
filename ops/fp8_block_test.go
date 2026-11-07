package ops_test

import (
	"math/rand"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// DequantizeF8Block against a per-element reference on a 3 × 4 grid of
// 4 × 8 blocks (every block has its own scale), F32 and BF16 out, CPU and
// CUDA.
func TestDequantizeF8Block(t *testing.T) {
	const R, C, br, bc = 12, 32, 4, 8
	rng := rand.New(rand.NewSource(1))
	codes := make([]float32, R*C)
	for i := range codes {
		codes[i] = dtype.F8FromF32(float32(rng.NormFloat64() * 50)).ToF32()
	}
	scales := make([]float32, (R/br)*(C/bc))
	for i := range scales {
		scales[i] = float32(rng.Float64()*0.01 + 1e-4)
	}
	want := make([]float32, R*C)
	for i := range R {
		for j := range C {
			want[i*C+j] = codes[i*C+j] * scales[(i/br)*(C/bc)+j/bc]
		}
	}
	devs := []device.Device{device.CPU()}
	if device.CUDAAvailable() {
		if g, err := device.CUDA(0); err == nil {
			devs = append(devs, g)
		}
	}
	for _, dev := range devs {
		x, _ := tensor.New(codes, []int{R, C}, dev)
		w, err := ops.Cast(x, dtype.F8E4M3)
		if err != nil {
			t.Fatal(err)
		}
		s, _ := tensor.New(scales, []int{R / br, C / bc}, dev)
		for _, dt := range []dtype.DType{dtype.F32, dtype.BF16} {
			out, err := ops.DequantizeF8Block(w, s, br, bc, dt)
			if err != nil {
				t.Fatal(err)
			}
			if out.DType() != dt || out.Dim(0) != R || out.Dim(1) != C {
				t.Fatalf("%v %v: got %v %v", dev, dt, out.DType(), out.Shape())
			}
			o32, _ := ops.Cast(out, dtype.F32)
			o32, _ = o32.To(device.CPU())
			got := tensor.MustSlice[float32](o32)
			for i := range want {
				ref := want[i]
				if dt == dtype.BF16 {
					ref = dtype.BF16FromF32(ref).ToF32()
				}
				if got[i] != ref {
					t.Fatalf("%v %v: [%d, %d] = %g, want %g", dev, dt, i/C, i%C, got[i], ref)
				}
			}
		}
		if _, err := ops.DequantizeF8Block(w, s, 5, bc, dtype.F32); err == nil {
			t.Errorf("%v: block 5 on 12 rows accepted", dev)
		}
	}
}

// The CUDA kernels decode E4M3 with a bit trick instead of FP8 hardware
// (fp8x4_to_half2x2 in kernels/cuda/src/fp8_block.cu): four codes in a
// little-endian word become two half2 — codes (0, 2) and (1, 3) — each
// value · 2⁻⁸. Replayed here, word-level masks included, for every code
// against the reference decoder.
func TestFP8HalfTrickMatchesDecoder(t *testing.T) {
	for c := 0; c < 256; c++ {
		if c&0x7F == 0x7F { // E4M3's NaN, never a weight
			continue
		}
		want := dtype.Float8(c).ToF32()
		for pos := 0; pos < 4; pos++ {
			v := uint32(c) << (8 * pos)
			lo := ((v & 0x007F007F) << 7) | ((v & 0x00800080) << 8)
			hi := ((v & 0x7F007F00) >> 1) | (v & 0x80008000)
			var bits uint16
			switch pos { // lo = (c0, c2), hi = (c1, c3); low half first
			case 0:
				bits = uint16(lo)
			case 2:
				bits = uint16(lo >> 16)
			case 1:
				bits = uint16(hi)
			case 3:
				bits = uint16(hi >> 16)
			}
			if got := dtype.Half(bits).ToF32() * 256; got != want {
				t.Fatalf("code %#02x in byte %d: %g, want %g", c, pos, got, want)
			}
		}
	}
}

// MatMulF8Block against x · dequant(W)ᵀ computed in F64 on the host: the
// fused GEMV (M ≤ 8, CUDA) and the dequantize + GEMM path (M = 9, and CPU),
// F32 and BF16 activations, 128 × 128 blocks on a weight with several of
// them in each direction.
func TestMatMulF8Block(t *testing.T) {
	const N, K, b = 384, 512, 128
	rng := rand.New(rand.NewSource(2))
	codes := make([]float32, N*K)
	for i := range codes {
		codes[i] = dtype.F8FromF32(float32(rng.NormFloat64() * 60)).ToF32()
	}
	scales := make([]float32, (N/b)*(K/b))
	for i := range scales {
		scales[i] = float32(rng.Float64()*2e-3 + 1e-4)
	}
	devs := []device.Device{device.CPU()}
	if device.CUDAAvailable() {
		if g, err := device.CUDA(0); err == nil {
			devs = append(devs, g)
		}
	}
	for _, dev := range devs {
		wf, _ := tensor.New(codes, []int{N, K}, dev)
		w, err := ops.Cast(wf, dtype.F8E4M3)
		if err != nil {
			t.Fatal(err)
		}
		s, _ := tensor.New(scales, []int{N / b, K / b}, dev)
		for _, M := range []int{1, 3, 8, 9} {
			xs := make([]float32, M*K)
			for i := range xs {
				xs[i] = dtype.BF16FromF32(float32(rng.NormFloat64())).ToF32() // exact in BF16 too
			}
			want := make([]float64, M*N)
			for m := range M {
				for n := range N {
					var acc float64
					for k := range K {
						acc += float64(xs[m*K+k]) * float64(codes[n*K+k]) * float64(scales[(n/b)*(K/b)+k/b])
					}
					want[m*N+n] = acc
				}
			}
			for _, dt := range []dtype.DType{dtype.F32, dtype.BF16} {
				x, _ := tensor.New(xs, []int{M, K}, dev)
				if dt != dtype.F32 {
					x, _ = ops.Cast(x, dt)
				}
				y, err := ops.MatMulF8Block(x, w, s, b, b)
				if err != nil {
					t.Fatalf("%v M=%d %v: %v", dev, M, dt, err)
				}
				if y.DType() != dt || y.Dim(0) != M || y.Dim(1) != N {
					t.Fatalf("%v M=%d %v: got %v %v", dev, M, dt, y.DType(), y.Shape())
				}
				y32, _ := ops.Cast(y, dtype.F32)
				y32, _ = y32.To(device.CPU())
				got := tensor.MustSlice[float32](y32)
				// |y| is O(1); F32 accumulation ~1e-5, BF16 output rounding
				// (and BF16 weights on the GEMM path) ~1e-2 relative.
				tol := 1e-4
				if dt == dtype.BF16 {
					tol = 3e-2
				}
				var worst float64
				for i := range want {
					d := float64(got[i]) - want[i]
					if d < 0 {
						d = -d
					}
					if r := d / (1 + abs64(want[i])); r > worst {
						worst = r
					}
				}
				if worst > tol {
					t.Errorf("%v M=%d %v: max relative error %.2e (tol %g)", dev, M, dt, worst, tol)
				}
			}
		}
	}
}

func abs64(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
