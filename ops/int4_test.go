package ops_test

import (
	"math/rand"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/quant"
	"light/tensor"
)

// randomInt4 builds a canonical 4-bit weight (N, K) with groups of G.
func randomInt4(t *testing.T, rng *rand.Rand, N, K, G int, dev device.Device) quant.Int4 {
	t.Helper()
	codes := make([]uint32, N*K/8)
	for i := range codes {
		codes[i] = rng.Uint32()
	}
	s := make([]float32, N*K/G)
	z := make([]float32, N*K/G)
	for i := range s {
		s[i] = float32(rng.Float64()*0.02 + 1e-3)
		z[i] = float32(rng.Intn(16))
	}
	c, _ := tensor.New(codes, []int{N, K / 8}, dev)
	st, _ := tensor.New(s, []int{N, K / G}, dev)
	zt, _ := tensor.New(z, []int{N, K / G}, dev)
	return quant.Int4{Codes: c, Scales: st, Zeros: zt, Group: G}
}

func int4Devices() []device.Device {
	devs := []device.Device{device.CPU()}
	if device.CUDAAvailable() {
		if g, err := device.CUDA(0); err == nil {
			devs = append(devs, g)
		}
	}
	return devs
}

// DequantizeInt4 (CPU and CUDA kernels, F32 / BF16 out) equals the Go
// reference quant.Int4.Dequant bit for bit.
func TestDequantizeInt4(t *testing.T) {
	const N, K, G = 24, 256, 64
	for _, dev := range int4Devices() {
		w := randomInt4(t, rand.New(rand.NewSource(3)), N, K, G, dev)
		want, err := w.Dequant()
		if err != nil {
			t.Fatal(err)
		}
		for _, dt := range []dtype.DType{dtype.F32, dtype.BF16} {
			out, err := ops.DequantizeInt4(w, dt)
			if err != nil {
				t.Fatal(err)
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
					t.Fatalf("%v %v: [%d] = %g, want %g", dev, dt, i, got[i], ref)
				}
			}
		}
	}
}

// MatMulInt4 against x · dequant(W)ᵀ in F64: the fused GEMV (M ≤ 8, CUDA)
// and the dequantize + GEMM path (M = 9, and CPU), F32 and BF16.
func TestMatMulInt4(t *testing.T) {
	const N, K, G = 96, 256, 64
	for _, dev := range int4Devices() {
		rng := rand.New(rand.NewSource(4))
		w := randomInt4(t, rng, N, K, G, dev)
		wd, err := w.Dequant()
		if err != nil {
			t.Fatal(err)
		}
		for _, M := range []int{1, 3, 8, 9} {
			xs := make([]float32, M*K)
			for i := range xs {
				xs[i] = dtype.BF16FromF32(float32(rng.NormFloat64())).ToF32()
			}
			want := make([]float64, M*N)
			for m := range M {
				for n := range N {
					var acc float64
					for k := range K {
						acc += float64(xs[m*K+k]) * float64(wd[n*K+k])
					}
					want[m*N+n] = acc
				}
			}
			for _, dt := range []dtype.DType{dtype.F32, dtype.BF16} {
				x, _ := tensor.New(xs, []int{M, K}, dev)
				if dt != dtype.F32 {
					x, _ = ops.Cast(x, dt)
				}
				y, err := ops.MatMulInt4(x, w)
				if err != nil {
					t.Fatalf("%v M=%d %v: %v", dev, M, dt, err)
				}
				if y.DType() != dt || y.Dim(0) != M || y.Dim(1) != N {
					t.Fatalf("%v M=%d %v: got %v %v", dev, M, dt, y.DType(), y.Shape())
				}
				y32, _ := ops.Cast(y, dtype.F32)
				y32, _ = y32.To(device.CPU())
				got := tensor.MustSlice[float32](y32)
				tol := 1e-4
				if dt == dtype.BF16 {
					tol = 3e-2
				}
				for i := range want {
					d := float64(got[i]) - want[i]
					if d < 0 {
						d = -d
					}
					scale := 1.0
					if a := want[i]; a > 1 || a < -1 {
						scale = abs64(a)
					}
					if d/scale > tol {
						t.Fatalf("%v M=%d %v: [%d] = %g, want %g", dev, M, dt, i, got[i], want[i])
					}
				}
			}
		}
	}
}
