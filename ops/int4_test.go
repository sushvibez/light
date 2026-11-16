package ops_test

import (
	"math"
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
	if g, err := device.WebGPU(0); err == nil { // F32 only
		devs = append(devs, g)
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
			if dt != dtype.F32 && device.IsWebGPU(dev) {
				continue
			}
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
				if dt != dtype.F32 && device.IsWebGPU(dev) {
					continue
				}
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

// MoEInt4 with F32 activations on the CPU (each expert's rows through the
// fused int4 kernel) against Σ_k w_k · down_e(silu(gate_e x) · up_e x) in
// F64 on the dequantized experts.
func TestMoEInt4MatchesDense(t *testing.T) {
	const T, H, I, E, topk, G = 5, 64, 32, 4, 2, 32
	rng := rand.New(rand.NewSource(7))
	cpu := device.CPU()
	gu, dn := make([]quant.Int4, E), make([]quant.Int4, E)
	for e := range E {
		gu[e], dn[e] = randomInt4(t, rng, 2*I, H, G, cpu), randomInt4(t, rng, H, I, G, cpu)
	}
	x := make([]float32, T*H)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	ids := []int64{0, 2, 2, 1, 3, 0, 2, 3, 1, 2}
	ws := make([]float32, T*topk)
	for i := range ws {
		ws[i] = float32(rng.Float64())
	}
	route, err := ops.MoERouteFromIDs(ids, ws, T, topk, cpu)
	if err != nil {
		t.Fatal(err)
	}
	xt, _ := tensor.New(x, []int{T, H}, cpu)
	out, err := ops.MoEInt4(xt, gu, dn, route, topk, ops.SiLU)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)
	for tk := range T {
		want := make([]float64, H)
		for j := range topk {
			e := ids[tk*topk+j]
			wg, _ := gu[e].Dequant()
			wd, _ := dn[e].Dequant()
			h := make([]float64, I)
			for i := range I {
				var g, u float64
				for k := range H {
					g += float64(wg[i*H+k]) * float64(x[tk*H+k])
					u += float64(wg[(I+i)*H+k]) * float64(x[tk*H+k])
				}
				h[i] = g / (1 + math.Exp(-g)) * u
			}
			for n := range H {
				var y float64
				for i := range I {
					y += float64(wd[n*I+i]) * h[i]
				}
				want[n] += float64(ws[tk*topk+j]) * y
			}
		}
		for n := range H {
			if d := math.Abs(float64(got[tk*H+n]) - want[n]); d > 1e-4*(1+math.Abs(want[n])) {
				t.Fatalf("token %d [%d] = %g, want %g", tk, n, got[tk*H+n], want[n])
			}
		}
	}
}
