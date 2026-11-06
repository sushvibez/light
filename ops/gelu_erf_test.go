package ops_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

func geluErfRef(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }

func TestGeluErfCPU(t *testing.T) {
	xs := []float32{-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4}
	a := mkF32(xs, len(xs))
	out, err := ops.GeluErf(a)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](out)
	for i, x := range xs {
		want := geluErfRef(float64(x))
		if math.Abs(float64(got[i])-want) > 1e-6 {
			t.Errorf("gelu_erf(%v) = %v, want %v", x, got[i], want)
		}
	}
	// F64 path.
	xs64 := make([]float64, len(xs))
	for i, x := range xs {
		xs64[i] = float64(x)
	}
	a64, _ := tensor.New(xs64, []int{len(xs)}, device.CPU())
	out64, err := ops.GeluErf(a64)
	if err != nil {
		t.Fatal(err)
	}
	if out64.DType() != dtype.F64 {
		t.Fatalf("dtype %v", out64.DType())
	}
	for i, v := range tensor.MustSlice[float64](out64) {
		if math.Abs(v-geluErfRef(xs64[i])) > 1e-12 {
			t.Errorf("f64 gelu_erf(%v) = %v", xs64[i], v)
		}
	}
	// Exact vs tanh-approx differ measurably away from 0 (sanity that the
	// two ops are not aliases).
	approx, _ := ops.Gelu(a)
	ga := tensor.MustSlice[float32](approx)
	if math.Abs(float64(ga[1]-got[1])) < 1e-5 {
		t.Errorf("gelu and gelu_erf should differ at x=-2: %v vs %v", ga[1], got[1])
	}
}

// GELU (tanh and erf) in F32, BF16 and F16 on CUDA against the CPU result
// (half kernels compute in F32 and round once).
func TestGeluHalfPrecisionCUDAParity(t *testing.T) {
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	cpu := device.CPU()
	xs := []float32{-3, -2, -1.5, -0.5, -0.1, 0, 0.1, 0.7, 1, 1.5, 2, 3.25}
	x32, _ := tensor.New(xs, []int{len(xs)}, cpu)
	for _, dt := range []dtype.DType{dtype.F32, dtype.BF16, dtype.F16} {
		x, err := ops.Cast(x32, dt)
		if err != nil {
			t.Fatal(err)
		}
		xg, err := x.To(gpu)
		if err != nil {
			t.Fatal(err)
		}
		for name, fn := range map[string]func(*tensor.Tensor) (*tensor.Tensor, error){"gelu": ops.Gelu, "gelu_erf": ops.GeluErf} {
			c, err := fn(x)
			if err != nil {
				t.Fatalf("%s %v CPU: %v", name, dt, err)
			}
			g, err := fn(xg)
			if err != nil {
				t.Fatalf("%s %v CUDA: %v", name, dt, err)
			}
			if g.DType() != dt {
				t.Fatalf("%s %v: CUDA output dtype %v", name, dt, g.DType())
			}
			c32, _ := ops.Cast(c, dtype.F32)
			gb, _ := g.To(cpu)
			g32, _ := ops.Cast(gb, dtype.F32)
			cs, gs := tensor.MustSlice[float32](c32), tensor.MustSlice[float32](g32)
			tol := 1e-5
			if dt != dtype.F32 {
				tol = 2e-2
			}
			for i := range cs {
				if math.Abs(float64(cs[i]-gs[i])) > tol {
					t.Errorf("%s %v [%d]: CPU %v CUDA %v", name, dt, i, cs[i], gs[i])
				}
			}
		}
	}
}
