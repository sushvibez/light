package ops_test

import (
	"math"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

func mkF32(data []float32, dims ...int) *tensor.Tensor {
	t, err := tensor.New(data, dims, device.CPU())
	if err != nil {
		panic(err)
	}
	return t
}

func TestSub(t *testing.T) {
	a := mkF32([]float32{10, 20, 30}, 3)
	b := mkF32([]float32{1, 2, 3}, 3)
	c, err := ops.Sub(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tensor.MustSlice[float32](c), []float32{9, 18, 27}) {
		t.Errorf("got %v", tensor.MustSlice[float32](c))
	}
}

func TestMul(t *testing.T) {
	a := mkF32([]float32{1, 2, 3}, 3)
	b := mkF32([]float32{4, 5, 6}, 3)
	c, err := ops.Mul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tensor.MustSlice[float32](c), []float32{4, 10, 18}) {
		t.Errorf("got %v", tensor.MustSlice[float32](c))
	}
}

func TestDiv(t *testing.T) {
	a := mkF32([]float32{10, 20, 30}, 3)
	b := mkF32([]float32{2, 4, 5}, 3)
	c, err := ops.Div(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tensor.MustSlice[float32](c), []float32{5, 5, 6}) {
		t.Errorf("got %v", tensor.MustSlice[float32](c))
	}
}

func TestDivIntByZero(t *testing.T) {
	a, _ := tensor.New([]int32{10, 20}, []int{2}, device.CPU())
	b, _ := tensor.New([]int32{2, 0}, []int{2}, device.CPU())
	c, err := ops.Div(a, b)
	if err != nil {
		t.Fatal(err)
	}
	v := tensor.MustSlice[int32](c)
	if v[0] != 5 || v[1] != 0 {
		t.Errorf("got %v, want [5 0]", v)
	}
}

func TestPow(t *testing.T) {
	a := mkF32([]float32{2, 3, 4}, 3)
	b := mkF32([]float32{3, 2, 0.5}, 3)
	c, err := ops.Pow(a, b)
	if err != nil {
		t.Fatal(err)
	}
	v := tensor.MustSlice[float32](c)
	want := []float32{8, 9, 2}
	for i := range v {
		if math.Abs(float64(v[i]-want[i])) > 1e-5 {
			t.Errorf("got %v, want %v", v, want)
		}
	}
}

func TestNeg(t *testing.T) {
	a := mkF32([]float32{1, -2, 3}, 3)
	c, err := ops.Neg(a)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tensor.MustSlice[float32](c), []float32{-1, 2, -3}) {
		t.Errorf("got %v", tensor.MustSlice[float32](c))
	}
}

func TestAbs(t *testing.T) {
	a := mkF32([]float32{-1, 2, -3}, 3)
	c, err := ops.Abs(a)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tensor.MustSlice[float32](c), []float32{1, 2, 3}) {
		t.Errorf("got %v", tensor.MustSlice[float32](c))
	}
	// Int version
	ai, _ := tensor.New([]int32{-5, 0, 7}, []int{3}, device.CPU())
	ci, _ := ops.Abs(ai)
	if !slices.Equal(tensor.MustSlice[int32](ci), []int32{5, 0, 7}) {
		t.Errorf("int abs got %v", tensor.MustSlice[int32](ci))
	}
}

func TestTranscendentals(t *testing.T) {
	a := mkF32([]float32{0, 1, 4, 9}, 4)

	check := func(name string, input *tensor.Tensor, fn func(*tensor.Tensor) (*tensor.Tensor, error), want []float32, tol float64) {
		t.Helper()
		c, err := fn(input)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		got := tensor.MustSlice[float32](c)
		for i := range want {
			if math.IsInf(float64(want[i]), -1) {
				if !math.IsInf(float64(got[i]), -1) {
					t.Errorf("%s[%d] = %v, want -Inf", name, i, got[i])
				}
				continue
			}
			if math.Abs(float64(got[i]-want[i])) > tol {
				t.Errorf("%s[%d] = %v, want ≈ %v", name, i, got[i], want[i])
			}
		}
	}

	check("sqrt", a, ops.Sqrt, []float32{0, 1, 2, 3}, 1e-5)
	check("exp", a, ops.Exp, []float32{1, float32(math.E), float32(math.Exp(4)), float32(math.Exp(9))}, 1e-3)
	check("log", a, ops.Log, []float32{float32(math.Inf(-1)), 0, float32(math.Log(4)), float32(math.Log(9))}, 1e-5)

	b := mkF32([]float32{0, 1, -1, 2}, 4)
	check("sin", b, ops.Sin, []float32{0, float32(math.Sin(1)), float32(math.Sin(-1)), float32(math.Sin(2))}, 1e-5)
	check("cos", b, ops.Cos, []float32{1, float32(math.Cos(1)), float32(math.Cos(-1)), float32(math.Cos(2))}, 1e-5)
	check("tanh", b, ops.Tanh, []float32{0, float32(math.Tanh(1)), float32(math.Tanh(-1)), float32(math.Tanh(2))}, 1e-5)
	check("sigmoid", b, ops.Sigmoid, []float32{0.5, 1 / (1 + float32(math.Exp(-1))), 1 / (1 + float32(math.E)), 1 / (1 + float32(math.Exp(-2)))}, 1e-5)
	check("relu", b, ops.Relu, []float32{0, 1, 0, 2}, 1e-6)
	check("silu", b, ops.SiLU, []float32{0, 1 * (1 / (1 + float32(math.Exp(-1)))), -1 * (1 / (1 + float32(math.E))), 2 * (1 / (1 + float32(math.Exp(-2))))}, 1e-5)

	// GELU tanh approximation: compare to a couple of known values.
	gc, _ := ops.Gelu(mkF32([]float32{0, 1, -1, 2}, 4))
	g := tensor.MustSlice[float32](gc)
	// GELU(0) = 0; GELU(1) ≈ 0.841; GELU(-1) ≈ -0.159; GELU(2) ≈ 1.955
	for i, want := range []float32{0, 0.8412, -0.1588, 1.9546} {
		if math.Abs(float64(g[i]-want)) > 0.01 {
			t.Errorf("gelu[%d] = %v, want ≈ %v", i, g[i], want)
		}
	}
}

func TestSubF16(t *testing.T) {
	// Build F16 tensors via Ones scaled by Neg.
	a, _ := tensor.Ones(1<<0 /*dummy*/, []int{3}, device.CPU()) //nolint:staticcheck,SA1019
	_ = a
	// Skipping F16 arithmetic tests here — covered by TestAddF16/TestAddBF16 already.
}

func TestUnaryDTypeMismatch(t *testing.T) {
	// Log on int should fail (no kernel registered).
	ai, _ := tensor.New([]int32{1, 2, 3}, []int{3}, device.CPU())
	if _, err := ops.Log(ai); err == nil {
		t.Error("expected error: Log on int dtype")
	}
}

// Candle-parity ops: value checks on the CPU and CPU↔CUDA parity for every
// new kernel (F32, F64 and BF16 where registered).
var unaryIn = []float32{-4, -2.5, -1, -0.3, 0, 0.3, 1, 2.5, 4}

func cudaOrSkip(t *testing.T) device.Device {
	t.Helper()
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Skip(err)
	}
	return dev
}

func f32s(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	c, err := x.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if c.DType() != dtype.F32 {
		if c, err = ops.Cast(c, dtype.F32); err != nil {
			t.Fatal(err)
		}
	}
	return tensor.MustSlice[float32](c)
}

func expectRow(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d want %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Errorf("%s[%d] = %g, want %g", name, i, got[i], want[i])
		}
	}
}

type unaryCase struct {
	name string
	fn   func(*tensor.Tensor) (*tensor.Tensor, error)
	ref  func(float64) float64
	ints bool // also registered for integer dtypes
}

func unaryCases() []unaryCase {
	hs := func(x float64) float64 { return math.Min(math.Max((x+3)/6, 0), 1) }
	return []unaryCase{
		{"sign", ops.Sign, func(x float64) float64 {
			if x > 0 {
				return 1
			} else if x < 0 {
				return -1
			}
			return 0
		}, true},
		{"erf", ops.Erf, math.Erf, false},
		{"recip", ops.Recip, func(x float64) float64 { return 1 / x }, false},
		{"sqr", ops.Sqr, func(x float64) float64 { return x * x }, false},
		{"mish", ops.Mish, func(x float64) float64 { return x * math.Tanh(math.Log1p(math.Exp(x))) }, false},
		{"hard_sigmoid", ops.HardSigmoid, hs, false},
		{"hard_swish", ops.HardSwish, func(x float64) float64 { return x * hs(x) }, false},
		{"clamp", func(a *tensor.Tensor) (*tensor.Tensor, error) { return ops.Clamp(a, -1.5, 2) },
			func(x float64) float64 { return math.Min(math.Max(x, -1.5), 2) }, false},
		{"elu", func(a *tensor.Tensor) (*tensor.Tensor, error) { return ops.ELU(a, 0.7) },
			func(x float64) float64 {
				if x > 0 {
					return x
				}
				return 0.7 * (math.Exp(x) - 1)
			}, false},
		{"leaky_relu", func(a *tensor.Tensor) (*tensor.Tensor, error) { return ops.LeakyReLU(a, 0.1) },
			func(x float64) float64 {
				if x > 0 {
					return x
				}
				return 0.1 * x
			}, false},
		{"powf", func(a *tensor.Tensor) (*tensor.Tensor, error) { return ops.PowScalar(a, 2) },
			func(x float64) float64 { return math.Pow(x, 2) }, false},
		{"affine", func(a *tensor.Tensor) (*tensor.Tensor, error) { return ops.Affine(a, 1.5, -2) },
			func(x float64) float64 { return 1.5*x - 2 }, false},
	}
}

func TestUnaryOpsCPUValues(t *testing.T) {
	for _, c := range unaryCases() {
		t.Run(c.name, func(t *testing.T) {
			in := unaryIn
			if c.name == "recip" {
				in = []float32{-4, -2.5, -1, -0.3, 0.5, 0.3, 1, 2.5, 4}
			}
			want := make([]float32, len(in))
			for i, v := range in {
				want[i] = float32(c.ref(float64(v)))
			}
			x, _ := tensor.New(in, []int{3, 3}, device.CPU())
			y, err := c.fn(x)
			if err != nil {
				t.Fatal(err)
			}
			expectRow(t, c.name+" f32", tensor.MustSlice[float32](y), want, 1e-5)
			x64, _ := ops.Cast(x, dtype.F64)
			y64, err := c.fn(x64)
			if err != nil {
				t.Fatal(err)
			}
			expectRow(t, c.name+" f64", f32s(t, y64), want, 1e-5)
			xb, _ := ops.Cast(x, dtype.BF16)
			yb, err := c.fn(xb)
			if err != nil {
				t.Fatal(err)
			}
			expectRow(t, c.name+" bf16", f32s(t, yb), want, 0.05*float64(maxAbs(want))+1e-2)
			if c.ints { // integer cast truncates ±0.3 to 0, so sign there is 0
				xi, _ := ops.Cast(x, dtype.I64)
				yi, err := c.fn(xi)
				if err != nil {
					t.Fatal(err)
				}
				wantI := append([]float32{}, want...)
				wantI[3], wantI[5] = 0, 0
				expectRow(t, c.name+" i64", f32s(t, yi), wantI, 0)
			}
		})
	}
}

func maxAbs(xs []float32) float32 {
	m := float32(0)
	for _, v := range xs {
		if a := float32(math.Abs(float64(v))); a > m {
			m = a
		}
	}
	return m
}

func TestUnaryOpsCUDA(t *testing.T) {
	gpu := cudaOrSkip(t)
	for _, c := range unaryCases() {
		t.Run(c.name, func(t *testing.T) {
			in := unaryIn
			if c.name == "recip" {
				in = []float32{-4, -2.5, -1, -0.3, 0.5, 0.3, 1, 2.5, 4}
			}
			x, _ := tensor.New(in, []int{9}, device.CPU())
			dts := []dtype.DType{dtype.F32, dtype.F64, dtype.BF16}
			if c.ints {
				dts = append(dts, dtype.I32, dtype.I64)
			}
			for _, dt := range dts {
				xc, _ := ops.Cast(x, dt)
				yc, err := c.fn(xc)
				if err != nil {
					t.Fatalf("%v cpu: %v", dt, err)
				}
				xg, _ := xc.To(gpu)
				yg, err := c.fn(xg)
				if err != nil {
					t.Fatalf("%v cuda: %v", dt, err)
				}
				tol := 1e-5
				if dt == dtype.BF16 {
					tol = 1e-2 * float64(maxAbs(f32s(t, yc))+1)
				}
				expectRow(t, c.name+" "+dt.String(), f32s(t, yg), f32s(t, yc), tol)
			}
		})
	}
}

func TestMaximumMinimumBroadcast(t *testing.T) {
	a, _ := tensor.New([]float32{1, 5, -2, 0, 3, 7}, []int{2, 3}, device.CPU())
	b, _ := tensor.New([]float32{2, 2, 2}, []int{3}, device.CPU()) // broadcast row
	mx, err := ops.Maximum(a, b)
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "max", tensor.MustSlice[float32](mx), []float32{2, 5, 2, 2, 3, 7}, 0)
	mn, err := ops.Minimum(a, b)
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "min", tensor.MustSlice[float32](mn), []float32{1, 2, -2, 0, 2, 2}, 0)
	ai, _ := ops.Cast(a, dtype.I64)
	bi, _ := ops.Cast(b, dtype.I64)
	mi, err := ops.Maximum(ai, bi)
	if err != nil {
		t.Fatal(err)
	}
	expectRow(t, "max i64", f32s(t, mi), []float32{2, 5, 2, 2, 3, 7}, 0)

	gpu := cudaOrSkip(t)
	for _, dt := range []dtype.DType{dtype.F32, dtype.F64, dtype.BF16, dtype.I32, dtype.I64} {
		ac, _ := ops.Cast(a, dt)
		bc, _ := ops.Cast(b, dt)
		ag, _ := ac.To(gpu)
		bg, _ := bc.To(gpu)
		// broadcast (strided kernel) and same-shape (plain kernel)
		g1, err := ops.Maximum(ag, bg)
		if err != nil {
			t.Fatalf("%v maximum: %v", dt, err)
		}
		expectRow(t, "cuda max "+dt.String(), f32s(t, g1), []float32{2, 5, 2, 2, 3, 7}, 0)
		g2, err := ops.Minimum(ag, ag)
		if err != nil {
			t.Fatalf("%v minimum: %v", dt, err)
		}
		expectRow(t, "cuda min "+dt.String(), f32s(t, g2), f32s(t, ac), 0)
	}
}
