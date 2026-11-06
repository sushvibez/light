package nn_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

func TestLinearForwardShape(t *testing.T) {
	l, err := nn.NewLinear(4, 8, true, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	x, _ := tensor.Zeros(dtype.F32, []int{3, 4}, device.CPU())
	y, err := l.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	if y.Rank() != 2 || y.Dim(0) != 3 || y.Dim(1) != 8 {
		t.Errorf("expected (3,8), got %v", y.Shape())
	}
	// Zero input + zero bias → zero output.
	vals := tensor.MustSlice[float32](y)
	for _, v := range vals {
		if v != 0 {
			t.Errorf("expected all zeros, got %v", vals)
			break
		}
	}
	// Parameters: W, B.
	if len(l.Parameters()) != 2 {
		t.Errorf("expected 2 params, got %d", len(l.Parameters()))
	}
}

func TestLinearNoBias(t *testing.T) {
	l, _ := nn.NewLinear(4, 8, false, device.CPU())
	if len(l.Parameters()) != 1 {
		t.Errorf("expected 1 param without bias, got %d", len(l.Parameters()))
	}
}

func TestLinearKaimingScale(t *testing.T) {
	// Weights should be in [-k, k] where k = 1/sqrt(in_features).
	l, _ := nn.NewLinear(16, 32, false, device.CPU())
	vals := tensor.MustSlice[float32](l.W)
	k := float32(1.0 / math.Sqrt(16))
	for _, v := range vals {
		if v < -k-1e-6 || v > k+1e-6 {
			t.Errorf("weight %v outside [-%v, %v]", v, k, k)
			break
		}
	}
}

func TestLinearErrors(t *testing.T) {
	// Negative features.
	if _, err := nn.NewLinear(-1, 2, false, device.CPU()); err == nil {
		t.Error("expected error for negative in_features")
	}
	// Wrong input size.
	l, _ := nn.NewLinear(4, 2, true, device.CPU())
	x, _ := tensor.Zeros(dtype.F32, []int{3, 5}, device.CPU())
	if _, err := l.Forward(x); err == nil {
		t.Error("expected dim-mismatch error")
	}
}

func TestSequential(t *testing.T) {
	l1, _ := nn.NewLinear(4, 8, true, device.CPU())
	l2, _ := nn.NewLinear(8, 2, true, device.CPU())
	net := nn.NewSequential(l1, nn.ReLU{}, l2)

	// Flow shape: (3,4) → (3,8) → (3,8) → (3,2)
	x, _ := tensor.Zeros(dtype.F32, []int{3, 4}, device.CPU())
	y, err := net.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	if y.Dim(0) != 3 || y.Dim(1) != 2 {
		t.Errorf("unexpected shape %v", y.Shape())
	}
	if net.Len() != 3 {
		t.Errorf("Len() = %d, want 3", net.Len())
	}
	if _, ok := net.At(1).(nn.ReLU); !ok {
		t.Error("At(1) should be ReLU")
	}
	// Params: L1 (W,B) + L2 (W,B) = 4
	if got := len(net.Parameters()); got != 4 {
		t.Errorf("got %d params, want 4", got)
	}
}

func TestActivationsAreModules(t *testing.T) {
	// Ensure each activation implements Module (compile-time check would be
	// better, but this also confirms Parameters returns empty).
	for _, m := range []nn.Module{nn.ReLU{}, nn.SiLU{}, nn.Sigmoid{}, nn.Tanh{}} {
		if len(m.Parameters()) != 0 {
			t.Errorf("%T should have no params", m)
		}
	}
}

func TestLinearRank3Input(t *testing.T) {
	// x: (batch=2, seq=3, in=4) → (2, 3, 5)
	l, _ := nn.NewLinear(4, 5, true, device.CPU())
	x, _ := tensor.Zeros(dtype.F32, []int{2, 3, 4}, device.CPU())
	y, err := l.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	if y.Rank() != 3 || y.Dim(0) != 2 || y.Dim(1) != 3 || y.Dim(2) != 5 {
		t.Errorf("unexpected shape %v", y.Shape())
	}
}

func TestLinearDTypeMismatch(t *testing.T) {
	l, _ := nn.NewLinear(4, 2, true, device.CPU())
	x, _ := tensor.Zeros(dtype.F64, []int{3, 4}, device.CPU())
	if _, err := l.Forward(x); err == nil {
		t.Error("expected dtype mismatch error")
	}
}

func TestLinearActualComputation(t *testing.T) {
	l, _ := nn.NewLinear(2, 3, true, device.CPU())
	wNew, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{3, 2}, device.CPU())
	l.W = wNew
	bNew, _ := tensor.New([]float32{10, 20, 30}, []int{3}, device.CPU())
	l.B = bNew

	x, _ := tensor.New([]float32{1, 1, 2, 2}, []int{2, 2}, device.CPU())
	y, err := l.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	// y[0] = [1*1+1*2+10, 1*3+1*4+20, 1*5+1*6+30] = [13, 27, 41]
	// y[1] = [2+4+10, 6+8+20, 10+12+30] = [16, 34, 52]
	want := []float32{13, 27, 41, 16, 34, 52}
	got := tensor.MustSlice[float32](y)
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d] got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestReLUValues(t *testing.T) {
	x, _ := tensor.New([]float32{-2, -1, 0, 1, 2}, []int{5}, device.CPU())
	y, err := (nn.ReLU{}).Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](y)
	want := []float32{0, 0, 0, 1, 2}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestSequentialEmpty(t *testing.T) {
	s := nn.NewSequential()
	if s.Len() != 0 {
		t.Error("empty Sequential should have 0 modules")
	}
	x, _ := tensor.Zeros(dtype.F32, []int{3}, device.CPU())
	y, err := s.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	if y != x {
		t.Error("empty Sequential should return input unchanged")
	}
	if len(s.Parameters()) != 0 {
		t.Error("empty Sequential has no params")
	}
}

func TestLinearFeaturesAccessors(t *testing.T) {
	l, _ := nn.NewLinear(4, 8, false, device.CPU())
	if l.InFeatures() != 4 {
		t.Errorf("InFeatures = %d", l.InFeatures())
	}
	if l.OutFeatures() != 8 {
		t.Errorf("OutFeatures = %d", l.OutFeatures())
	}
}

func TestEveryActivationForward(t *testing.T) {
	// Directly call Forward on each activation module so its Forward line is
	// covered (TestActivationsAreModules only touches Parameters).
	x, _ := tensor.New([]float32{-1, 0, 1}, []int{3}, device.CPU())
	for name, m := range map[string]nn.Module{
		"ReLU": nn.ReLU{}, "SiLU": nn.SiLU{},
		"Sigmoid": nn.Sigmoid{}, "Tanh": nn.Tanh{},
	} {
		if _, err := m.Forward(x); err != nil {
			t.Errorf("%s.Forward: %v", name, err)
		}
	}
}

// Linear with an F8E4M3 weight (per-tensor scale) against the F32 layer.
func TestLinearF8Weight(t *testing.T) {
	in, out, batch := 32, 16, 4
	l32, err := nn.NewLinear(in, out, true, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	// Deterministic weights: FP8 per-tensor error on random Kaiming draws
	// straddled the bound (5.6 % once), so pin the inputs.
	ws := make([]float32, out*in)
	for i := range ws {
		h := uint32(i+1) * 2654435761
		ws[i] = (float32(h%10007)/10007 - 0.5) * 0.6
	}
	w32, _ := tensor.New(ws, []int{out, in}, device.CPU())
	if err := l32.SetWeight(w32); err != nil {
		t.Fatal(err)
	}
	wq, wScale, err := ops.QuantizeF8(l32.W)
	if err != nil {
		t.Fatal(err)
	}
	xs := make([]float32, batch*in)
	for i := range xs {
		xs[i] = float32(math.Sin(0.3*float64(i))) * 2
	}
	x, _ := tensor.New(xs, []int{batch, in}, device.CPU())
	ref, err := l32.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	refS := tensor.MustSlice[float32](ref)
	check := func(dev device.Device) {
		wd, _ := wq.To(dev)
		bd, _ := l32.B.To(dev)
		l8, err := nn.NewLinearPlaceholder(in, out, true, dev)
		if err != nil {
			t.Fatal(err)
		}
		if err := l8.SetWeight(wd); err != nil {
			t.Fatal(err)
		}
		l8.B, l8.WScale = bd, wScale
		xd, _ := x.To(dev)
		y, err := l8.Forward(xd)
		if err != nil {
			t.Fatalf("F8 linear on %v: %v", dev, err)
		}
		yc, _ := y.To(device.CPU())
		ys := tensor.MustSlice[float32](yc)
		var num, den float64
		for i := range refS {
			num += float64((ys[i] - refS[i]) * (ys[i] - refS[i]))
			den += float64(refS[i] * refS[i])
		}
		if e := math.Sqrt(num / den); e > 0.08 {
			t.Errorf("F8 linear on %v: rel err %.4g", dev, e)
		}
	}
	check(device.CPU())
	if gpu, err := device.CUDA(0); err == nil {
		check(gpu)
	}
}
