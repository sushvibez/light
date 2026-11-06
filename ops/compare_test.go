package ops_test

import (
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

func TestCompareLT(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, device.CPU())
	b, _ := tensor.New([]float32{2, 2, 2, 2}, []int{4}, device.CPU())
	out, err := ops.LT(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if out.DType() != dtype.U8 {
		t.Errorf("dtype = %v", out.DType())
	}
	if !slices.Equal(tensor.MustSlice[uint8](out), []uint8{1, 0, 0, 0}) {
		t.Errorf("got %v", tensor.MustSlice[uint8](out))
	}
}

func TestCompareLEGTGEEQNE(t *testing.T) {
	a, _ := tensor.New([]int32{1, 2, 3}, []int{3}, device.CPU())
	b, _ := tensor.New([]int32{2, 2, 2}, []int{3}, device.CPU())
	cases := map[string]struct {
		fn   func(*tensor.Tensor, *tensor.Tensor) (*tensor.Tensor, error)
		want []uint8
	}{
		"LE": {ops.LE, []uint8{1, 1, 0}},
		"GT": {ops.GT, []uint8{0, 0, 1}},
		"GE": {ops.GE, []uint8{0, 1, 1}},
		"EQ": {ops.EQ, []uint8{0, 1, 0}},
		"NE": {ops.NE, []uint8{1, 0, 1}},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			got, err := c.fn(a, b)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(tensor.MustSlice[uint8](got), c.want) {
				t.Errorf("%s: got %v, want %v", name, tensor.MustSlice[uint8](got), c.want)
			}
		})
	}
}

func TestCompareBroadcast(t *testing.T) {
	// (2, 3) < scalar (1,)
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	b, _ := tensor.New([]float32{3}, []int{1}, device.CPU())
	out, _ := ops.LT(a, b)
	if !slices.Equal(tensor.MustSlice[uint8](out), []uint8{1, 1, 0, 0, 0, 0}) {
		t.Errorf("got %v", tensor.MustSlice[uint8](out))
	}
}

func TestCompareErrors(t *testing.T) {
	a, _ := tensor.New([]float32{1}, []int{1}, device.CPU())
	b, _ := tensor.New([]float64{1}, []int{1}, device.CPU())
	if _, err := ops.LT(a, b); err == nil {
		t.Error("expected dtype mismatch")
	}
}

func TestLogicalAndOrNot(t *testing.T) {
	a, _ := tensor.New([]uint8{1, 0, 1, 0}, []int{4}, device.CPU())
	b, _ := tensor.New([]uint8{1, 1, 0, 0}, []int{4}, device.CPU())
	andOut, _ := ops.LogicalAnd(a, b)
	if !slices.Equal(tensor.MustSlice[uint8](andOut), []uint8{1, 0, 0, 0}) {
		t.Errorf("AND: %v", tensor.MustSlice[uint8](andOut))
	}
	orOut, _ := ops.LogicalOr(a, b)
	if !slices.Equal(tensor.MustSlice[uint8](orOut), []uint8{1, 1, 1, 0}) {
		t.Errorf("OR: %v", tensor.MustSlice[uint8](orOut))
	}
	notOut, _ := ops.LogicalNot(a)
	if !slices.Equal(tensor.MustSlice[uint8](notOut), []uint8{0, 1, 0, 1}) {
		t.Errorf("NOT: %v", tensor.MustSlice[uint8](notOut))
	}
}

func TestLogicalErrors(t *testing.T) {
	a, _ := tensor.New([]float32{1, 0}, []int{2}, device.CPU())
	b, _ := tensor.New([]uint8{1, 0}, []int{2}, device.CPU())
	if _, err := ops.LogicalAnd(a, b); err == nil {
		t.Error("expected dtype error")
	}
	if _, err := ops.LogicalNot(a); err == nil {
		t.Error("expected dtype error")
	}
}

func TestWhere(t *testing.T) {
	// cond = [1,0,1,0], a = [10,20,30,40], b = [1,2,3,4] → [10,2,30,4]
	cond, _ := tensor.New([]uint8{1, 0, 1, 0}, []int{4}, device.CPU())
	a, _ := tensor.New([]float32{10, 20, 30, 40}, []int{4}, device.CPU())
	b, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, device.CPU())
	out, err := ops.Where(cond, a, b)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{10, 2, 30, 4}
	if !slices.Equal(tensor.MustSlice[float32](out), want) {
		t.Errorf("got %v, want %v", tensor.MustSlice[float32](out), want)
	}
}

func TestWhereBroadcast(t *testing.T) {
	// cond scalar (true) → all of a
	cond, _ := tensor.New([]uint8{1}, []int{1}, device.CPU())
	a, _ := tensor.New([]float32{10, 20, 30}, []int{3}, device.CPU())
	b, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	out, _ := ops.Where(cond, a, b)
	if !slices.Equal(tensor.MustSlice[float32](out), []float32{10, 20, 30}) {
		t.Errorf("got %v", tensor.MustSlice[float32](out))
	}
}

func TestWhereErrors(t *testing.T) {
	cond, _ := tensor.New([]float32{1}, []int{1}, device.CPU())
	a, _ := tensor.New([]float32{1}, []int{1}, device.CPU())
	b, _ := tensor.New([]float32{2}, []int{1}, device.CPU())
	if _, err := ops.Where(cond, a, b); err == nil {
		t.Error("expected cond-dtype error")
	}
}

func TestNewTrigAndRounding(t *testing.T) {
	x, _ := tensor.New([]float32{0, 1, 2.5, -3.7}, []int{4}, device.CPU())
	// Just verify no errors and sensible shapes.
	for _, fn := range []func(*tensor.Tensor) (*tensor.Tensor, error){
		ops.Tan, ops.Floor, ops.Ceil, ops.Round,
	} {
		y, err := fn(x)
		if err != nil {
			t.Fatal(err)
		}
		if y.Dim(0) != 4 {
			t.Errorf("shape = %v", y.Shape())
		}
	}
}

func TestFloorCeilRoundValues(t *testing.T) {
	x, _ := tensor.New([]float32{1.9, -1.9, 0.5, -0.5, 2.5}, []int{5}, device.CPU())
	f, _ := ops.Floor(x)
	if !slices.Equal(tensor.MustSlice[float32](f), []float32{1, -2, 0, -1, 2}) {
		t.Errorf("Floor: %v", tensor.MustSlice[float32](f))
	}
	c, _ := ops.Ceil(x)
	if !slices.Equal(tensor.MustSlice[float32](c), []float32{2, -1, 1, 0, 3}) {
		t.Errorf("Ceil: %v", tensor.MustSlice[float32](c))
	}
	r, _ := ops.Round(x)
	// Go's math.Round rounds half away from zero: 0.5→1, -0.5→-1, 2.5→3
	if !slices.Equal(tensor.MustSlice[float32](r), []float32{2, -2, 1, -1, 3}) {
		t.Errorf("Round: %v", tensor.MustSlice[float32](r))
	}
}

func TestProdReduction(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, device.CPU())
	p, err := ops.Prod(a, ops.ReduceOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if tensor.MustSlice[float32](p)[0] != 24 {
		t.Errorf("Prod = %v, want 24", tensor.MustSlice[float32](p)[0])
	}
}

func TestL2Norm(t *testing.T) {
	a, _ := tensor.New([]float32{3, 4}, []int{2}, device.CPU())
	n, err := ops.L2Norm(a, ops.ReduceOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if v := tensor.MustSlice[float32](n)[0]; v < 4.99 || v > 5.01 {
		t.Errorf("L2Norm([3,4]) = %v, want 5", v)
	}
}

func TestAllAny(t *testing.T) {
	a, _ := tensor.New([]uint8{1, 1, 1, 0, 0, 0, 1, 0, 1}, []int{3, 3}, device.CPU())
	all0, _ := ops.All(a, ops.ReduceOptions{Axes: []int{1}})
	want := []uint8{1, 0, 0}
	if !slices.Equal(tensor.MustSlice[uint8](all0), want) {
		t.Errorf("All(axis=1): %v", tensor.MustSlice[uint8](all0))
	}
	any0, _ := ops.Any(a, ops.ReduceOptions{Axes: []int{1}})
	want = []uint8{1, 0, 1}
	if !slices.Equal(tensor.MustSlice[uint8](any0), want) {
		t.Errorf("Any(axis=1): %v", tensor.MustSlice[uint8](any0))
	}
}

func TestCompareWhereCUDADtypes(t *testing.T) {
	gpu := cudaOrSkip(t)
	a, _ := tensor.New([]float32{1, 5, -2, 0, 3, 7}, []int{6}, device.CPU())
	b, _ := tensor.New([]float32{1, 2, 2, 2, 3, 2}, []int{6}, device.CPU())
	for _, dt := range []dtype.DType{dtype.F64, dtype.I32, dtype.I64, dtype.U8, dtype.BF16} {
		ac, _ := ops.Cast(a, dt)
		bc, _ := ops.Cast(b, dt)
		ag, _ := ac.To(gpu)
		bg, _ := bc.To(gpu)
		for name, fn := range map[string]func(x, y *tensor.Tensor) (*tensor.Tensor, error){
			"lt": ops.LT, "le": ops.LE, "gt": ops.GT, "ge": ops.GE, "eq": ops.EQ, "ne": ops.NE,
		} {
			c, err := fn(ac, bc)
			if err != nil {
				t.Fatalf("%s %v cpu: %v", name, dt, err)
			}
			g, err := fn(ag, bg)
			if err != nil {
				t.Fatalf("%s %v cuda: %v", name, dt, err)
			}
			expectRow(t, name+" "+dt.String(), f32s(t, g), f32s(t, c), 0)
		}
		if dt == dtype.F64 {
			continue // where_f64 already existed
		}
		cond, _ := ops.GT(ac, bc)
		wc, err := ops.Where(cond, ac, bc)
		if err != nil {
			t.Fatal(err)
		}
		condG, _ := cond.To(gpu)
		wg, err := ops.Where(condG, ag, bg)
		if err != nil {
			t.Fatalf("where %v cuda: %v", dt, err)
		}
		expectRow(t, "where "+dt.String(), f32s(t, wg), f32s(t, wc), 0)
	}
}
