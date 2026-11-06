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

func TestAddF32(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	b, _ := tensor.New([]float32{10, 20, 30}, []int{3}, device.CPU())
	got, err := ops.Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{11, 22, 33}
	if !slices.Equal(tensor.MustSlice[float32](got), want) {
		t.Errorf("got %v, want %v", tensor.MustSlice[float32](got), want)
	}
}

func TestAddBroadcast(t *testing.T) {
	// (2,3) + (3,) → (2,3)
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	b, _ := tensor.New([]float32{10, 20, 30}, []int{3}, device.CPU())
	got, err := ops.Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{11, 22, 33, 14, 25, 36}
	if !slices.Equal(tensor.MustSlice[float32](got), want) {
		t.Errorf("got %v, want %v", tensor.MustSlice[float32](got), want)
	}
}

func TestAddAllDTypes(t *testing.T) {
	cases := []struct {
		name string
		run  func(*testing.T)
	}{
		{"F32", func(t *testing.T) {
			a, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
			b, _ := tensor.New([]float32{3, 4}, []int{2}, device.CPU())
			c, _ := ops.Add(a, b)
			if v := tensor.MustSlice[float32](c); v[0] != 4 || v[1] != 6 {
				t.Errorf("got %v", v)
			}
		}},
		{"F64", func(t *testing.T) {
			a, _ := tensor.New([]float64{1, 2}, []int{2}, device.CPU())
			b, _ := tensor.New([]float64{3, 4}, []int{2}, device.CPU())
			c, _ := ops.Add(a, b)
			if v := tensor.MustSlice[float64](c); v[0] != 4 || v[1] != 6 {
				t.Errorf("got %v", v)
			}
		}},
		{"I32", func(t *testing.T) {
			a, _ := tensor.New([]int32{1, 2}, []int{2}, device.CPU())
			b, _ := tensor.New([]int32{3, 4}, []int{2}, device.CPU())
			c, _ := ops.Add(a, b)
			if v := tensor.MustSlice[int32](c); v[0] != 4 || v[1] != 6 {
				t.Errorf("got %v", v)
			}
		}},
		{"I64", func(t *testing.T) {
			a, _ := tensor.New([]int64{1, 2}, []int{2}, device.CPU())
			b, _ := tensor.New([]int64{3, 4}, []int{2}, device.CPU())
			c, _ := ops.Add(a, b)
			if v := tensor.MustSlice[int64](c); v[0] != 4 || v[1] != 6 {
				t.Errorf("got %v", v)
			}
		}},
		{"U8", func(t *testing.T) {
			a, _ := tensor.New([]uint8{1, 2}, []int{2}, device.CPU())
			b, _ := tensor.New([]uint8{3, 4}, []int{2}, device.CPU())
			c, _ := ops.Add(a, b)
			if v := tensor.MustSlice[uint8](c); v[0] != 4 || v[1] != 6 {
				t.Errorf("got %v", v)
			}
		}},
		{"I8", func(t *testing.T) {
			a, _ := tensor.New([]int8{1, 2}, []int{2}, device.CPU())
			b, _ := tensor.New([]int8{-5, 4}, []int{2}, device.CPU())
			c, _ := ops.Add(a, b)
			if v := tensor.MustSlice[int8](c); v[0] != -4 || v[1] != 6 {
				t.Errorf("got %v", v)
			}
		}},
	}
	for _, c := range cases {
		t.Run(c.name, c.run)
	}
}

func TestAddF16(t *testing.T) {
	// Use Ones to create two F16 tensors of 1.0, add → expect 2.0 in each cell.
	a, err := tensor.Ones(dtype.F16, []int{4}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	b, err := tensor.Ones(dtype.F16, []int{4}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	c, err := ops.Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	halves, err := tensor.ToHalfSlice(c)
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range halves {
		got := h.ToF32()
		if math.Abs(float64(got)-2.0) > 1e-3 {
			t.Errorf("F16 add: got %v, want 2.0", got)
		}
	}
}

func TestAddBF16(t *testing.T) {
	a, err := tensor.Ones(dtype.BF16, []int{4}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	b, err := tensor.Ones(dtype.BF16, []int{4}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	c, err := ops.Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	bfs, err := tensor.ToBF16Slice(c)
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range bfs {
		got := b.ToF32()
		if math.Abs(float64(got)-2.0) > 1e-2 {
			t.Errorf("BF16 add: got %v, want 2.0", got)
		}
	}
}

func TestAddMismatch(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
	b, _ := tensor.New([]float64{3, 4}, []int{2}, device.CPU())
	if _, err := ops.Add(a, b); err == nil {
		t.Error("expected dtype mismatch error")
	}
	c, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	if _, err := ops.Add(a, c); err == nil {
		t.Error("expected shape mismatch error")
	}
}
