package tensor_test

import (
	"math"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

func TestZeros(t *testing.T) {
	tt, err := tensor.Zeros(dtype.F32, []int{2, 3}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if tt.Rank() != 2 {
		t.Errorf("Rank = %d, want 2", tt.Rank())
	}
	if tt.NumElements() != 6 {
		t.Errorf("NumElements = %d", tt.NumElements())
	}
	if tt.DType() != dtype.F32 {
		t.Errorf("DType = %v", tt.DType())
	}
	vals, err := tensor.ToSlice[float32](tt)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range vals {
		if v != 0 {
			t.Fatalf("zeros has non-zero: %v", vals)
		}
	}
}

func TestOnes(t *testing.T) {
	tt, err := tensor.Ones(dtype.F32, []int{4}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	vals := tensor.MustSlice[float32](tt)
	for _, v := range vals {
		if v != 1.0 {
			t.Fatalf("expected all ones, got %v", vals)
		}
	}
}

func TestOnesQuantizedRejected(t *testing.T) {
	if _, err := tensor.Ones(dtype.Q4_0, []int{32}, device.CPU()); err == nil {
		t.Error("expected error for quantized Ones")
	}
}

func TestFull(t *testing.T) {
	tt, err := tensor.Full[float64](3.5, []int{3}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	vals := tensor.MustSlice[float64](tt)
	want := []float64{3.5, 3.5, 3.5}
	if !slices.Equal(vals, want) {
		t.Errorf("got %v, want %v", vals, want)
	}
}

func TestNew(t *testing.T) {
	tt, err := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	vals := tensor.MustSlice[float32](tt)
	want := []float32{1, 2, 3, 4, 5, 6}
	if !slices.Equal(vals, want) {
		t.Errorf("got %v, want %v", vals, want)
	}
}

func TestNewMismatchedLen(t *testing.T) {
	if _, err := tensor.New([]float32{1, 2, 3}, []int{2, 3}, device.CPU()); err == nil {
		t.Error("expected error for mismatched length")
	}
}

func TestToSliceDTypeMismatch(t *testing.T) {
	tt, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
	if _, err := tensor.ToSlice[int64](tt); err == nil {
		t.Error("expected error on dtype mismatch")
	}
}

func TestReshape(t *testing.T) {
	tt, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{6}, device.CPU())
	r, err := tt.Reshape(2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if r.Rank() != 2 || r.Dim(0) != 2 || r.Dim(1) != 3 {
		t.Errorf("unexpected shape %v", r.Shape())
	}
	vals := tensor.MustSlice[float32](r)
	want := []float32{1, 2, 3, 4, 5, 6}
	if !slices.Equal(vals, want) {
		t.Errorf("got %v", vals)
	}
}

func TestReshapeInfer(t *testing.T) {
	tt, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{6}, device.CPU())
	r, err := tt.Reshape(2, -1)
	if err != nil {
		t.Fatal(err)
	}
	if r.Dim(1) != 3 {
		t.Errorf("inferred dim = %d, want 3", r.Dim(1))
	}
}

func TestReshapeErrors(t *testing.T) {
	tt, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{6}, device.CPU())
	// Too few elements.
	if _, err := tt.Reshape(2, 4); err == nil {
		t.Error("expected size-mismatch error")
	}
	// Two -1.
	if _, err := tt.Reshape(-1, -1); err == nil {
		t.Error("expected double-infer error")
	}
	// Negative (non -1) dim.
	if _, err := tt.Reshape(-2, 3); err == nil {
		t.Error("expected negative dim error")
	}
}

func TestPermuteAndTranspose(t *testing.T) {
	tt, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	p, err := tt.Transpose(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if p.Dim(0) != 3 || p.Dim(1) != 2 {
		t.Errorf("permuted shape = %v, want (3,2)", p.Shape())
	}
	if p.IsContiguous() {
		t.Error("transposed tensor should not be contiguous")
	}
	c, err := p.Contiguous()
	if err != nil {
		t.Fatal(err)
	}
	if !c.IsContiguous() {
		t.Error("Contiguous should produce contiguous tensor")
	}
	vals := tensor.MustSlice[float32](c)
	// Original: [[1,2,3],[4,5,6]]; transpose: [[1,4],[2,5],[3,6]] → [1,4,2,5,3,6]
	want := []float32{1, 4, 2, 5, 3, 6}
	if !slices.Equal(vals, want) {
		t.Errorf("got %v, want %v", vals, want)
	}
}

func TestPermuteErrors(t *testing.T) {
	tt, _ := tensor.New([]float32{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	if _, err := tt.Permute(0); err == nil {
		t.Error("expected length error")
	}
	if _, err := tt.Permute(0, 0); err == nil {
		t.Error("expected duplicate error")
	}
	if _, err := tt.Permute(0, 5); err == nil {
		t.Error("expected out-of-range error")
	}
}

func TestReshapeNonContiguous(t *testing.T) {
	tt, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	p, _ := tt.Transpose(0, 1)
	if _, err := p.Reshape(6); err == nil {
		t.Error("Reshape should reject non-contiguous input")
	}
}

func TestContiguousOnContiguous(t *testing.T) {
	tt, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	c, err := tt.Contiguous()
	if err != nil {
		t.Fatal(err)
	}
	if c != tt {
		t.Error("Contiguous on already-contiguous should return receiver")
	}
}

func TestArange(t *testing.T) {
	tt, err := tensor.Arange(0, 10, 2, dtype.F32, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	vals := tensor.MustSlice[float32](tt)
	want := []float32{0, 2, 4, 6, 8}
	if !slices.Equal(vals, want) {
		t.Errorf("got %v, want %v", vals, want)
	}
	// Zero step
	if _, err := tensor.Arange(0, 10, 0, dtype.F32, device.CPU()); err == nil {
		t.Error("expected zero-step error")
	}
	// Unsupported dtype
	if _, err := tensor.Arange(0, 10, 1, dtype.I32, device.CPU()); err == nil {
		t.Error("expected unsupported dtype error in Phase 1")
	}
}

func TestRand(t *testing.T) {
	tt, err := tensor.Rand(dtype.F32, []int{100}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	vals := tensor.MustSlice[float32](tt)
	// All in [0,1), some variance
	mean := float32(0)
	for _, v := range vals {
		if v < 0 || v >= 1 {
			t.Fatalf("Rand value out of range: %v", v)
		}
		mean += v
	}
	mean /= float32(len(vals))
	if math.Abs(float64(mean)-0.5) > 0.15 {
		t.Errorf("mean = %v, expected around 0.5", mean)
	}

	// Unsupported dtype.
	if _, err := tensor.Rand(dtype.I32, []int{10}, device.CPU()); err == nil {
		t.Error("expected unsupported dtype error")
	}
}

func TestRequiresGrad(t *testing.T) {
	tt, _ := tensor.Zeros(dtype.F32, []int{2}, device.CPU())
	if tt.RequiresGrad() {
		t.Error("default should be false")
	}
	tt.SetRequiresGrad(true)
	if !tt.RequiresGrad() {
		t.Error("SetRequiresGrad(true) not reflected")
	}
}

func TestGradAccessors(t *testing.T) {
	tt, _ := tensor.Zeros(dtype.F32, []int{2}, device.CPU())
	if tt.Grad() != nil {
		t.Error("initial Grad should be nil")
	}
	g, _ := tensor.Ones(dtype.F32, []int{2}, device.CPU())
	tt.SetGrad(g)
	if tt.Grad() != g {
		t.Error("Grad setter")
	}
	tt.SetGradFn(nil, 0)
	if tt.GradFn() != nil {
		t.Error("GradFn should be nil after set to nil")
	}
}

func TestToSliceNonCPU(t *testing.T) {
	dev, err := device.CUDA(0)
	if err != nil {
		t.Skip("cuda device unavailable")
	}
	_ = dev
}

func TestFullAllDTypes(t *testing.T) {
	// Full[T] should work for every numeric T.
	// F32
	tt, err := tensor.Full[float32](2.5, []int{3}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if v := tensor.MustSlice[float32](tt); v[0] != 2.5 {
		t.Errorf("F32 full: %v", v)
	}
	// F64
	tf64, _ := tensor.Full[float64](3.25, []int{2}, device.CPU())
	if v := tensor.MustSlice[float64](tf64); v[0] != 3.25 {
		t.Errorf("F64 full: %v", v)
	}
	// I8
	ti8, _ := tensor.Full[int8](-7, []int{2}, device.CPU())
	if v := tensor.MustSlice[int8](ti8); v[0] != -7 {
		t.Errorf("I8 full: %v", v)
	}
	// U8
	tu8, _ := tensor.Full[uint8](200, []int{2}, device.CPU())
	if v := tensor.MustSlice[uint8](tu8); v[0] != 200 {
		t.Errorf("U8 full: %v", v)
	}
	// I32, I64, U32, U64, I16, U16
	for _, c := range []struct {
		name string
		run  func(*testing.T)
	}{
		{"i16", func(t *testing.T) {
			x, _ := tensor.Full[int16](-5, []int{2}, device.CPU())
			if v := tensor.MustSlice[int16](x); v[0] != -5 {
				t.Error()
			}
		}},
		{"u16", func(t *testing.T) {
			x, _ := tensor.Full[uint16](500, []int{2}, device.CPU())
			if v := tensor.MustSlice[uint16](x); v[0] != 500 {
				t.Error()
			}
		}},
		{"i32", func(t *testing.T) {
			x, _ := tensor.Full[int32](-12345, []int{2}, device.CPU())
			if v := tensor.MustSlice[int32](x); v[0] != -12345 {
				t.Error()
			}
		}},
		{"u32", func(t *testing.T) {
			x, _ := tensor.Full[uint32](1<<20, []int{2}, device.CPU())
			if v := tensor.MustSlice[uint32](x); v[0] != 1<<20 {
				t.Error()
			}
		}},
		{"i64", func(t *testing.T) {
			x, _ := tensor.Full[int64](-1<<40, []int{2}, device.CPU())
			if v := tensor.MustSlice[int64](x); v[0] != -1<<40 {
				t.Error()
			}
		}},
		{"u64", func(t *testing.T) {
			x, _ := tensor.Full[uint64](1<<50, []int{2}, device.CPU())
			if v := tensor.MustSlice[uint64](x); v[0] != 1<<50 {
				t.Error()
			}
		}},
	} {
		t.Run(c.name, c.run)
	}
}

func TestOnesAllSupportedDTypes(t *testing.T) {
	// Ones goes through fillScalar — exercise all dtypes fillScalar handles.
	for _, dt := range []dtype.DType{dtype.F32, dtype.F64, dtype.I32, dtype.I64, dtype.U8, dtype.I8, dtype.F16, dtype.BF16} {
		tt, err := tensor.Ones(dt, []int{4}, device.CPU())
		if err != nil {
			t.Errorf("Ones(%v) error: %v", dt, err)
			continue
		}
		// Check byte buffer size.
		if tt.NumElements() != 4 {
			t.Errorf("%v NumElements wrong", dt)
		}
	}
}

func TestOnesUnsupportedDType(t *testing.T) {
	// U16/U32/U64/I16 are not in fillScalar (Phase 1 scope).
	if _, err := tensor.Ones(dtype.U16, []int{2}, device.CPU()); err == nil {
		t.Error("expected unsupported dtype error")
	}
}

func TestNewOnCUDA(t *testing.T) {
	dev, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	tt, err := tensor.New([]float32{1, 2, 3}, []int{3}, dev)
	if err != nil {
		t.Fatal(err)
	}
	if tt.Device().Kind() != device.KindCUDA {
		t.Errorf("expected CUDA, got %v", tt.Device())
	}
	// Round-trip back to CPU and verify values.
	back, err := tt.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	vals := tensor.MustSlice[float32](back)
	if len(vals) != 3 || vals[0] != 1 || vals[2] != 3 {
		t.Errorf("round-trip: %v", vals)
	}
}

func TestFromBytesF32(t *testing.T) {
	// Little-endian F32: [1.0, 2.0] → 8 bytes
	data := []byte{0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40}
	tt, err := tensor.FromBytes(data, dtype.F32, []int{2}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	vals := tensor.MustSlice[float32](tt)
	if vals[0] != 1.0 || vals[1] != 2.0 {
		t.Errorf("got %v, want [1 2]", vals)
	}
}

func TestFromBytesSizeMismatch(t *testing.T) {
	data := []byte{1, 2, 3}
	if _, err := tensor.FromBytes(data, dtype.F32, []int{2}, device.CPU()); err == nil {
		t.Error("expected size mismatch error")
	}
}

func TestNarrow(t *testing.T) {
	tt, _ := tensor.New([]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, []int{3, 4}, device.CPU())
	n, err := tt.Narrow(1, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if n.Dim(0) != 3 || n.Dim(1) != 2 {
		t.Errorf("shape = %v", n.Shape())
	}
	c, err := n.Contiguous()
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 2, 5, 6, 9, 10}
	got := tensor.MustSlice[float32](c)
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestNarrowErrors(t *testing.T) {
	tt, _ := tensor.New([]float32{0, 1, 2, 3}, []int{2, 2}, device.CPU())
	if _, err := tt.Narrow(5, 0, 1); err == nil {
		t.Error("expected dim-out-of-range")
	}
	if _, err := tt.Narrow(0, -1, 1); err == nil {
		t.Error("expected start<0 error")
	}
	if _, err := tt.Narrow(0, 0, -1); err == nil {
		t.Error("expected length<0 error")
	}
	if _, err := tt.Narrow(0, 1, 5); err == nil {
		t.Error("expected out-of-range error")
	}
}

func TestTensorAccessors(t *testing.T) {
	tt, _ := tensor.New([]float32{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	if s := tt.Strides(); s.Rank() != 2 {
		t.Errorf("Strides rank = %d", s.Rank())
	}
	if tt.Offset() != 0 {
		t.Errorf("Offset = %d", tt.Offset())
	}
	if tt.Storage() == nil {
		t.Error("Storage() nil")
	}
}

func TestRegisterCUDAContigNoop(t *testing.T) {
	tensor.RegisterCUDAContig(func(dst, src *tensor.Tensor) error { return nil })
}

func TestRandNInTensorPkg(t *testing.T) {
	tt, err := tensor.RandN(dtype.F32, []int{100}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if tt.DType() != dtype.F32 || tt.NumElements() != 100 {
		t.Errorf("dtype=%v n=%d", tt.DType(), tt.NumElements())
	}
	// Also cover F64 + unsupported path.
	_, err = tensor.RandN(dtype.F64, []int{10}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tensor.RandN(dtype.I32, []int{10}, device.CPU()); err == nil {
		t.Error("expected unsupported dtype error")
	}
	// Empty shape path
	_, err = tensor.RandN(dtype.F32, []int{0}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
}

func TestToMethod(t *testing.T) {
	tt, _ := tensor.New([]float32{1, 2}, []int{2}, device.CPU())
	same, err := tt.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if same != tt {
		t.Error("To(same device) should be a no-op")
	}
}

func TestFromBytesToCUDA(t *testing.T) {
	dev, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	data := []byte{0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40}
	tt, err := tensor.FromBytes(data, dtype.F32, []int{2}, dev)
	if err != nil {
		t.Fatal(err)
	}
	if tt.Device().Kind() != device.KindCUDA {
		t.Error("expected CUDA tensor")
	}
	back, _ := tt.To(device.CPU())
	vals := tensor.MustSlice[float32](back)
	if vals[0] != 1.0 || vals[1] != 2.0 {
		t.Errorf("got %v, want [1 2]", vals)
	}
}

func TestZerosOnCUDA(t *testing.T) {
	dev, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	tt, err := tensor.Zeros(dtype.F32, []int{4}, dev)
	if err != nil {
		t.Fatal(err)
	}
	if tt.Device().Kind() != device.KindCUDA {
		t.Errorf("expected CUDA, got %v", tt.Device())
	}
	// Zeros on CUDA should memset to 0 — verify by downloading.
	back, _ := tt.To(device.CPU())
	vals := tensor.MustSlice[float32](back)
	for _, v := range vals {
		if v != 0 {
			t.Errorf("not zero: %v", vals)
			break
		}
	}
}

func TestToHalfSlice(t *testing.T) {
	tt, err := tensor.Ones(dtype.F16, []int{3}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	hs, err := tensor.ToHalfSlice(tt)
	if err != nil {
		t.Fatal(err)
	}
	if len(hs) != 3 {
		t.Errorf("len = %d", len(hs))
	}
	for _, h := range hs {
		if math.Abs(float64(h.ToF32())-1.0) > 1e-3 {
			t.Errorf("ToHalfSlice value = %v", h.ToF32())
		}
	}

	// Wrong dtype rejected
	tf32, _ := tensor.Ones(dtype.F32, []int{3}, device.CPU())
	if _, err := tensor.ToHalfSlice(tf32); err == nil {
		t.Error("expected dtype error")
	}
}

func TestToBF16Slice(t *testing.T) {
	tt, err := tensor.Ones(dtype.BF16, []int{3}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	bs, err := tensor.ToBF16Slice(tt)
	if err != nil {
		t.Fatal(err)
	}
	if len(bs) != 3 {
		t.Errorf("len = %d", len(bs))
	}
	for _, b := range bs {
		if math.Abs(float64(b.ToF32())-1.0) > 1e-2 {
			t.Errorf("value = %v", b.ToF32())
		}
	}

	// Wrong dtype rejected
	tf32, _ := tensor.Ones(dtype.F32, []int{3}, device.CPU())
	if _, err := tensor.ToBF16Slice(tf32); err == nil {
		t.Error("expected dtype error")
	}
}
