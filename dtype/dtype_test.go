package dtype_test

import (
	"testing"

	"light/dtype"
)

func TestSize(t *testing.T) {
	cases := []struct {
		dt   dtype.DType
		want int
	}{
		{dtype.F16, 2},
		{dtype.BF16, 2},
		{dtype.F32, 4},
		{dtype.F64, 8},
		{dtype.U8, 1},
		{dtype.I8, 1},
		{dtype.U16, 2},
		{dtype.I16, 2},
		{dtype.U32, 4},
		{dtype.I32, 4},
		{dtype.U64, 8},
		{dtype.I64, 8},
		{dtype.Q4_0, 0}, // quantized → 0 (use ByteSize instead)
		{dtype.Q4_K, 0},
	}
	for _, c := range cases {
		if got := c.dt.Size(); got != c.want {
			t.Errorf("%v.Size() = %d, want %d", c.dt, got, c.want)
		}
	}
}

func TestString(t *testing.T) {
	cases := map[dtype.DType]string{
		dtype.F32:  "F32",
		dtype.BF16: "BF16",
		dtype.I64:  "I64",
		dtype.Q8_0: "Q8_0",
		dtype.Q4_K: "Q4_K",
	}
	for dt, want := range cases {
		if got := dt.String(); got != want {
			t.Errorf("%v.String() = %q, want %q", dt, got, want)
		}
	}
	// Invalid value must produce *something* that doesn't panic.
	var bad dtype.DType = 250
	if s := bad.String(); s == "" {
		t.Errorf("expected non-empty string for invalid dtype, got %q", s)
	}
}

func TestIsFloat(t *testing.T) {
	floats := []dtype.DType{dtype.F16, dtype.BF16, dtype.F32, dtype.F64}
	for _, d := range floats {
		if !d.IsFloat() {
			t.Errorf("%v should be float", d)
		}
	}
	notFloats := []dtype.DType{dtype.U8, dtype.I8, dtype.I32, dtype.U32, dtype.Q4_0, dtype.Q8_K}
	for _, d := range notFloats {
		if d.IsFloat() {
			t.Errorf("%v should not be float", d)
		}
	}
}

func TestIsInt(t *testing.T) {
	ints := []dtype.DType{dtype.U8, dtype.I8, dtype.U16, dtype.I16, dtype.U32, dtype.I32, dtype.U64, dtype.I64}
	for _, d := range ints {
		if !d.IsInt() {
			t.Errorf("%v should be int", d)
		}
	}
	notInts := []dtype.DType{dtype.F32, dtype.F64, dtype.F16, dtype.BF16, dtype.Q4_0}
	for _, d := range notInts {
		if d.IsInt() {
			t.Errorf("%v should not be int", d)
		}
	}
}

func TestIsSigned(t *testing.T) {
	signed := []dtype.DType{dtype.F16, dtype.BF16, dtype.F32, dtype.F64, dtype.I8, dtype.I16, dtype.I32, dtype.I64}
	for _, d := range signed {
		if !d.IsSigned() {
			t.Errorf("%v should be signed", d)
		}
	}
	unsigned := []dtype.DType{dtype.U8, dtype.U16, dtype.U32, dtype.U64}
	for _, d := range unsigned {
		if d.IsSigned() {
			t.Errorf("%v should be unsigned", d)
		}
	}
}

func TestIsQuantized(t *testing.T) {
	for _, d := range []dtype.DType{dtype.Q4_0, dtype.Q4_1, dtype.Q5_0, dtype.Q5_1, dtype.Q8_0, dtype.Q8_1, dtype.Q2_K, dtype.Q3_K, dtype.Q4_K, dtype.Q5_K, dtype.Q6_K, dtype.Q8_K} {
		if !d.IsQuantized() {
			t.Errorf("%v should be quantized", d)
		}
	}
	for _, d := range []dtype.DType{dtype.F16, dtype.F32, dtype.I32, dtype.U8} {
		if d.IsQuantized() {
			t.Errorf("%v should not be quantized", d)
		}
	}
}

func TestIsQuantizedExpanded(t *testing.T) {
	// All Kquants
	for _, d := range []dtype.DType{dtype.Q2_K, dtype.Q3_K, dtype.Q4_K, dtype.Q5_K, dtype.Q6_K, dtype.Q8_K} {
		if !d.IsQuantized() {
			t.Errorf("%v should be quantized", d)
		}
	}
}

func TestByteSize(t *testing.T) {
	// Scalar
	if got := dtype.F32.ByteSize(100); got != 400 {
		t.Errorf("F32.ByteSize(100) = %d, want 400", got)
	}
	if got := dtype.I64.ByteSize(10); got != 80 {
		t.Errorf("I64.ByteSize(10) = %d, want 80", got)
	}
	// Quantized: 32 elems per block, 18 bytes per block for Q4_0
	if got := dtype.Q4_0.ByteSize(32); got != 18 {
		t.Errorf("Q4_0.ByteSize(32) = %d, want 18", got)
	}
	// Rounds up to whole block
	if got := dtype.Q4_0.ByteSize(33); got != 36 { // 2 blocks
		t.Errorf("Q4_0.ByteSize(33) = %d, want 36", got)
	}
	// Zero
	if got := dtype.F32.ByteSize(0); got != 0 {
		t.Errorf("F32.ByteSize(0) = %d, want 0", got)
	}
	// Negative n — guarded
	if got := dtype.F32.ByteSize(-1); got != 0 {
		t.Errorf("F32.ByteSize(-1) = %d, want 0", got)
	}
}

func TestBlockSizeAndBytes(t *testing.T) {
	if got := dtype.F32.BlockElems(); got != 1 {
		t.Errorf("F32.BlockElems() = %d, want 1", got)
	}
	if got := dtype.Q4_0.BlockElems(); got != 32 {
		t.Errorf("Q4_0.BlockElems() = %d, want 32", got)
	}
	if got := dtype.Q4_K.BlockElems(); got != 256 {
		t.Errorf("Q4K.BlockElems() = %d, want 256", got)
	}
	if got := dtype.F64.BlockBytes(); got != 8 {
		t.Errorf("F64.BlockBytes() = %d, want 8", got)
	}
	if got := dtype.Q8_0.BlockBytes(); got != 34 {
		t.Errorf("Q8_0.BlockBytes() = %d, want 34", got)
	}
}

func TestValid(t *testing.T) {
	if !dtype.F32.Valid() {
		t.Errorf("F32 should be valid")
	}
	var bad dtype.DType = 250
	if bad.Valid() {
		t.Errorf("250 should not be valid")
	}
}
