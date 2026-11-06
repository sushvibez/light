package dtype_test

import (
	"testing"

	"light/dtype"
)

func TestOf(t *testing.T) {
	if got, want := dtype.Of[float32](), dtype.F32; got != want {
		t.Errorf("Of[float32] = %v, want %v", got, want)
	}
	if got, want := dtype.Of[float64](), dtype.F64; got != want {
		t.Errorf("Of[float64] = %v, want %v", got, want)
	}
	if got, want := dtype.Of[int8](), dtype.I8; got != want {
		t.Errorf("Of[int8] = %v, want %v", got, want)
	}
	if got, want := dtype.Of[int16](), dtype.I16; got != want {
		t.Errorf("Of[int16] = %v, want %v", got, want)
	}
	if got, want := dtype.Of[int32](), dtype.I32; got != want {
		t.Errorf("Of[int32] = %v, want %v", got, want)
	}
	if got, want := dtype.Of[int64](), dtype.I64; got != want {
		t.Errorf("Of[int64] = %v, want %v", got, want)
	}
	if got, want := dtype.Of[uint8](), dtype.U8; got != want {
		t.Errorf("Of[uint8] = %v, want %v", got, want)
	}
	if got, want := dtype.Of[uint16](), dtype.U16; got != want {
		t.Errorf("Of[uint16] = %v, want %v", got, want)
	}
	if got, want := dtype.Of[uint32](), dtype.U32; got != want {
		t.Errorf("Of[uint32] = %v, want %v", got, want)
	}
	if got, want := dtype.Of[uint64](), dtype.U64; got != want {
		t.Errorf("Of[uint64] = %v, want %v", got, want)
	}
}
