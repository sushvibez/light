package safetensors_test

import (
	"bytes"
	"slices"
	"testing"

	"light/device"
	"light/safetensors"
	"light/tensor"
)

// A contiguous prefix view keeps its parent's larger backing storage; Write
// must serialise only the logical bytes or every later tensor shifts (R1).
func TestWritePrefixViewKeepsLaterOffsets(t *testing.T) {
	parent, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, device.CPU())
	a, err := parent.Narrow(0, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := tensor.New([]float32{9}, []int{1}, device.CPU())
	r := buildFile(t, map[string]*tensor.Tensor{"a": a, "b": b}, nil)
	loaded, _, err := safetensors.Load(r, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if got := tensor.MustSlice[float32](loaded["a"]); !slices.Equal(got, []float32{1, 2}) {
		t.Errorf("a = %v, want [1 2]", got)
	}
	if got := tensor.MustSlice[float32](loaded["b"]); !slices.Equal(got, []float32{9}) {
		t.Errorf("b = %v, want [9]", got)
	}
}

func TestWriteRejectsReservedName(t *testing.T) {
	x, _ := tensor.New([]float32{1}, []int{1}, device.CPU())
	var buf bytes.Buffer
	if err := safetensors.Write(&buf, map[string]*tensor.Tensor{"__metadata__": x}, nil); err == nil {
		t.Fatal("Write accepted the reserved __metadata__ name")
	}
}
