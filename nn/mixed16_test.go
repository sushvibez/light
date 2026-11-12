package nn_test

import (
	"testing"

	"light/device"
	"light/dtype"
	"light/nn"
)

// Linears built under WithWeightStorage16Mixed are marked Mixed16 (and
// stored at 16 bits); under WithWeightStorage16 or outside a scope they
// are not.
func TestWeightStorage16MixedMarksLinears(t *testing.T) {
	dev := device.CPU()
	build := func() *nn.Linear {
		l, err := nn.NewLinearUninit(8, 4, true, dtype.F32, dev)
		if err != nil {
			t.Fatal(err)
		}
		return l
	}
	var mixed, plain *nn.Linear
	if err := nn.WithWeightStorage16Mixed(dtype.BF16, func() error { mixed = build(); return nil }); err != nil {
		t.Fatal(err)
	}
	if err := nn.WithWeightStorage16(dtype.BF16, func() error { plain = build(); return nil }); err != nil {
		t.Fatal(err)
	}
	outside := build()
	if !mixed.Mixed16 || mixed.W.DType() != dtype.BF16 {
		t.Errorf("mixed scope: Mixed16 %v, W %v", mixed.Mixed16, mixed.W.DType())
	}
	if plain.Mixed16 || plain.W.DType() != dtype.BF16 {
		t.Errorf("storage16 scope: Mixed16 %v, W %v", plain.Mixed16, plain.W.DType())
	}
	if outside.Mixed16 || outside.W.DType() != dtype.F32 {
		t.Errorf("no scope: Mixed16 %v, W %v", outside.Mixed16, outside.W.DType())
	}
}
