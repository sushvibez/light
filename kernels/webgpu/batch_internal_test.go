//go:build !nowebgpu

package webgpu

import (
	"testing"

	"light/device"
	"light/internal/wgpudev"
	"light/tensor"
)

// A per-dispatch parameter created before a submit (here forced; in use, a
// full batch or a host write between creating params and dispatching)
// is re-staged into the new batch: the dispatch still sees its values.
func TestParamRestagedAcrossSubmit(t *testing.T) {
	g, err := device.WebGPU(0)
	if err != nil {
		t.Skip(err)
	}
	d, err := dev()
	if err != nil {
		t.Fatal(err)
	}
	xs := []float32{1, -2, 3, -4, 5}
	x, _ := tensor.New(xs, []int{len(xs)}, g)
	out, _ := tensor.Zeros(x.DType(), []int{len(xs)}, g)
	params, err := d.Uniform(unaryOp["neg"], uint32(len(xs)), 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	before := d.Submits
	// Record other work with its own parameters, then submit the batch.
	tmp, _ := tensor.Zeros(x.DType(), []int{len(xs)}, g)
	for range 3 {
		p, err := d.Uniform(unaryOp["abs"], uint32(len(xs)), 0, 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := run1D(d, "unary", []wgpudev.Binding{rd(raw(x)), rw(raw(tmp)), uni(p)}, len(xs)); err != nil {
			t.Fatal(err)
		}
	}
	d.Submit()
	if err := run1D(d, "unary", []wgpudev.Binding{rd(raw(x)), rw(raw(out)), uni(params)}, len(xs)); err != nil {
		t.Fatal(err)
	}
	got, _ := out.To(device.CPU())
	for i, v := range tensor.MustSlice[float32](got) {
		if v != -xs[i] {
			t.Fatalf("[%d] = %g, want %g (param slot not re-staged)", i, v, -xs[i])
		}
	}
	if d.Submits == before {
		t.Fatal("no submit happened; the test did not exercise re-staging")
	}
}
