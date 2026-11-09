package models

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"light/device"
	"light/tensor"
)

// A seeded tiny-random FluxTransformer2DModel (2 double + 2 single blocks,
// guidance embedder, 3-axis RoPE; scripts/verify/ref_flux.py) against
// diffusers' forward pass, on the CPU and WebGPU.
func TestFluxTransformerTinyMatchesDiffusers(t *testing.T) {
	raw, err := os.ReadFile("testdata/refs/flux-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		X        []float32 `json:"x"`
		Txt      []float32 `json:"txt"`
		Pooled   []float32 `json:"pooled"`
		ImgIDs   []float32 `json:"img_ids"`
		TxtIDs   []float32 `json:"txt_ids"`
		Out      []float32 `json:"out"`
		XShape   []int     `json:"x_shape"`
		TxtShape []int     `json:"txt_shape"`
		Timestep []float32 `json:"timestep"`
		Guidance []float32 `json:"guidance"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		m, err := LoadFluxTransformer("testdata/flux-tiny", dev)
		if err != nil {
			t.Fatal(err)
		}
		mk := func(v []float32, dims ...int) *tensor.Tensor {
			x, err := tensor.New(v, dims, dev)
			if err != nil {
				t.Fatal(err)
			}
			return x
		}
		y, err := m.Forward(FluxInputs{
			Image:    mk(ref.X, ref.XShape...),
			Text:     mk(ref.Txt, ref.TxtShape...),
			Pooled:   mk(ref.Pooled, 1, len(ref.Pooled)),
			Timestep: ref.Timestep, Guidance: ref.Guidance,
			ImgIDs: ref.ImgIDs, TxtIDs: ref.TxtIDs,
		})
		if err != nil {
			t.Fatal(err)
		}
		if y, err = y.To(device.CPU()); err != nil {
			t.Fatal(err)
		}
		got := tensor.MustSlice[float32](y)
		if len(got) != len(ref.Out) {
			t.Fatalf("%v: %d values, want %d", dev, len(got), len(ref.Out))
		}
		md := 0.0
		for i := range got {
			md = math.Max(md, math.Abs(float64(got[i]-ref.Out[i])))
		}
		t.Logf("%v: max |light - diffusers| %.3g over %d values", dev, md, len(got))
		if md > 1e-4 {
			t.Errorf("%v: output differs by %.3g", dev, md)
		}
	}
}
