package models

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"testing"

	"light/device"
	"light/tensor"
)

func readF32File(t *testing.T, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

// Qwen3.5-0.8B's vision tower against transformers' model.visual in F32
// on the processor's patches of a synthetic 320 x 256 image
// (scripts/verify/ref_qwen35_vision.py), on the CPU and WebGPU.
func TestQwen35VisionMatchesTransformers(t *testing.T) {
	const dir = "testdata/qwen35-vision/"
	raw, err := os.ReadFile(dir + "meta.json")
	if err != nil {
		t.Fatal(err)
	}
	var meta struct {
		Repo   string   `json:"repo"`
		PVS    []int    `json:"pixel_values_shape"`
		Grid   [][3]int `json:"grid_thw"`
		Merged []int    `json:"merged_shape"`
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	pv, want := readF32File(t, dir+"pixel_values.f32"), readF32File(t, dir+"merged.f32")
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		m, err := LoadQwen35VisionFromHF(meta.Repo, dev)
		if err != nil {
			t.Skip(err)
		}
		x, err := tensor.New(pv, meta.PVS, dev)
		if err != nil {
			t.Fatal(err)
		}
		y, err := m.Forward(x, meta.Grid)
		if err != nil {
			t.Fatal(err)
		}
		if y, err = y.To(device.CPU()); err != nil {
			t.Fatal(err)
		}
		got := tensor.MustSlice[float32](y)
		if len(got) != len(want) {
			t.Fatalf("%v: %d values, want %d", dev, len(got), len(want))
		}
		md, ma := 0.0, 0.0
		for i := range got {
			md = math.Max(md, math.Abs(float64(got[i]-want[i])))
			ma = math.Max(ma, math.Abs(float64(want[i])))
		}
		t.Logf("%v: max |light - transformers| %.3g (max |ref| %.3g) over %d values", dev, md, ma, len(got))
		// The residual stream reaches ~2100 in the last blocks (ViT massive
		// activations), where F32 summation order alone moves values by
		// ~1e-4 relative; block 0 agrees to 1.3e-5 and the merged tokens
		// (|ref| <= 3.8) to ~1e-3.
		if md > 2e-3 {
			t.Errorf("%v: merged tokens differ by %.3g", dev, md)
		}
	}
}
