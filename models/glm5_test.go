package models

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"light/device"
	"light/tensor"
)

// A tiny GLM-5.3 (scripts/verify/ref_glm5.py: transformers' Glm5NextTextModel
// with Kimi delta attention, MLA without RoPE, dense and sparse MLPs,
// hyper-connections) against transformers: every position's logits, for a
// prompt within index_topk (dense attention is exact there).
func TestGLM5TinyMatchesTransformers(t *testing.T) {
	for _, name := range []string{"glm5-tiny", "glm5-fp8-tiny"} {
		t.Run(name, func(t *testing.T) { testGLM5Tiny(t, name) })
	}
}

// testGLM5Tiny compares testdata/<name> (glm5-fp8-tiny: block-scaled FP8
// projections, per-expert layout, against the dequantized model).
func testGLM5Tiny(t *testing.T, name string) {
	raw, err := os.ReadFile("testdata/" + name + ".json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		IDs    []int64
		Logits []float32
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		t.Run(fmt.Sprint(dev), func(t *testing.T) {
			m, err := LoadGLM5("testdata/"+name, dev)
			if name == "glm5-fp8-tiny" && device.IsCPU(dev) && m != nil && m.Layers[0].Q.WBlockScale == nil { // WebGPU widens at load
				t.Fatalf("FP8 projections did not load block-scaled")
			}
			if err != nil {
				t.Fatal(err)
			}
			ids, err := tensor.New(ref.IDs, []int{1, len(ref.IDs)}, dev)
			if err != nil {
				t.Fatal(err)
			}
			out, err := m.Forward(ids)
			if err != nil {
				t.Fatal(err)
			}
			c, _ := out.To(device.CPU())
			got := tensor.MustSlice[float32](c)
			if len(got) != len(ref.Logits) {
				t.Fatalf("%d logits, want %d", len(got), len(ref.Logits))
			}
			md := 0.0
			for i := range got {
				md = math.Max(md, math.Abs(float64(got[i]-ref.Logits[i])))
			}
			t.Logf("logits: max diff %.3g over %d", md, len(got))
			if !(md <= 1e-3) { // NaN fails too
				t.Errorf("logits differ by %.3g", md)
			}
		})
	}
}

// Prefill then token-by-token decode through GLM5Cache reproduces
// transformers' full-prompt logits at every position (CPU and WebGPU).
func TestGLM5CacheMatchesFullForward(t *testing.T) {
	raw, err := os.ReadFile("testdata/glm5-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		IDs    []int64
		Logits []float32
		Vocab  int
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		t.Run(fmt.Sprint(dev), func(t *testing.T) {
			m, err := LoadGLM5("testdata/glm5-tiny", dev)
			if err != nil {
				t.Fatal(err)
			}
			c := m.NewCache()
			const prefill = 7
			var got []float32
			step := func(ids []int64) {
				x, err := tensor.New(ids, []int{1, len(ids)}, dev)
				if err != nil {
					t.Fatal(err)
				}
				out, err := m.ForwardWithCache(x, c)
				if err != nil {
					t.Fatal(err)
				}
				h, _ := out.To(device.CPU())
				got = append(got, tensor.MustSlice[float32](h)...)
			}
			step(ref.IDs[:prefill])
			for i := prefill; i < len(ref.IDs); i++ {
				step(ref.IDs[i : i+1])
			}
			if c.Len != len(ref.IDs) {
				t.Errorf("cache length %d, want %d", c.Len, len(ref.IDs))
			}
			md := 0.0
			for i := range got {
				md = math.Max(md, math.Abs(float64(got[i]-ref.Logits[i])))
			}
			t.Logf("prefill %d + %d decode steps: max diff %.3g", prefill, len(ref.IDs)-prefill, md)
			if !(md <= 1e-3) {
				t.Errorf("cached logits differ by %.3g", md)
			}
		})
	}
}
