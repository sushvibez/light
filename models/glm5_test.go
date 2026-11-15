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
	raw, err := os.ReadFile("testdata/glm5-tiny.json")
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
			m, err := LoadGLM5("testdata/glm5-tiny", dev)
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
			if md > 1e-3 {
				t.Errorf("logits differ by %.3g", md)
			}
		})
	}
}
