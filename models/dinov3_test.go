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

// Tiny DINOv3 ViTs (scripts/verify/ref_dinov3.py: transformers, gated
// SwiGLU and plain GELU MLPs) on a 48 x 64 image: last_hidden_state, and
// Idol's tokens (a LayerNorm without affine on top).
func TestDINOv3TinyMatchesTransformers(t *testing.T) {
	raw, err := os.ReadFile("testdata/dinov3-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref map[string]struct {
		Pixels, Hidden []float32
		IdolTokens     []float32 `json:"idol_tokens"`
		Shape          []int
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, name := range []string{"gated", "plain"} {
		r := ref[name]
		for _, dev := range devs {
			t.Run(fmt.Sprintf("%s/%v", name, dev), func(t *testing.T) {
				m, err := LoadDINOv3("testdata/dinov3-tiny/"+name, dev)
				if err != nil {
					t.Fatal(err)
				}
				px, err := tensor.New(r.Pixels, []int{1, 3, 48, 64}, dev)
				if err != nil {
					t.Fatal(err)
				}
				h, err := m.Forward(px)
				if err != nil {
					t.Fatal(err)
				}
				tok, err := IdolImageTokens(h)
				if err != nil {
					t.Fatal(err)
				}
				for _, c := range []struct {
					name string
					x    *tensor.Tensor
					want []float32
				}{{"hidden", h, r.Hidden}, {"idol tokens", tok, r.IdolTokens}} {
					x, _ := c.x.To(device.CPU())
					got := tensor.MustSlice[float32](x)
					md := 0.0
					for i := range got {
						md = math.Max(md, math.Abs(float64(got[i]-c.want[i])))
					}
					t.Logf("%s: max diff %.3g over %d", c.name, md, len(got))
					if len(got) != len(c.want) || md > 1e-4 {
						t.Errorf("%s differs by %.3g", c.name, md)
					}
				}
			})
		}
	}
}
