package models

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// Tiny LLaDA and Dream (scripts/verify/ref_masked_diffusion.py: each
// release's own modeling and sampling code on seeded 3-layer models): the
// bidirectional logits of a prompt + mask canvas, and deterministic
// generations - LLaDA's low-confidence remasking over 2 blocks with and
// without CFG, Dream's maskgit_plus / topk_margin / entropy.
func TestMaskedDiffusionTinyMatchesReference(t *testing.T) {
	type ref struct {
		Input       []int64
		Logits      []float32
		Vocab       int
		MaskID      int64 `json:"mask_id"`
		Prompt      []int64
		Steps       int
		GenLength   int     `json:"gen_length"`
		BlockLength int     `json:"block_length"`
		CFGScale    float64 `json:"cfg_scale"`
		MaxNew      int     `json:"max_new_tokens"`
		Gens        map[string][]int64
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, name := range []string{"llada", "dream"} {
		raw, err := os.ReadFile("testdata/" + name + "-tiny.json")
		if err != nil {
			t.Fatal(err)
		}
		var r ref
		if err := json.Unmarshal(raw, &r); err != nil {
			t.Fatal(err)
		}
		for _, dev := range devs {
			t.Run(fmt.Sprintf("%s/%v", name, dev), func(t *testing.T) {
				m, err := LoadCausalLMFromHF("testdata/"+name+"-tiny", dev, dtype.F32)
				if err != nil {
					t.Fatal(err)
				}
				x, _ := tensor.New(r.Input, []int{1, len(r.Input)}, dev)
				out, err := m.ForwardBidirectional(x)
				if err != nil {
					t.Fatal(err)
				}
				c, _ := out.To(device.CPU())
				got := tensor.MustSlice[float32](c)
				md := 0.0
				for i := range got {
					md = math.Max(md, math.Abs(float64(got[i]-r.Logits[i])))
				}
				t.Logf("bidirectional logits: max diff %.3g over %d", md, len(got))
				if !(md <= 1e-4) {
					t.Errorf("logits differ by %.3g", md)
				}
				for mode, want := range r.Gens {
					var toks []int64
					if name == "llada" {
						o := MaskedDiffusionOpts{MaskID: r.MaskID, GenLength: r.GenLength, Steps: r.Steps, BlockLength: r.BlockLength}
						if mode == "cfg" {
							o.CFGScale = r.CFGScale
						}
						toks, err = LLaDAGenerate(m, r.Prompt, o)
					} else {
						toks, err = DreamGenerate(m, r.Prompt, MaskedDiffusionOpts{MaskID: r.MaskID, GenLength: r.MaxNew, Steps: r.Steps, Alg: mode})
					}
					if err != nil {
						t.Fatal(err)
					}
					if fmt.Sprint(toks) != fmt.Sprint(want) {
						t.Errorf("%s: %v, reference %v", mode, toks, want)
					}
				}
			})
		}
	}
}
