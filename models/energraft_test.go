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

// A tiny Energraft-EBT (scripts/verify/ref_energraft.py: the release's own
// energraft code on a seeded 3-layer Qwen3 and energy head): with the
// reference's Gaussian draws, every token's refined candidate state (1 and
// 2 energy updates, the second relaxed by 0.05) and the greedy tokens.
func TestEnergraftTinyMatchesReference(t *testing.T) {
	raw, err := os.ReadFile("testdata/energraft-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Prompt []int64
		Vocab  int
		Runs   map[string]struct {
			Tokens []int64
			Draws  [][]float32
			States [][]float32
		}
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
			m, err := LoadEnergraft("testdata/energraft-tiny", dev)
			if err != nil {
				t.Fatal(err)
			}
			for _, steps := range []int{1, 2} {
				run := ref.Runs[fmt.Sprint(steps)]
				// The states, token by token along the reference's tokens.
				c := NewKVCache(len(m.Backbone.Blocks))
				feed := func(ids []int64) {
					x, _ := tensor.New(ids, []int{1, len(ids)}, dev)
					if _, err := m.Backbone.ForwardWithCache(x, c); err != nil {
						t.Fatal(err)
					}
				}
				feed(ref.Prompt)
				md, mr := 0.0, 0.0
				for i, draw := range run.Draws {
					init, _ := tensor.New(draw, []int{1, ref.Vocab}, dev)
					states, err := m.Refine(c, init, steps)
					if err != nil {
						t.Fatal(err)
					}
					h, _ := states[len(states)-1].To(device.CPU())
					got := tensor.MustSlice[float32](h)
					for j, v := range got {
						md = math.Max(md, math.Abs(float64(v-run.States[i][j])))
						mr = math.Max(mr, math.Abs(float64(run.States[i][j])))
						if v != v {
							md = math.Inf(1)
						}
					}
					if i+1 < len(run.Tokens) {
						feed([]int64{run.Tokens[i]})
					}
				}
				c.Reset()
				// Greedy generation with the same draws.
				toks, err := m.Generate(ref.Prompt, EnergraftOpts{MaxTokens: len(run.Tokens), Steps: steps,
					EOS:  map[int64]bool{1: true},
					Init: func(i int) (*tensor.Tensor, error) { return tensor.New(run.Draws[i], []int{1, ref.Vocab}, dev) }})
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("%d update(s): states max diff %.3g (max |ref| %.3g); tokens %v, reference %v", steps, md, mr, toks, run.Tokens)
				if !(md <= 1e-4*math.Max(mr, 1)) {
					t.Errorf("%d update(s): states differ by %.3g", steps, md)
				}
				if fmt.Sprint(toks) != fmt.Sprint(run.Tokens) {
					t.Errorf("%d update(s): tokens %v, reference %v", steps, toks, run.Tokens)
				}
			}
		})
	}
}
