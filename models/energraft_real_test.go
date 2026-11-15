package models

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/tensor"
)

// Energraft-EBT-Qwen3-1.7B-Base on its real weights
// (scripts/colab/energraft_check.py downloads the release and records its
// own generation): replaying the reference's Gaussian draws, each token's
// refined candidate state and the greedy tokens. Runs when
// LIGHT_ENERGRAFT_DIR names that folder.
func TestEnergraftRealMatchesReference(t *testing.T) {
	dir := os.Getenv("LIGHT_ENERGRAFT_DIR")
	if dir == "" {
		t.Skip("LIGHT_ENERGRAFT_DIR not set")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "ref.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Prompt, Tokens, EOS []int64
		Draws, States       [][]float32
		Text                string
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	m, err := LoadEnergraft(dir, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	V := m.Backbone.Cfg.VocabSize
	c := NewKVCache(len(m.Backbone.Blocks))
	feed := func(ids []int64) {
		x, _ := tensor.New(ids, []int{1, len(ids)}, device.CPU())
		if _, err := m.Backbone.ForwardWithCache(x, c); err != nil {
			t.Fatal(err)
		}
	}
	feed(ref.Prompt)
	var toks []int64
	md, mr := 0.0, 0.0
	for i, draw := range ref.Draws {
		init, _ := tensor.New(draw[:V], []int{1, V}, device.CPU())
		states, err := m.Refine(c, init, 0)
		if err != nil {
			t.Fatal(err)
		}
		got := tensor.MustSlice[float32](states[len(states)-1])
		best := 0
		for j, v := range got {
			md = math.Max(md, math.Abs(float64(v-ref.States[i][j])))
			mr = math.Max(mr, math.Abs(float64(ref.States[i][j])))
			if v > got[best] {
				best = j
			}
		}
		toks = append(toks, int64(best))
		if i+1 < len(ref.Tokens) {
			feed([]int64{ref.Tokens[i]})
		}
	}
	t.Logf("states max diff %.3g (max |ref| %.3g); tokens %v, reference %v (%q)", md, mr, toks, ref.Tokens, ref.Text)
	for i := range ref.Tokens {
		if i >= len(toks) || toks[i] != ref.Tokens[i] {
			t.Errorf("token %d: %v, reference %v", i, toks, ref.Tokens)
			break
		}
	}
	if !(md <= 1e-3*math.Max(mr, 1)) {
		t.Errorf("states differ by %.3g", md)
	}
}
