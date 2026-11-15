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

// A tiny DeepSeek-V4.1 (scripts/verify/ref_deepseek_v41.py: DeepSeek's own
// inference model.py with plain-PyTorch kernels; engram layers, compressors
// at ratios 2 and 1 shared across layers, indexers, candidate blocks,
// hyper-connections, sqrtsoftplus MoE, attention sinks, grouped low-rank
// output) in the released storage (FP8 + UE8M0 block scales, FP4 experts,
// FP8 engram table): every position's logits for a (2, 20) prompt.
func TestDeepseekV41TinyMatchesReference(t *testing.T) {
	raw, err := os.ReadFile("testdata/deepseek-v41-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		IDs    [][]int64
		Logits []float32
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	B, S := len(ref.IDs), len(ref.IDs[0])
	var ids []int64
	for _, r := range ref.IDs {
		ids = append(ids, r...)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		t.Run(fmt.Sprint(dev), func(t *testing.T) {
			m, err := LoadDeepseekV41("testdata/deepseek-v41-tiny", dev)
			if err != nil {
				t.Fatal(err)
			}
			if device.IsCPU(dev) && m.Layers[0].WqA.WBlockScale == nil {
				t.Fatalf("FP8 projections did not load block-scaled")
			}
			tok, err := tensor.New(ids, []int{B, S}, dev)
			if err != nil {
				t.Fatal(err)
			}
			out, err := m.Forward(tok)
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
			if !(md <= 1e-4) { // NaN fails too
				t.Errorf("logits differ by %.3g", md)
			}
		})
	}
}

// numpy.random.default_rng(seed).integers(0, high, n, dtype=np.int64), the
// engram hash multipliers' source (values from numpy 2).
func TestDeepseekV41NumpyIntegers(t *testing.T) {
	cases := []struct {
		seed, high uint64
		want       []int64
	}{
		{10007, 23058430092136939, []int64{18984069153536582, 1198982504596558, 8908289623672409}},
		{5 * 10007, 23058430092136939, []int64{9017781564827026, 7667466432094251, 5045411920603748}},
		{0, 23058430092136939, []int64{14687336538471342, 6220858079111564, 944785137311245}},
		{14 * 10007, 46539438283891, []int64{33858405369630, 25755403400457, 15460673601360, 41309613242795}},
	}
	for _, c := range cases {
		got, err := numpyIntegers(c.seed, len(c.want), c.high)
		if err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("seed %d: %v, want %v", c.seed, got, c.want)
				break
			}
		}
	}
}

// Prefill, then single tokens and a short chunk through DeepseekV41Cache
// reproduce the full prompt's logits: the sliding window rolls past its 4
// keys, ratio-2 groups complete across steps, the indexers and candidate
// blocks see the cached compressed positions and the engram n-grams reach
// back into the history.
func TestDeepseekV41CacheMatchesFullForward(t *testing.T) {
	raw, err := os.ReadFile("testdata/deepseek-v41-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct{ IDs [][]int64 }
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	B, S := len(ref.IDs), len(ref.IDs[0])
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		t.Run(fmt.Sprint(dev), func(t *testing.T) {
			m, err := LoadDeepseekV41("testdata/deepseek-v41-tiny", dev)
			if err != nil {
				t.Fatal(err)
			}
			span := func(a, n int) *tensor.Tensor {
				var ids []int64
				for _, r := range ref.IDs {
					ids = append(ids, r[a:a+n]...)
				}
				x, err := tensor.New(ids, []int{B, n}, dev)
				if err != nil {
					t.Fatal(err)
				}
				return x
			}
			full, err := m.Forward(span(0, S))
			if err != nil {
				t.Fatal(err)
			}
			fc, _ := full.To(device.CPU())
			want := tensor.MustSlice[float32](fc)
			V := len(want) / (B * S)
			c := m.NewCache()
			md := 0.0
			for _, step := range [][2]int{{0, 7}, {7, 1}, {8, 1}, {9, 3}, {12, 1}, {13, 1}, {14, 1}, {15, 5}} {
				out, err := m.ForwardWithCache(span(step[0], step[1]), c)
				if err != nil {
					t.Fatal(err)
				}
				oc, _ := out.To(device.CPU())
				got := tensor.MustSlice[float32](oc)
				for b := range B {
					for s := range step[1] {
						for v := range V {
							g := got[(b*step[1]+s)*V+v]
							w := want[(b*S+step[0]+s)*V+v]
							md = math.Max(md, math.Abs(float64(g-w)))
							if g != g {
								md = math.Inf(1)
							}
						}
					}
				}
			}
			if c.Pos() != S {
				t.Fatalf("cache at %d, want %d", c.Pos(), S)
			}
			t.Logf("cached decode vs full forward: max diff %.3g", md)
			if !(md <= 1e-4) {
				t.Errorf("cached logits differ by %.3g", md)
			}
		})
	}
}
