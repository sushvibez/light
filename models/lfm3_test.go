package models

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/tensor"
)

// A seeded tiny LFM3 (conv, full and sliding attention, window 4, GQA,
// zero-tail partial RoPE, sandwich norms; scripts/verify/ref_lfm3.py)
// against the float64 reference: one 12-token forward, and the same tokens
// as a 5-token prefill plus single-token decode steps through the cache
// (conv state, the window sliding past the cache).
func TestLFM3TinyMatchesReference(t *testing.T) {
	raw, err := os.ReadFile("testdata/lfm3/tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		IDs    []int64   `json:"ids"`
		Logits []float32 `json:"logits"`
		Vocab  int       `json:"vocab"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	V, T := ref.Vocab, len(ref.IDs)
	cmp := func(t *testing.T, name string, x *tensor.Tensor, from int) {
		t.Helper()
		x, err := x.To(device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		got := tensor.MustSlice[float32](x)
		want := ref.Logits[from*V : from*V+len(got)]
		md := 0.0
		for i := range got {
			md = math.Max(md, math.Abs(float64(got[i]-want[i])))
		}
		if md > 2e-4 {
			t.Errorf("%s: logits differ by %.3g", name, md)
		}
		t.Logf("%s: max diff %.3g", name, md)
	}
	for _, dev := range devs {
		t.Run(dev.String(), func(t *testing.T) {
			m, err := LoadLFM3FromHF("testdata/lfm3/tiny", dev, dtype.F32)
			if err != nil {
				t.Fatal(err)
			}
			ids, _ := tensor.New(ref.IDs, []int{1, T}, dev)
			logits, err := m.ForwardWithCache(ids, nil)
			if err != nil {
				t.Fatal(err)
			}
			cmp(t, "full forward", logits, 0)

			cache := NewKVCache(m.NumLayers())
			pre, _ := tensor.New(ref.IDs[:5], []int{1, 5}, dev)
			if logits, err = m.ForwardWithCache(pre, cache); err != nil {
				t.Fatal(err)
			}
			cmp(t, "prefill 5", logits, 0)
			worst := 0.0
			for i := 5; i < T; i++ {
				one, _ := tensor.New(ref.IDs[i:i+1], []int{1, 1}, dev)
				if logits, err = m.ForwardWithCache(one, cache); err != nil {
					t.Fatal(err)
				}
				x, _ := logits.To(device.CPU())
				for j, v := range tensor.MustSlice[float32](x) {
					worst = math.Max(worst, math.Abs(float64(v-ref.Logits[i*V+j])))
				}
			}
			t.Logf("decode steps 5..%d: max diff %.3g", T-1, worst)
			if worst > 2e-4 {
				t.Errorf("decode logits differ by %.3g", worst)
			}
		})
	}
}

// LiquidAI/LFM3-1.7B (BF16 weights, F32 compute) against the float64
// reference (ref_lfm3.py --real): the top-5 logits after a chat prompt and
// 8 greedy tokens through the cache.
func TestLFM3RealMatchesReference(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (3.5 GB checkpoint)")
	}
	raw, err := os.ReadFile("testdata/lfm3/real.json")
	if err != nil {
		t.Skip(err)
	}
	var ref struct {
		IDs       []int64   `json:"ids"`
		TopIDs    []int     `json:"top_ids"`
		TopLogits []float64 `json:"top_logits"`
		Greedy    []int64   `json:"greedy"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	requireHostMemGB(t, 6)
	releaseModelMemory(t)
	dev := device.CPU()
	var m *LFM3
	if err := nn.WithWeightStorage16(dtype.BF16, func() (err error) {
		m, err = LoadLFM3FromHF("LiquidAI/LFM3-1.7B", dev, dtype.F32)
		return
	}); err != nil {
		t.Fatal(err)
	}
	cache := NewKVCache(m.NumLayers())
	ids, _ := tensor.New(ref.IDs, []int{1, len(ref.IDs)}, dev)
	logits, err := m.ForwardWithCache(ids, cache)
	if err != nil {
		t.Fatal(err)
	}
	V := m.VocabSize()
	last := tensor.MustSlice[float32](logits)[(len(ref.IDs)-1)*V:]
	for i, id := range ref.TopIDs {
		d := math.Abs(float64(last[id]) - ref.TopLogits[i])
		t.Logf("top %d: id %d logit %.4f (ref %.4f)", i, id, last[id], ref.TopLogits[i])
		if d > 0.02 {
			t.Errorf("logit of %d differs by %.3g", id, d)
		}
	}
	var gen []int64
	next := argmaxF32(last)
	for range ref.Greedy {
		gen = append(gen, next)
		one, _ := tensor.New([]int64{next}, []int{1, 1}, dev)
		if logits, err = m.ForwardWithCache(one, cache); err != nil {
			t.Fatal(err)
		}
		next = argmaxF32(tensor.MustSlice[float32](logits))
	}
	t.Logf("greedy %v (ref %v)", gen, ref.Greedy)
	for i := range gen {
		if gen[i] != ref.Greedy[i] {
			t.Fatalf("greedy token %d: %d, want %d", i, gen[i], ref.Greedy[i])
		}
	}
}

func argmaxF32(x []float32) int64 {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return int64(best)
}
