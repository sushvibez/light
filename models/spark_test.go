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

// A seeded tiny Spark-X (spark2_5: fused QKV, per-head sigmoid output gate,
// GELU MLP, window-4 sliding layers with full RoPE and a full layer with
// 0.25 partial RoPE; scripts/verify/ref_spark.py) against the float64
// reference: one 12-token forward, and a 5-token prefill plus single-token
// decode steps through the cache.
func TestSparkTinyMatchesReference(t *testing.T) {
	raw, err := os.ReadFile("testdata/spark/tiny.json")
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
	diff := func(x *tensor.Tensor, from int) float64 {
		x, err := x.To(device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		md := 0.0
		for i, v := range tensor.MustSlice[float32](x) {
			md = math.Max(md, math.Abs(float64(v-ref.Logits[from*V+i])))
		}
		return md
	}
	for _, dev := range devs {
		t.Run(dev.String(), func(t *testing.T) {
			m, err := LoadSparkFromHF("testdata/spark/tiny", dev, dtype.F32)
			if err != nil {
				t.Fatal(err)
			}
			ids, _ := tensor.New(ref.IDs, []int{1, T}, dev)
			logits, err := m.ForwardWithCache(ids, nil)
			if err != nil {
				t.Fatal(err)
			}
			full := diff(logits, 0)
			cache := NewKVCache(m.NumLayers())
			pre, _ := tensor.New(ref.IDs[:5], []int{1, 5}, dev)
			if logits, err = m.ForwardWithCache(pre, cache); err != nil {
				t.Fatal(err)
			}
			dec := diff(logits, 0)
			for i := 5; i < T; i++ {
				one, _ := tensor.New(ref.IDs[i:i+1], []int{1, 1}, dev)
				if logits, err = m.ForwardWithCache(one, cache); err != nil {
					t.Fatal(err)
				}
				dec = math.Max(dec, diff(logits, i))
			}
			t.Logf("full forward max diff %.3g; prefill + decode %.3g", full, dec)
			if full > 2e-4 || dec > 2e-4 {
				t.Errorf("logits differ: full %.3g, cached %.3g", full, dec)
			}
		})
	}
}

// XHToken/Spark-X2.6-4B (BF16 weights, F32 compute) against the float64
// reference (ref_spark.py --real): top-5 logits after a chat prompt and 8
// greedy tokens through the cache.
func TestSparkRealMatchesReference(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (8 GB checkpoint)")
	}
	raw, err := os.ReadFile("testdata/spark/real.json")
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
	requireHostMemGB(t, 12)
	releaseModelMemory(t)
	dev := device.CPU()
	var m *Llama
	if err := nn.WithWeightStorage16(dtype.BF16, func() (err error) {
		m, err = LoadSparkFromHF("XHToken/Spark-X2.6-4B", dev, dtype.F32)
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
		t.Logf("top %d: id %d logit %.4f (ref %.4f)", i, id, last[id], ref.TopLogits[i])
		if d := math.Abs(float64(last[id]) - ref.TopLogits[i]); d > 0.02 {
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
