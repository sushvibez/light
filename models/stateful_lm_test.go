package models

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// GLM-5.3, DeepSeek-V4.1 and Kimi K3.1 load through LoadChatModelFromHF (a
// local folder) as CausalLMs whose own decode state rides in KVCache:
// prefill 5 then single tokens reproduces the full forward's logits, Reset
// starts over, and a cache with that state refuses to rewind.
func TestStatefulLMsThroughKVCache(t *testing.T) {
	for _, dir := range []string{"testdata/glm5-tiny", "testdata/deepseek-v41-tiny", "testdata/kimi-k31-tiny"} {
		t.Run(dir, func(t *testing.T) {
			lm, err := LoadChatModelFromHF(dir, device.CPU(), dtype.F32)
			if err != nil {
				t.Fatal(err)
			}
			if lm.ParamCount() <= 0 || lm.MaxContext() <= 0 || lm.NumLayers() <= 0 {
				t.Fatalf("meta: params %d, max context %d, layers %d", lm.ParamCount(), lm.MaxContext(), lm.NumLayers())
			}
			ids := []int64{5, 17, 99, 3, 42, 8, 120, 77, 9, 61}
			S, V := len(ids), lm.VocabSize()
			tok := func(a, n int) *tensor.Tensor {
				x, err := tensor.New(append([]int64(nil), ids[a:a+n]...), []int{1, n}, device.CPU())
				if err != nil {
					t.Fatal(err)
				}
				return x
			}
			full, err := lm.ForwardWithCache(tok(0, S), nil)
			if err != nil {
				t.Fatal(err)
			}
			want := tensor.MustSlice[float32](full)
			c := NewKVCache(lm.NumLayers())
			for round := range 2 { // the second after Reset
				md := 0.0
				for _, st := range [][2]int{{0, 5}, {5, 1}, {6, 1}, {7, 3}} {
					out, err := lm.ForwardWithCache(tok(st[0], st[1]), c)
					if err != nil {
						t.Fatal(err)
					}
					got := tensor.MustSlice[float32](out)
					for i, g := range got {
						w := want[st[0]*V+i]
						md = math.Max(md, math.Abs(float64(g-w)))
						if g != g {
							md = math.Inf(1)
						}
					}
				}
				t.Logf("round %d: cached decode vs full forward max diff %.3g", round, md)
				if !(md <= 1e-4) || c.Len != S {
					t.Fatalf("round %d: diff %.3g, cache length %d", round, md, c.Len)
				}
				if err := c.Truncate(3); err == nil {
					t.Fatal("Truncate rewound a recurrent decode state")
				}
				c.Reset()
			}
		})
	}
}
