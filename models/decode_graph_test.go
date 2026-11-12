package models

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
)

// A captured decode step must produce the same greedy continuation as the
// eager step, across a run long enough to cross a split-K boundary of the
// decode attention kernel and, for the small reserve used here, a buffer
// growth and re-capture.
func TestDecodeGraphMatchesEager(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		repo   string
		prompt int // prefill length (past a sliding window where the model has one)
		load   func(string, device.Device, dtype.DType) (CausalLM, error)
	}{
		{"meta-llama/Llama-3.2-1B-Instruct", 200, func(r string, d device.Device, dt dtype.DType) (CausalLM, error) { return LoadCausalLMFromHF(r, d, dt) }},
		{"Qwen/Qwen3.5-0.8B", 200, func(r string, d device.Device, dt dtype.DType) (CausalLM, error) { return LoadQwen35FromHF(r, d, dt) }},
		{"gguf:llama-3.2-1b-instruct-q4_k_m.gguf", 200, func(r string, d device.Device, dt dtype.DType) (CausalLM, error) {
			return LoadLlamaGGUF(filepath.Join(ggufSearchDir(), strings.TrimPrefix(r, "gguf:")), d)
		}},
		{"unsloth/gemma-3-1b-it", 600, func(r string, d device.Device, dt dtype.DType) (CausalLM, error) { return LoadGemma3FromHF(r, d, dt) }},
		{"ibm-granite/granite-4.0-micro", 200, func(r string, d device.Device, dt dtype.DType) (CausalLM, error) { return LoadCausalLMFromHF(r, d, dt) }},
	} {
		if f := os.Getenv("LIGHT_PROFILE_MODEL"); f != "" && !strings.Contains(tc.repo, f) {
			continue
		}
		if strings.HasPrefix(tc.repo, "gguf:") {
			if _, err := os.Stat(filepath.Join(ggufSearchDir(), strings.TrimPrefix(tc.repo, "gguf:"))); err != nil {
				t.Logf("%s not on disk, skipped", tc.repo)
				continue
			}
		} else if !hfCached(tc.repo) {
			t.Logf("%s not cached, skipped", tc.repo)
			continue
		}
		releaseModelMemory(t)
		m, err := tc.load(tc.repo, dev, dtype.BF16)
		if err != nil {
			t.Fatal(err)
		}
		ids := make([]int64, tc.prompt) // past the first split-K blocks; past the window for Gemma 3
		for i := range ids {
			ids[i] = int64(1000 + 7*i)
		}
		const steps = 80
		eager := NewKVCache(m.NumLayers())
		next := int64(argmaxRow(lastLogits(t, m, ids, eager, dev)))
		var want []int64
		for i := 0; i < steps; i++ {
			want = append(want, next)
			next = int64(argmaxRow(lastLogits(t, m, []int64{next}, eager, dev)))
		}
		eager.Reset()

		cache := NewKVCache(m.NumLayers())
		next = int64(argmaxRow(lastLogits(t, m, ids, cache, dev)))
		g, err := NewDecodeGraph(m, cache, dev, len(ids)+1) // tight: the run must outgrow it and re-capture
		if err != nil {
			t.Fatalf("%s: %v", tc.repo, err)
		}
		capBefore := cache.Capacity()
		var got []int64
		for i := 0; i < steps; i++ {
			got = append(got, next)
			logits, err := g.Step(next)
			if err != nil {
				t.Fatalf("%s step %d: %v", tc.repo, i, err)
			}
			next = int64(argmaxDevice(t, logits))
		}
		_ = cudart.DeviceSynchronize()
		g.Close()
		if cache.Len != len(ids)+steps {
			t.Fatalf("%s: cache.Len %d, want %d", tc.repo, cache.Len, len(ids)+steps)
		}
		t.Logf("%s: %d graph nodes, capacity %d → %d, tokens %v", tc.repo, g.Nodes(), capBefore, cache.Capacity(), got[:8])
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s: token %d: graph %d, eager %d (eager %v / graph %v)", tc.repo, i, got[i], want[i], want[:i+1], got[:i+1])
			}
		}
		cache.Reset()
	}
}
