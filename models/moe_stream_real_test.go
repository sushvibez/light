package models

import (
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"

	"light/device"
	"light/dtype"
)

// A real MoE checkpoint with its experts streamed (LIGHT_EXPERT_SLOTS,
// default 16 per layer): greedy decoding of a few tokens from a fixed
// prompt, reporting the Go heap, the decode rate and how often the
// next-layer prediction was right. Runs when LIGHT_STREAM_REPO names the
// repo or folder (e.g. Qwen/Qwen3-30B-A3B), BF16 weights.
func TestMoEStreamedRealCheckpoint(t *testing.T) {
	repo := os.Getenv("LIGHT_STREAM_REPO")
	if repo == "" {
		t.Skip("LIGHT_STREAM_REPO not set")
	}
	if os.Getenv("LIGHT_EXPERT_SLOTS") == "" {
		t.Setenv("LIGHT_EXPERT_SLOTS", "16")
	}
	slots, _ := strconv.Atoi(os.Getenv("LIGHT_EXPERT_SLOTS"))
	t0 := time.Now()
	m, err := LoadCausalLMFromHF(repo, device.CPU(), dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("loaded in %.1fs, %d layers, %d experts each, %d slots", time.Since(t0).Seconds(), m.NumLayers(), m.Cfg.NumExperts, slots)
	// "The capital of France is" in Qwen's tokenizer.
	ids := []int64{785, 6722, 315, 9625, 374}
	c := NewKVCache(m.NumLayers())
	row := lastLogits(t, m, ids, c, device.CPU())
	var out []int64
	t1 := time.Now()
	const N = 16
	for range N {
		best := 0
		for i, v := range row {
			if v > row[best] {
				best = i
			}
		}
		out = append(out, int64(best))
		row = lastLogits(t, m, []int64{int64(best)}, c, device.CPU())
	}
	dt := time.Since(t1).Seconds()
	var loads, hits, pf, used int64
	for _, b := range m.Blocks {
		if s := b.MoE; s != nil && s.Stream != nil {
			loads, hits, pf, used = loads+s.Stream.Loads, hits+s.Stream.Hits, pf+s.Stream.Prefetched, used+s.Stream.PrefetchUsed
		}
	}
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	t.Logf("tokens %v; %.2f s/token; expert loads %d, hits %d; prefetched %d, used %d (%.0f%%); Go heap %.1f GB",
		out, dt/N, loads, hits, pf, used, 100*float64(used)/max(float64(pf), 1), float64(ms.HeapAlloc)/(1<<30))
}
