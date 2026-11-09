package models

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/tensor"
)

// TestDecodeStepProfile counts the driver calls behind one decode step of a
// small model on CUDA and times the steady-state step. It is the yardstick
// for the per-token overhead work (ROADMAP: CUDA graphs / per-token
// overhead). Run with -cpuprofile to see where the host time goes.
//
//	LIGHT_HEAVY_TESTS=1 go test ./models -run TestDecodeStepProfile -v
func TestDecodeStepProfile(t *testing.T) {
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
		repo string
		load func(string, device.Device, dtype.DType) (CausalLM, error)
	}{
		{"meta-llama/Llama-3.2-1B-Instruct", func(r string, d device.Device, dt dtype.DType) (CausalLM, error) { return LoadCausalLMFromHF(r, d, dt) }},
		{"Qwen/Qwen3.5-0.8B", func(r string, d device.Device, dt dtype.DType) (CausalLM, error) { return LoadQwen35FromHF(r, d, dt) }},
		{"gguf:llama-3.2-1b-instruct-q4_k_m.gguf", func(r string, d device.Device, dt dtype.DType) (CausalLM, error) {
			return LoadLlamaGGUF(filepath.Join(ggufSearchDir(), strings.TrimPrefix(r, "gguf:")), d)
		}},
	} {
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
		cache := NewKVCache(m.NumLayers())
		ids := make([]int64, 16)
		for i := range ids {
			ids[i] = int64(1000 + 7*i)
		}
		row := lastLogits(t, m, ids, cache, dev)
		next := int64(argmaxRow(row))
		for i := 0; i < 8; i++ { // warm up: pool, JIT, cache growth
			row = lastLogits(t, m, []int64{next}, cache, dev)
			next = int64(argmaxRow(row))
			runtime.GC()
		}
		const steps = 32
		before := cudart.ReadCounters()
		var issue, drain, read, gc time.Duration
		t0 := time.Now()
		for i := 0; i < steps; i++ {
			// Phases of one engine step: issue the forward (async launches),
			// wait for the GPU, read the logits back, collect the step's garbage.
			t1 := time.Now()
			x, err := tensor.New([]int64{next}, []int{1, 1}, dev)
			if err != nil {
				t.Fatal(err)
			}
			logits, err := m.ForwardWithCache(x, cache)
			if err != nil {
				t.Fatal(err)
			}
			t2 := time.Now()
			if err := cudart.DeviceSynchronize(); err != nil {
				t.Fatal(err)
			}
			t3 := time.Now()
			cpu, _ := logits.To(device.CPU())
			row = tensor.MustSlice[float32](cpu)
			next = int64(argmaxRow(row))
			t4 := time.Now()
			runtime.GC() // as engine.Generate does: returns the step's device buffers to the pool
			t5 := time.Now()
			issue += t2.Sub(t1)
			drain += t3.Sub(t2)
			read += t4.Sub(t3)
			gc += t5.Sub(t4)
		}
		el := time.Since(t0)
		c := cudart.ReadCounters().Sub(before)
		L := uint64(m.NumLayers())
		ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 / steps }
		t.Logf("%s BF16: %d layers, %.2f ms/step (%.1f tok/s): issue %.2f + GPU drain %.2f + logits read %.2f + GC %.2f ms",
			tc.repo, L, ms(time.Duration(el)), float64(steps)/el.Seconds(), ms(issue), ms(drain), ms(read), ms(gc))
		t.Logf("  per step: launches %d (%d/layer) gemms %d mallocs %d frees %d HtoD %d DtoH %d DtoD %d (same-dtype casts %d) memsets %d syncs %d",
			c.Launches/steps, c.Launches/steps/L, c.Gemms/steps, c.Mallocs/steps, c.Frees/steps, c.HtoD/steps, c.DtoH/steps, c.DtoD/steps, c.CastCopies/steps, c.Memsets/steps, c.Syncs/steps)
		cache.Reset()
	}
}

// hfCached reports whether repo has a config.json in the local HF cache.
func hfCached(repo string) bool {
	home, _ := os.UserHomeDir()
	m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--"+strings.ReplaceAll(repo, "/", "--")+"/snapshots/*/config.json"))
	return len(m) > 0
}
