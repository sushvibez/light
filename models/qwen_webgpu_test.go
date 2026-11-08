package models

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
)

// Qwen2.5-0.5B-Instruct on WebGPU (F32) against the CPU: prefill logits and
// 12 greedy decode steps through the KV cache (skips without a WebGPU
// adapter or the cached checkpoint).
func TestQwen2WebGPUMatchesCPU(t *testing.T) {
	g, err := device.WebGPU(0)
	if err != nil {
		t.Skip(err)
	}
	cpu := device.CPU()
	const repo = "Qwen/Qwen2.5-0.5B-Instruct"
	mc, err := LoadCausalLMFromHF(repo, cpu, dtype.F32)
	if err != nil {
		t.Skip(err)
	}
	mg, err := LoadCausalLMFromHF(repo, g, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	// "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
	ids := []int64{151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198}
	cc, cg := NewKVCache(mc.NumLayers()), NewKVCache(mg.NumLayers())
	rc, rg := lastLogits(t, mc, ids, cc, cpu), lastLogits(t, mg, ids, cg, g)
	if d := maxAbsDiff(rc, rg); d > 1e-2 {
		t.Fatalf("prefill logits differ by %.3g", d)
	} else {
		t.Logf("prefill logits max |cpu - webgpu| %.3g", d)
	}
	var tc, tg []int64
	for range 12 {
		nc, ng := int64(top5(rc)[0]), int64(top5(rg)[0])
		tc, tg = append(tc, nc), append(tg, ng)
		rc, rg = lastLogits(t, mc, []int64{nc}, cc, cpu), lastLogits(t, mg, []int64{ng}, cg, g)
	}
	t.Logf("greedy cpu %v\n       webgpu %v", tc, tg)
	if !slices.Equal(tc, tg) {
		t.Errorf("greedy decode differs")
	}
	if d := maxAbsDiff(rc, rg); d > 2e-2 {
		t.Errorf("logits after 12 steps differ by %.3g", d)
	}
}

// Qwen2.5-0.5B-Instruct Q4_K_M GGUF (Q5_0 / Q8_0 / Q4_K / Q6_K tensors) on
// WebGPU against the CPU: prefill logits and 12 greedy steps (skips without
// an adapter or the file in the GGUF directory).
func TestQwen2GGUFWebGPUMatchesCPU(t *testing.T) {
	g, err := device.WebGPU(0)
	if err != nil {
		t.Skip(err)
	}
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, ".cache", "light", "gguf", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skip("qwen2.5-0.5b-instruct-q4_k_m.gguf not in ~/.cache/light/gguf (light pull Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf)")
	}
	cpu := device.CPU()
	mc, err := LoadChatModelGGUF(path, cpu)
	if err != nil {
		t.Fatal(err)
	}
	mg, err := LoadChatModelGGUF(path, g)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int64{151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198}
	cc, cg := NewKVCache(mc.NumLayers()), NewKVCache(mg.NumLayers())
	rc, rg := lastLogits(t, mc, ids, cc, cpu), lastLogits(t, mg, ids, cg, g)
	if d := maxAbsDiff(rc, rg); d > 1e-2 {
		t.Fatalf("prefill logits differ by %.3g", d)
	} else {
		t.Logf("prefill logits max |cpu - webgpu| %.3g", d)
	}
	var tc, tg []int64
	for range 12 {
		nc, ng := int64(top5(rc)[0]), int64(top5(rg)[0])
		tc, tg = append(tc, nc), append(tg, ng)
		rc, rg = lastLogits(t, mc, []int64{nc}, cc, cpu), lastLogits(t, mg, []int64{ng}, cg, g)
	}
	t.Logf("greedy cpu %v\n       webgpu %v", tc, tg)
	if !slices.Equal(tc, tg) {
		t.Errorf("greedy decode differs")
	}
}
