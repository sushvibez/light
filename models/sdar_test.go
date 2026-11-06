package models

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// Block-diffusion forward on a tiny random model: the block-bidirectional
// mask computed in one shot over prefix+block must equal prefill(prefix,
// store) followed by a no-store pass over the block, and the no-store pass
// must leave the cache untouched.
func TestForwardDiffusionConsistency(t *testing.T) {
	cfg := LlamaConfig{VocabSize: 64, HiddenSize: 32, IntermediateSize: 48, NumLayers: 2, NumHeads: 4, NumKVHeads: 2,
		HeadDim: 8, MaxPosition: 64, RopeBase: 10000, RmsEps: 1e-6, WeightDType: dtype.F32, QKNorm: true}
	m, err := NewLlama(cfg, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	B := 4
	prefix := []int64{3, 7, 11, 13, 17, 19, 23, 29} // two complete blocks
	block := []int64{31, 1, 1, 1}                   // last block, three "masks"
	all := append(append([]int64{}, prefix...), block...)
	x, _ := tensor.New(all, []int{1, len(all)}, device.CPU())
	oneShot, err := m.ForwardDiffusion(x, nil, B, false)
	if err != nil {
		t.Fatal(err)
	}
	os1 := tensor.MustSlice[float32](oneShot)
	V := cfg.VocabSize
	cache := NewKVCache(cfg.NumLayers)
	xp, _ := tensor.New(prefix, []int{1, len(prefix)}, device.CPU())
	if _, err := m.ForwardDiffusion(xp, cache, B, true); err != nil {
		t.Fatal(err)
	}
	if cache.Len != len(prefix) {
		t.Fatalf("cache.Len %d after prefill, want %d", cache.Len, len(prefix))
	}
	k0 := tensor.MustSlice[float32](cache.K[0])
	xb, _ := tensor.New(block, []int{1, B}, device.CPU())
	noStore, err := m.ForwardDiffusion(xb, cache, B, false)
	if err != nil {
		t.Fatal(err)
	}
	if cache.Len != len(prefix) || cache.K[0].Dim(2) != len(prefix) {
		t.Fatalf("no-store pass changed the cache: Len %d, K len %d", cache.Len, cache.K[0].Dim(2))
	}
	k1 := tensor.MustSlice[float32](cache.K[0])
	for i := range k0 {
		if k0[i] != k1[i] {
			t.Fatalf("cache K changed at %d", i)
		}
	}
	ns := tensor.MustSlice[float32](noStore)
	for i := range B * V {
		if d := math.Abs(float64(ns[i] - os1[len(prefix)*V+i])); d > 1e-4 {
			t.Fatalf("block logits differ at %d: %v vs %v", i, ns[i], os1[len(prefix)*V+i])
		}
	}
	// The block's first position must see the later block positions
	// (bidirectional): a pure causal forward differs.
	causal, _ := m.Forward(x)
	cs := tensor.MustSlice[float32](causal)
	same := true
	for i := range V {
		if math.Abs(float64(cs[len(prefix)*V+i]-os1[len(prefix)*V+i])) > 1e-4 {
			same = false
		}
	}
	if same {
		t.Fatal("block mask had no effect on the first block position")
	}
	// Store pass appends the block.
	if _, err := m.ForwardDiffusion(xb, cache, B, true); err != nil {
		t.Fatal(err)
	}
	if cache.Len != len(all) || cache.K[0].Dim(2) != len(all) {
		t.Fatalf("store pass: Len %d K %d", cache.Len, cache.K[0].Dim(2))
	}
	if got := transferSchedule(4, 4); got[0] != 1 || got[3] != 1 {
		t.Fatalf("schedule %v", got)
	}
	if got := transferSchedule(3, 4); got[0] != 1 || got[2] != 1 || got[3] != 0 {
		t.Fatalf("schedule %v", got)
	}
}

const sdarRepo = "JetLM/SDAR-1.7B-Chat"

func sdarCUDA(t *testing.T) device.Device {
	t.Helper()
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--JetLM--SDAR-1.7B-Chat/snapshots/*/model.safetensors")); len(m) == 0 {
		t.Skip(sdarRepo + " not in HF cache")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
	return dev
}

// Reference: scripts/verify/ref_sdar.py JetLM/SDAR-1.7B-Chat bf16 cuda
// (transformers 5.21 Qwen3ForCausalLM with the block mask, BF16).
var sdarPromptIDs = []int64{151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198}

func TestSDARMatchesReference(t *testing.T) {
	dev := sdarCUDA(t)
	m, err := LoadCausalLMFromHF(sdarRepo, dev, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	if !m.Cfg.QKNorm || m.Cfg.NumLayers != 28 {
		t.Fatalf("config: %+v", m.Cfg)
	}
	opts := DefaultSDAROpts()
	// Block 0, first denoising pass: prompt has 15 ids = 3 complete blocks
	// + 3, so block 0 is [<|im_start|>, assistant, \n, <|MASK|>].
	cache := NewKVCache(m.NumLayers())
	xp, _ := tensor.New(sdarPromptIDs[:12], []int{1, 12}, dev)
	if _, err := m.ForwardDiffusion(xp, cache, opts.BlockLength, true); err != nil {
		t.Fatal(err)
	}
	blk := append(append([]int64{}, sdarPromptIDs[12:]...), opts.MaskID)
	xb, _ := tensor.New(blk, []int{1, 4}, dev)
	logits, err := m.ForwardDiffusion(xb, cache, opts.BlockLength, false)
	if err != nil {
		t.Fatal(err)
	}
	cpu, _ := logits.To(device.CPU())
	V := m.VocabSize()
	row := tensor.MustSlice[float32](cpu)[3*V : 4*V]
	got := top5(row)
	wantIDs := []int{17, 13874, 785, 18, 19}
	wantLogits := []float32{54.0, 41.25, 40.5, 39.25, 39.0}
	for i := range 3 {
		if got[i] != wantIDs[i] {
			t.Errorf("top-%d id %d (%.2f), want %d (%.2f)", i+1, got[i], row[got[i]], wantIDs[i], wantLogits[i])
		}
	}
	for i, id := range wantIDs {
		if d := math.Abs(float64(row[id] - wantLogits[i])); d > 1.0 {
			t.Errorf("logit[%d] = %.3f, want %.3f (BF16 tol 1.0)", id, row[id], wantLogits[i])
		}
	}
	t.Logf("block0 pass0 top-5 %v logits %.2f %.2f %.2f %.2f %.2f", got, row[got[0]], row[got[1]], row[got[2]], row[got[3]], row[got[4]])
	// Greedy block diffusion, block 4 / 4 steps: "2 + 2 = 4<|im_end|>".
	opts.MaxTokens = 24
	gen, err := m.GenerateDiffusion(sdarPromptIDs, opts, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := []int64{17, 488, 220, 17, 284, 220, 19}
	if len(gen) != len(want) {
		t.Fatalf("greedy ids %v, want %v", gen, want)
	}
	for i := range want {
		if gen[i] != want[i] {
			t.Fatalf("greedy ids %v, want %v", gen, want)
		}
	}
	t.Logf("greedy ids %v", gen)
}
