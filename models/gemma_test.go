package models

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
)

// gemmaChatIDs is "<bos><start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n".
var gemmaChatIDs = []int64{2, 105, 2364, 107, 3689, 563, 236743, 236778, 236862, 236778, 236881, 106, 107, 105, 4368, 107}

const gemmaRepo = "unsloth/gemma-3-1b-it"

func gemmaCUDA(t *testing.T) device.Device {
	t.Helper()
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--unsloth--gemma-3-1b-it/snapshots/*/model.safetensors")); len(m) == 0 {
		t.Skip("unsloth/gemma-3-1b-it not in HF cache")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
	requireHostMemGB(t, 8)
	releaseModelMemory(t)
	return dev
}

// Reference: scripts/verify/ref_gemma.py unsloth/gemma-3-1b-it f32
// (transformers 5.7, eager, CUDA).
func TestGemma3MatchesTransformersF32(t *testing.T) {
	dev := gemmaCUDA(t)
	m, err := LoadGemma3FromHF(gemmaRepo, dev, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if got := m.ParamCount(); got < 990_000_000 || got > 1_010_000_000 {
		t.Fatalf("param count %d", got)
	}
	row := lastLogits(t, m, gemmaChatIDs, NewKVCache(m.NumLayers()), dev)
	want := []struct {
		id    int
		logit float32
	}{{236778, 54.9100}, {11634, 35.7269}, {818, 30.4699}, {238204, 30.1377}, {236743, 28.9408}}
	got := top5(row)
	for i, w := range want {
		if got[i] != w.id {
			t.Errorf("top-%d: got id %d (%.4f), want %d (%.4f)", i+1, got[i], row[got[i]], w.id, w.logit)
		}
		if d := math.Abs(float64(row[w.id] - w.logit)); d > 0.05 {
			t.Errorf("logit[%d] = %.4f, want %.4f", w.id, row[w.id], w.logit)
		}
	}
	t.Logf("top-1 %d logit %.4f (ref 54.9100)", got[0], row[got[0]])
}

// Chunked decode through the rotating cache (local layers window 512, so
// the window never trims on this prompt, but the mask/base split per layer
// and the cache path must still match the single-shot forward).
func TestGemma3CacheConsistency(t *testing.T) {
	dev := gemmaCUDA(t)
	m, err := LoadGemma3FromHF(gemmaRepo, dev, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	full := lastLogits(t, m, gemmaChatIDs, NewKVCache(m.NumLayers()), dev)
	cache := NewKVCache(m.NumLayers())
	lastLogits(t, m, gemmaChatIDs[:7], cache, dev)
	var step []float32
	for i := 7; i < len(gemmaChatIDs); i++ {
		step = lastLogits(t, m, gemmaChatIDs[i:i+1], cache, dev)
	}
	if top5(full)[0] != 236778 || top5(step)[0] != 236778 {
		t.Fatalf("BF16 top-1: full %d chunked %d, want 236778", top5(full)[0], top5(step)[0])
	}
	for _, id := range top5(full) {
		if d := math.Abs(float64(full[id] - step[id])); d > 0.5 {
			t.Errorf("logit %d: full %.3f chunked %.3f", id, full[id], step[id])
		}
	}
	// The forced small-window check lives in TestGemma3CacheConsistencyF32Window.
}

// TestGemma3CacheConsistencyF32Window forces the sliding window to bite
// (window 5 on a 16-token prompt) and checks that chunked decode through
// the rotating cache equals the single-shot forward. It runs in F32: window
// 5 is far from what the model was trained on, and in BF16 the two paths
// drift apart through rounding alone (A100: ~0.9 logit, while the F32 gap
// is 1.6e-4), so a BF16 version could not tell a cache / mask bug from
// precision. Here a gap means a real bug.
func TestGemma3CacheConsistencyF32Window(t *testing.T) {
	dev := gemmaCUDA(t)
	m, err := LoadGemma3FromHF(gemmaRepo, dev, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	m.Cfg.SlidingWindow = 5
	full := lastLogits(t, m, gemmaChatIDs, NewKVCache(m.NumLayers()), dev)
	cache := NewKVCache(m.NumLayers())
	lastLogits(t, m, gemmaChatIDs[:4], cache, dev)
	lastLogits(t, m, gemmaChatIDs[4:9], cache, dev)
	var step []float32
	for i := 9; i < len(gemmaChatIDs); i++ {
		step = lastLogits(t, m, gemmaChatIDs[i:i+1], cache, dev)
	}
	// Every layer keeps every position (the window is applied by the
	// mask and the decode kernel, not by trimming the cache).
	for l, k := range cache.K {
		if k.Dim(2) != len(gemmaChatIDs) {
			t.Fatalf("layer %d holds %d keys, want %d", l, k.Dim(2), len(gemmaChatIDs))
		}
	}
	d := maxAbsDiff(full, step)
	top := top5(full)
	t.Logf("F32 window 5: max |full - chunked| = %.3g; top-5 %v full %.3f %.3f chunked %.3f %.3f",
		d, top, full[top[0]], full[top[1]], step[top[0]], step[top[1]])
	if d > 0.05 {
		t.Errorf("F32 window 5: chunked and single-shot logits differ by %.3g", d)
	}
}
