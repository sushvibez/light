package models

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
)

// tiiuae/falcon-rw-1b (1.3B, ALiBi, multi-head, sequential blocks, biases;
// ships only pytorch_model.bin, read through light/pth). Reference:
// scripts/verify/ref_falcon.py tiiuae/falcon-rw-1b f32 --sdpa (CUDA).
const falconRW1BRepo = "tiiuae/falcon-rw-1b"

var falconRW1BIDs = []int64{464, 3139, 286, 4881, 318, 6342, 13, 383, 3139, 286, 4486, 318, 11307, 13, 383, 3139, 286, 8031, 318}

var falconRW1BRef = []struct {
	id    int
	logit float32
}{{10598, 35.2608}, {21574, 31.7368}, {28537, 30.4928}, {29600, 30.3152}, {29702, 29.2224}}

var falconRW1BGreedy = []int64{10598, 13, 383, 3139, 286, 262, 1578, 7526}

func TestFalconRW1BMatchesTransformersF32(t *testing.T) {
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--tiiuae--falcon-rw-1b/snapshots/*/pytorch_model.bin")); len(m) == 0 {
		t.Skip(falconRW1BRepo + " not in HF cache")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA (1.3B params: CPU F32 is over the host budget)")
	}
	requireHostMemGB(t, 8)
	releaseModelMemory(t)
	os.Setenv("HF_HUB_OFFLINE", "1")
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	m, err := LoadFalconFromHF(falconRW1BRepo, dev, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if !c.Alibi || c.MultiQuery || c.ParallelAttn || !c.Bias || c.NumLayers != 24 || c.NumHeads != 32 || c.VocabSize != 50304 {
		t.Fatalf("config %+v", c)
	}
	t.Logf("params %.3f B", float64(m.ParamCount())/1e9)
	row := lastLogits(t, m, falconRW1BIDs, NewKVCache(m.NumLayers()), dev)
	got := top5(row)
	for i, w := range falconRW1BRef {
		if got[i] != w.id {
			t.Errorf("top-%d: got id %d (%.3f), want %d (%.3f)", i+1, got[i], row[got[i]], w.id, w.logit)
		}
		if d := math.Abs(float64(row[w.id] - w.logit)); d > 0.05 {
			t.Errorf("logit[%d] = %.4f, want %.4f (F32 tol 0.05)", w.id, row[w.id], w.logit)
		}
	}
	t.Logf("top-5 %v logits %.4f %.4f %.4f %.4f %.4f", got, row[got[0]], row[got[1]], row[got[2]], row[got[3]], row[got[4]])
	cache := NewKVCache(m.NumLayers())
	var out []int64
	step := lastLogits(t, m, falconRW1BIDs, cache, dev)
	for len(out) < len(falconRW1BGreedy) {
		next := int64(top5(step)[0])
		out = append(out, next)
		step = lastLogits(t, m, []int64{next}, cache, dev)
	}
	t.Logf("greedy %v", out)
	for i := range falconRW1BGreedy {
		if out[i] != falconRW1BGreedy[i] {
			t.Fatalf("greedy ids %v, want %v", out, falconRW1BGreedy)
		}
	}
}
