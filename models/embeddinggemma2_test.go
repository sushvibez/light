package models

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/tensor"
	"light/tokenizer"
)

// EmbeddingGemma 2 (unsloth/embeddinggemma-2, text encoder: 24 layers,
// d=512, projection-only PLE, 512-dim single-KV-head global layers,
// bidirectional sliding radius 512, 768-dim output). Reference values from
// scripts/verify/ref_embeddinggemma2.py (transformers eager, BF16, CUDA):
// unpadded per-text forward, mean pooling, L2 normalisation.
const embGemma2Repo = "unsloth/embeddinggemma-2"

var embGemma2Texts = []string{
	"task: search result | query: What causes the northern lights?",
	"title: none | text: The northern lights are caused by charged particles from the sun colliding with gases in the upper atmosphere.",
	"title: none | text: A sourdough starter needs regular feeding with flour and water to stay active.",
}

var embGemma2RefIDs0 = []int64{2, 8071, 236787, 3927, 1354, 1109, 7609, 236787, 2900, 9454, 506, 16962, 11544, 236881, 1}

var embGemma2RefEmb = [][]float32{
	{0.0119, -0.0026, 0.0406, -0.0184, -0.0407, 0.0112, -0.0113, -0.0103},
	{-0.0119, -0.0064, 0.0579, -0.0316, -0.0067, 0.022, 0.0041, -0.0396},
	{0.0417, -0.0163, 0.0483, -0.0189, -0.0192, 0.0561, -0.0401, -0.0198},
}

var embGemma2RefCos = [][]float32{
	{1, 0.8545, 0.5935},
	{0.8545, 1, 0.5789},
	{0.5935, 0.5789, 1},
}

func embGemma2Cached(t *testing.T) {
	t.Helper()
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--unsloth--embeddinggemma-2/snapshots/*/model.safetensors")); len(m) == 0 {
		t.Skip(embGemma2Repo + " not in HF cache")
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
}

// embGemma2IDs tokenises with the Gemma tokenizer.json and the checkpoint's
// TemplateProcessing (<bos> … <eos>).
func embGemma2IDs(t *testing.T) [][]int64 {
	t.Helper()
	path, err := hub.Download(embGemma2Repo, "main", "tokenizer.json")
	if err != nil {
		t.Skip(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	tok, err := tokenizer.LoadBPE(f)
	if err != nil {
		t.Fatal(err)
	}
	out := make([][]int64, len(embGemma2Texts))
	for i, s := range embGemma2Texts {
		ids := []int64{2}
		for _, id := range tok.Encode(s) {
			ids = append(ids, int64(id))
		}
		out[i] = append(ids, 1)
	}
	if len(out[0]) != len(embGemma2RefIDs0) {
		t.Fatalf("ids[0] = %v, want %v", out[0], embGemma2RefIDs0)
	}
	for i := range out[0] {
		if out[0][i] != embGemma2RefIDs0[i] {
			t.Fatalf("ids[0] = %v, want %v", out[0], embGemma2RefIDs0)
		}
	}
	return out
}

func embGemma2Load(t *testing.T, dev device.Device, dt dtype.DType) *EmbeddingGemma2 {
	t.Helper()
	m, err := LoadEmbeddingGemma2FromHF(embGemma2Repo, dev, dt)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if c.NumLayers != 24 || c.HiddenSize != 512 || c.GlobalHeadDim != 512 || c.NumGlobalKVHeads != 1 || c.HeadDim != 256 ||
		c.NumKVHeads != 2 || c.EmbeddingDim != 768 || c.PLEDim != 512 || c.SlidingWindow != 512 || c.isSliding(5) || !c.isSliding(4) ||
		m.Blocks[5].KProj.W.Dim(0) != 512 || m.Blocks[4].KProj.W.Dim(0) != 512 || m.Blocks[5].QProj.W.Dim(0) != 2048 {
		t.Fatalf("config %+v", c)
	}
	return m
}

// embGemma2Run embeds each text unpadded, then all three as a padded batch,
// and returns both (rows of 768).
func embGemma2Run(t *testing.T, m *EmbeddingGemma2, ids [][]int64, dev device.Device) (single, batched [][]float32) {
	t.Helper()
	for _, s := range ids {
		tk, err := tensor.New(s, []int{1, len(s)}, dev)
		if err != nil {
			t.Fatal(err)
		}
		e, err := m.Embed(tk, nil)
		if err != nil {
			t.Fatal(err)
		}
		single = append(single, toCPUSlice(t, e))
	}
	T := 0
	for _, s := range ids {
		T = max(T, len(s))
	}
	B := len(ids)
	flat, keep := make([]int64, B*T), make([]int64, B*T)
	for b, s := range ids {
		copy(flat[b*T:], s)
		for j := range s {
			keep[b*T+j] = 1
		}
	}
	tk, err := tensor.New(flat, []int{B, T}, dev)
	if err != nil {
		t.Fatal(err)
	}
	mk, err := tensor.New(keep, []int{B, T}, dev)
	if err != nil {
		t.Fatal(err)
	}
	e, err := m.Embed(tk, mk)
	if err != nil {
		t.Fatal(err)
	}
	all := toCPUSlice(t, e)
	for b := range ids {
		batched = append(batched, all[b*768:(b+1)*768])
	}
	return single, batched
}

func embGemma2Check(t *testing.T, emb [][]float32, tol float64) {
	t.Helper()
	for i, e := range emb {
		if d := maxAbsDiff(e[:8], embGemma2RefEmb[i]); d > tol {
			t.Errorf("emb[%d][:8] max abs err %.2e > %.0e: got %.4f want %v", i, d, tol, e[:8], embGemma2RefEmb[i])
		}
		for j := range emb {
			var s float64
			for k := range e {
				s += float64(e[k] * emb[j][k])
			}
			if math.Abs(s-float64(embGemma2RefCos[i][j])) > tol {
				t.Errorf("cos[%d][%d] = %.4f, want %.4f", i, j, s, embGemma2RefCos[i][j])
			}
		}
	}
	t.Logf("cos(q,doc)=%.4f cos(q,other)=%.4f", dot(emb[0], emb[1]), dot(emb[0], emb[2]))
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i] * b[i])
	}
	return s
}

func TestEmbeddingGemma2ReferenceCUDA(t *testing.T) {
	embGemma2Cached(t)
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	ids := embGemma2IDs(t)
	m := embGemma2Load(t, dev, dtype.BF16)
	t.Logf("text params %.1f M", float64(m.ParamCount())/1e6)
	single, batched := embGemma2Run(t, m, ids, dev)
	embGemma2Check(t, single, 1.5e-2)
	for i := range single {
		if d := maxAbsDiff(single[i], batched[i]); d > 1.5e-2 {
			t.Errorf("padded batch row %d differs from unpadded by %.2e", i, d)
		}
	}
}

// CPU F32 against the BF16 reference, then CPU↔CUDA parity at F32.
func TestEmbeddingGemma2ReferenceCPUAndParity(t *testing.T) {
	embGemma2Cached(t)
	requireHostMemGB(t, 4)
	releaseModelMemory(t)
	ids := embGemma2IDs(t)
	cpu := device.CPU()
	m := embGemma2Load(t, cpu, dtype.F32)
	single, batched := embGemma2Run(t, m, ids, cpu)
	embGemma2Check(t, single, 1.5e-2)
	for i := range single {
		if d := maxAbsDiff(single[i], batched[i]); d > 1e-4 {
			t.Errorf("padded batch row %d differs from unpadded by %.2e", i, d)
		}
	}
	if !device.CUDAAvailable() {
		return
	}
	m = nil
	releaseModelMemory(t)
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	mg := embGemma2Load(t, gpu, dtype.F32)
	gs, _ := embGemma2Run(t, mg, ids, gpu)
	for i := range single {
		if d := maxAbsDiff(single[i], gs[i]); d > 2e-3 {
			t.Errorf("CPU/CUDA F32 row %d differ by %.2e", i, d)
		}
	}
}
