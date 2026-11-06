package models

import (
	"math"
	"testing"

	"light/device"
	"light/hub"
	"light/tensor"
	"light/tokenizer"
)

// Reference values produced by scripts/verify/ref_bert.py (transformers
// AutoModel on unsloth/bge-small-en-v1.5, float32, CPU).
const bgeRepo = "unsloth/bge-small-en-v1.5"

var bgeSentences = []string{
	"The cat sits on the mat.",
	"A feline rests on a rug.",
	"Stock markets fell sharply today.",
	"Go is a statically typed language.",
}

var bgeRefIDs = [][]int64{
	{101, 1996, 4937, 7719, 2006, 1996, 13523, 1012, 102},
	{101, 1037, 10768, 4179, 16626, 2006, 1037, 20452, 1012, 102},
	{101, 4518, 6089, 3062, 9249, 2651, 1012, 102},
	{101, 2175, 2003, 1037, 10763, 3973, 21189, 2653, 1012, 102},
}

var bgeRefHidden00 = []float32{0.326504, -0.195990, 0.088764, 0.482574, -0.245057, -0.112331, 0.974265, 0.659491}

var bgeRefEmb = [][]float32{
	{0.036275, -0.021775, 0.009862, 0.053615, -0.027226, -0.012480, 0.108243, 0.073271},
	{-0.008684, -0.029866, 0.031511, 0.101351, 0.019765, 0.033023, 0.014264, 0.061070},
	{0.037298, -0.059827, -0.013710, 0.004789, 0.033453, -0.006072, -0.004473, 0.018642},
	{-0.036524, -0.042864, 0.007152, -0.049553, -0.073982, 0.013263, 0.046007, 0.000025},
}

var bgeRefCos = [][]float32{
	{1.000000, 0.756123, 0.324302, 0.346669},
	{0.756123, 1.000000, 0.331726, 0.375185},
	{0.324302, 0.331726, 1.000000, 0.485633},
	{0.346669, 0.375185, 0.485633, 1.000000},
}

var bgeRefPooler0 = []float32{0.037190, -0.084269, 0.062409, 0.132646, -0.049384, 0.016795, -0.146028, 0.058109}

// bgeBatch tokenizes the fixed sentences with [CLS]/[SEP], pads to the
// longest and returns (ids, mask) tensors on dev.
func bgeBatch(t *testing.T, dev device.Device) (*tensor.Tensor, *tensor.Tensor) {
	t.Helper()
	tokPath, err := hub.Download(bgeRepo, "main", "tokenizer.json")
	if err != nil {
		t.Skipf("tokenizer.json for %s unavailable: %v", bgeRepo, err)
	}
	tk, err := tokenizer.Load(tokPath)
	if err != nil {
		t.Fatal(err)
	}
	te, ok := tk.(tokenizer.TemplateEncoder)
	if !ok {
		t.Fatalf("tokenizer %T has no template encoder", tk)
	}
	B := len(bgeSentences)
	seqs := make([][]int32, B)
	T := 0
	for i, s := range bgeSentences {
		seqs[i] = te.EncodeWithSpecial(s)
		if len(seqs[i]) != len(bgeRefIDs[i]) {
			t.Fatalf("sentence %d: got %d ids %v, want %v", i, len(seqs[i]), seqs[i], bgeRefIDs[i])
		}
		for j, id := range seqs[i] {
			if int64(id) != bgeRefIDs[i][j] {
				t.Fatalf("sentence %d token %d: got %d, want %d", i, j, id, bgeRefIDs[i][j])
			}
		}
		T = max(T, len(seqs[i]))
	}
	ids := make([]int64, B*T)
	mask := make([]int64, B*T)
	for b, s := range seqs {
		for j, id := range s {
			ids[b*T+j] = int64(id)
			mask[b*T+j] = 1
		}
	}
	return i64Tensor(t, ids, []int{B, T}, dev), i64Tensor(t, mask, []int{B, T}, dev)
}

func loadBGE(t *testing.T, dev device.Device) *BertModel {
	t.Helper()
	if _, err := hub.Download(bgeRepo, "main", "model.safetensors"); err != nil {
		t.Skipf("checkpoint %s unavailable: %v", bgeRepo, err)
	}
	m, err := LoadBertFromHF(bgeRepo, dev)
	if err != nil {
		t.Fatalf("LoadBertFromHF: %v", err)
	}
	return m
}

func bgeEmbed(t *testing.T, m *BertModel, dev device.Device) (hidden, emb *tensor.Tensor) {
	t.Helper()
	ids, mask := bgeBatch(t, dev)
	hidden, err := m.Forward(ids, mask)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	cls, err := CLSPooled(hidden)
	if err != nil {
		t.Fatal(err)
	}
	emb, err = Normalize(cls)
	if err != nil {
		t.Fatal(err)
	}
	return hidden, emb
}

func toCPUSlice(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	c, err := x.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	return tensor.MustSlice[float32](c)
}

func checkBGEAgainstReference(t *testing.T, m *BertModel, dev device.Device, tol float64) {
	t.Helper()
	hidden, emb := bgeEmbed(t, m, dev)
	H := m.Cfg.HiddenSize
	hv := toCPUSlice(t, hidden)
	if d := maxAbsDiff(hv[:8], bgeRefHidden00); d > tol {
		t.Errorf("last_hidden_state[0,0,:8] max abs err %.2e > %.0e: got %v want %v", d, tol, hv[:8], bgeRefHidden00)
	} else {
		t.Logf("last_hidden_state[0,0,:8] max abs err vs PyTorch: %.2e", d)
	}
	ev := toCPUSlice(t, emb)
	worst := 0.0
	for i, want := range bgeRefEmb {
		got := ev[i*H : i*H+8]
		d := maxAbsDiff(got, want)
		worst = math.Max(worst, d)
		if d > tol {
			t.Errorf("emb[%d][:8] max abs err %.2e: got %v want %v", i, d, got, want)
		}
	}
	t.Logf("CLS embedding[:8] max abs err vs PyTorch: %.2e", worst)
	// Cosine matrix over the full 384-d embeddings.
	B := len(bgeSentences)
	for i := range B {
		for j := range B {
			s := 0.0
			for k := range H {
				s += float64(ev[i*H+k]) * float64(ev[j*H+k])
			}
			if math.Abs(s-float64(bgeRefCos[i][j])) > tol {
				t.Errorf("cos[%d][%d] = %.6f, want %.6f", i, j, s, bgeRefCos[i][j])
			}
		}
	}
	// Pooler output (tanh(W·CLS+b)).
	pooled, err := m.Pool(hidden)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(toCPUSlice(t, pooled)[:8], bgeRefPooler0); d > tol {
		t.Errorf("pooler_output[0,:8] max abs err %.2e", d)
	}
}

func TestBertBGEReferenceCPU(t *testing.T) {
	cpu := device.CPU()
	m := loadBGE(t, cpu)
	if m.Pooler == nil {
		t.Fatal("expected pooler in bge checkpoint")
	}
	checkBGEAgainstReference(t, m, cpu, 1e-3)
}

func TestBertBGEReferenceCUDA(t *testing.T) {
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	m := loadBGE(t, gpu)
	checkBGEAgainstReference(t, m, gpu, 1e-3)
}

// TestBertBGECPUCUDAParity compares the full last hidden state and the
// embeddings between CPU and CUDA runs of the real checkpoint.
func TestBertBGECPUCUDAParity(t *testing.T) {
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	cpu := device.CPU()
	mc := loadBGE(t, cpu)
	mg := loadBGE(t, gpu)
	hc, ec := bgeEmbed(t, mc, cpu)
	hg, eg := bgeEmbed(t, mg, gpu)
	dh := maxAbsDiff(toCPUSlice(t, hc), toCPUSlice(t, hg))
	de := maxAbsDiff(toCPUSlice(t, ec), toCPUSlice(t, eg))
	t.Logf("CPU vs CUDA: hidden max abs diff %.2e, embedding max abs diff %.2e", dh, de)
	if dh > 1e-3 || de > 1e-3 {
		t.Fatalf("CPU/CUDA parity: hidden %.2e emb %.2e", dh, de)
	}
}
