package models

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

func tinyBertConfig() BertConfig {
	return BertConfig{
		VocabSize: 50, HiddenSize: 16, NumLayers: 2, NumHeads: 4,
		IntermediateSize: 32, MaxPosition: 12, TypeVocabSize: 2,
		LayerNormEps: 1e-12, HiddenAct: "gelu", PadTokenID: 0,
	}
}

// randomizeBert fills the (zero-initialised) embedding tables with random
// values so tokens are distinguishable; the Linear layers are already
// Kaiming-initialised.
func randomizeBert(t *testing.T, m *BertModel) {
	t.Helper()
	for _, w := range []*tensor.Tensor{m.WordEmb.Weight, m.PosEmb.Weight, m.TypeEmb.Weight} {
		r, err := tensor.Rand(dtype.F32, w.Shape().Dims(), device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		if err := w.CopyFrom(r); err != nil {
			t.Fatal(err)
		}
	}
}

func i64Tensor(t *testing.T, data []int64, dims []int, dev device.Device) *tensor.Tensor {
	t.Helper()
	x, err := tensor.New(data, dims, dev)
	if err != nil {
		t.Fatal(err)
	}
	return x
}

func maxAbsDiff(a, b []float32) float64 {
	m := 0.0
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > m {
			m = d
		}
	}
	return m
}

func TestBertShapes(t *testing.T) {
	cfg := tinyBertConfig()
	m, err := NewBertModel(cfg, true, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	randomizeBert(t, m)
	B, T := 2, 7
	ids := make([]int64, B*T)
	for i := range ids {
		ids[i] = int64(i%cfg.VocabSize) + 1
	}
	x := i64Tensor(t, ids, []int{B, T}, device.CPU())
	h, err := m.Forward(x, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := h.Shape().Dims(); len(got) != 3 || got[0] != B || got[1] != T || got[2] != cfg.HiddenSize {
		t.Fatalf("hidden shape %v", got)
	}
	for _, v := range tensor.MustSlice[float32](h) {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("non-finite hidden state")
		}
	}
	pooled, err := m.Pool(h)
	if err != nil {
		t.Fatal(err)
	}
	if got := pooled.Shape().Dims(); len(got) != 2 || got[0] != B || got[1] != cfg.HiddenSize {
		t.Fatalf("pooled shape %v", got)
	}
	cls, err := CLSPooled(h)
	if err != nil {
		t.Fatal(err)
	}
	mean, err := MeanPooled(h, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []*tensor.Tensor{cls, mean} {
		if got := p.Shape().Dims(); got[0] != B || got[1] != cfg.HiddenSize {
			t.Fatalf("pool shape %v", got)
		}
	}
	n, err := Normalize(cls)
	if err != nil {
		t.Fatal(err)
	}
	vals := tensor.MustSlice[float32](n)
	for b := range B {
		s := 0.0
		for j := range cfg.HiddenSize {
			s += float64(vals[b*cfg.HiddenSize+j]) * float64(vals[b*cfg.HiddenSize+j])
		}
		if math.Abs(s-1) > 1e-5 {
			t.Fatalf("row %d norm² = %v", b, s)
		}
	}
	if m.ParamCount() == 0 {
		t.Fatal("ParamCount == 0")
	}
}

func TestBertMLMShapes(t *testing.T) {
	cfg := tinyBertConfig()
	m, err := NewBertForMaskedLM(cfg, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	randomizeBert(t, m.Bert)
	x := i64Tensor(t, []int64{1, 2, 3, 4, 5, 6}, []int{1, 6}, device.CPU())
	logits, err := m.Forward(x, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := logits.Shape().Dims(); got[0] != 1 || got[1] != 6 || got[2] != cfg.VocabSize {
		t.Fatalf("logits shape %v", got)
	}
	if m.Decoder.W != m.Bert.WordEmb.Weight {
		t.Fatal("decoder weight should be tied to the word embeddings")
	}
}

// TestBertPaddingMask checks that padded positions do not change the outputs
// of real positions: running [a b c d PAD PAD] with a mask equals running
// [a b c d] alone, and the MeanPooled helper ignores padding.
func TestBertPaddingMask(t *testing.T) {
	cfg := tinyBertConfig()
	m, err := NewBertModel(cfg, false, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	randomizeBert(t, m)
	cpu := device.CPU()
	H := cfg.HiddenSize

	short := i64Tensor(t, []int64{5, 9, 13, 21}, []int{1, 4}, cpu)
	ref, err := m.Forward(short, nil)
	if err != nil {
		t.Fatal(err)
	}
	refV := tensor.MustSlice[float32](ref)

	// Batch of 2: row 0 is a full-length sequence, row 1 is `short` + 2 pads.
	ids := i64Tensor(t, []int64{
		7, 8, 9, 10, 11, 12,
		5, 9, 13, 21, 0, 0,
	}, []int{2, 6}, cpu)
	for _, maskDT := range []dtype.DType{dtype.I64, dtype.F32} {
		var mask *tensor.Tensor
		if maskDT == dtype.I64 {
			mask = i64Tensor(t, []int64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}, []int{2, 6}, cpu)
		} else {
			mask, _ = tensor.New([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}, []int{2, 6}, cpu)
		}
		out, err := m.Forward(ids, mask)
		if err != nil {
			t.Fatal(err)
		}
		outV := tensor.MustSlice[float32](out)
		row1 := outV[6*H : 6*H+4*H] // first 4 positions of batch row 1
		if d := maxAbsDiff(row1, refV); d > 1e-5 {
			t.Fatalf("mask %v: padded batch differs from unpadded run by %v", maskDT, d)
		}
		for _, v := range outV {
			if math.IsNaN(float64(v)) {
				t.Fatal("NaN in masked output")
			}
		}
		// Without a mask the pads leak into real positions (sanity check that
		// the mask is actually doing something).
		noMask, err := m.Forward(ids, nil)
		if err != nil {
			t.Fatal(err)
		}
		if d := maxAbsDiff(tensor.MustSlice[float32](noMask)[6*H:6*H+4*H], refV); d < 1e-4 {
			t.Fatalf("expected unmasked pads to perturb outputs, diff=%v", d)
		}
		// MeanPooled with mask == mean over the 4 real tokens.
		mp, err := MeanPooled(out, mask)
		if err != nil {
			t.Fatal(err)
		}
		mpV := tensor.MustSlice[float32](mp)
		for j := range H {
			want := float32(0)
			for p := range 4 {
				want += refV[p*H+j]
			}
			want /= 4
			if math.Abs(float64(mpV[H+j]-want)) > 1e-5 {
				t.Fatalf("MeanPooled[1][%d] = %v, want %v", j, mpV[H+j], want)
			}
		}
	}
}

// TestBertCPUCUDAParity runs the same random tiny model on both devices.
func TestBertCPUCUDAParity(t *testing.T) {
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	cfg := tinyBertConfig()
	mc, err := NewBertModel(cfg, true, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	randomizeBert(t, mc)
	mg, err := NewBertModel(cfg, true, gpu)
	if err != nil {
		t.Fatal(err)
	}
	pc, pg := mc.Parameters(), mg.Parameters()
	for i := range pc {
		if err := pg[i].CopyFrom(pc[i]); err != nil {
			t.Fatal(err)
		}
	}
	ids := []int64{7, 8, 9, 10, 11, 12, 5, 9, 13, 21, 0, 0}
	mask := []int64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}
	hc, err := mc.Forward(i64Tensor(t, ids, []int{2, 6}, device.CPU()), i64Tensor(t, mask, []int{2, 6}, device.CPU()))
	if err != nil {
		t.Fatal(err)
	}
	hg, err := mg.Forward(i64Tensor(t, ids, []int{2, 6}, gpu), i64Tensor(t, mask, []int{2, 6}, gpu))
	if err != nil {
		t.Fatalf("CUDA forward: %v", err)
	}
	hgCPU, err := hg.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(tensor.MustSlice[float32](hc), tensor.MustSlice[float32](hgCPU)); d > 1e-4 {
		t.Fatalf("CPU/CUDA hidden diff %v", d)
	}
	pc1, _ := mc.Pool(hc)
	pg1, _ := mg.Pool(hg)
	pg1c, _ := pg1.To(device.CPU())
	if d := maxAbsDiff(tensor.MustSlice[float32](pc1), tensor.MustSlice[float32](pg1c)); d > 1e-4 {
		t.Fatalf("CPU/CUDA pooled diff %v", d)
	}
}
