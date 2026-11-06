package models

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"light/audio"
	"light/device"
	"light/dtype"
	"light/hub"
	"light/ops"
	"light/tensor"
	"light/tokenizer"
)

const qwen3ASRRepo = "Qwen/Qwen3-ASR-1.7B-hf"

func qwen3ASRCached(t *testing.T) {
	t.Helper()
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B-hf/snapshots/*/model.safetensors")); len(m) == 0 {
		t.Skip("Qwen/Qwen3-ASR-1.7B-hf not in HF cache")
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
}

// qwen3ASRMel is the Qwen3ASRFeatureExtractor: 16 kHz audio padded to at
// least 8000 samples, 128-bin slaney log-mel with len/160 frames, frames
// zero-padded to a multiple of 100. Returns the mel and the valid count.
func qwen3ASRMel(t *testing.T, samples []float32, filters [][]float32) ([]float32, int) {
	t.Helper()
	if len(samples) < 8000 {
		samples = append(slices.Clone(samples), make([]float32, 8000-len(samples))...)
	}
	mel, frames, err := audio.LogMelFrames(samples, filters)
	if err != nil {
		t.Fatal(err)
	}
	padded := (frames + 99) / 100 * 100
	out := make([]float32, len(filters)*padded)
	for b := range filters {
		copy(out[b*padded:], mel[b*frames:(b+1)*frames])
	}
	return out, frames
}

// Reference values: scripts/verify/ref_qwen3_asr.py (transformers 5.21, F32, CPU).
var qwen3ASRRefPrompt = []int64{151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669}
var qwen3ASRRefGen = []int64{11528, 6364, 151704, 23395, 11, 4126, 13} // "language English<asr_text>Front, center."

func TestQwen3ASRFrontEnd(t *testing.T) {
	fb := audio.MelFilterBank(128, 400, 16000, 0, 8000)
	approxRow(t, "fb[0]", fb[0][0:6], []float32{0, 0.01237, 0, 0, 0, 0}, 1e-5)
	approxRow(t, "fb[127][190:]", fb[127][190:196], []float32{0, 0.00048, 0.00162, 0.00276, 0.00390, 0.00504}, 1e-5)
	var sum float64
	for _, r := range fb {
		for _, v := range r {
			sum += float64(v)
		}
	}
	if math.Abs(sum-3.190986) > 1e-4 {
		t.Errorf("filterbank sum %.6f, want 3.190986", sum)
	}
	samples, rate, err := audio.ReadWAV("../testdata/whisper/front_center_16k.wav")
	if err != nil {
		t.Fatal(err)
	}
	if rate != 16000 {
		t.Fatalf("rate %d", rate)
	}
	mel, valid := qwen3ASRMel(t, samples, fb)
	if valid != 142 || len(mel) != 128*200 {
		t.Fatalf("valid %d, mel len %d (want 142, %d)", valid, len(mel), 128*200)
	}
	F := 200
	approxRow(t, "mel[0]", mel[0:6], []float32{-0.67384, -0.66778, -0.27027, -0.43306, -0.22402, -0.03845}, 2e-3)
	approxRow(t, "mel[40][100:]", mel[40*F+100:40*F+106], []float32{0.44933, 0.28803, 0.26762, 0.06830, -0.05977, -0.38044}, 2e-3)
	approxRow(t, "mel[127][140:]", mel[127*F+140:127*F+146], []float32{-0.67384, -0.67384, 0, 0, 0, 0}, 2e-3)
	sum = 0
	for _, v := range mel {
		sum += float64(v)
	}
	if math.Abs(sum+4323.672) > 1 {
		t.Errorf("mel sum %.3f, want -4323.672", sum)
	}
	cfg := Qwen3ASRConfig{NWindow: 50, ChunkPositions: 13}
	if n := cfg.AudioTokens(valid); n != 19 {
		t.Errorf("AudioTokens(142) = %d, want 19", n)
	}
	if n := cfg.AudioTokens(3000); n != 390 {
		t.Errorf("AudioTokens(3000) = %d, want 390", n)
	}
}

func testQwen3ASR(t *testing.T, dev device.Device, lmDT dtype.DType) {
	qwen3ASRCached(t)
	fb := audio.MelFilterBank(128, 400, 16000, 0, 8000)
	samples, _, err := audio.ReadWAV("../testdata/whisper/front_center_16k.wav")
	if err != nil {
		t.Fatal(err)
	}
	mel, valid := qwen3ASRMel(t, samples, fb)

	t0 := time.Now()
	m, err := LoadQwen3ASRFromHF(qwen3ASRRepo, dev, lmDT)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("load %.1fs, %.0fM params", time.Since(t0).Seconds(), float64(m.ParamCount())/1e6)
	if m.Cfg.AudioStartID != 151669 || m.Cfg.AudioEndID != 151670 || m.Cfg.ASRTextID != 151704 || m.Cfg.AudioTokenID != 151676 {
		t.Fatalf("special ids %+v", m.Cfg)
	}

	// Prompt ids match the processor, and the tokenizer agrees on the
	// "language English" prefill the engine will feed for forced language.
	ids, off := m.PromptIDs(19, nil, nil)
	if !slices.Equal(ids[:9], qwen3ASRRefPrompt) || off != 9 || len(ids) != 34 || ids[9] != 151676 || ids[28] != 151670 || ids[33] != 198 {
		t.Fatalf("prompt ids %v (offset %d)", ids, off)
	}
	tokPath, err := hub.Download(qwen3ASRRepo, "main", "tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(tokPath)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.LoadBPE(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Encode("language English"); !slices.Equal(got, []int32{11528, 6364}) {
		t.Errorf("tokenize 'language English' = %v", got)
	}

	t1 := time.Now()
	enc, err := m.Encode(mel, valid)
	if err != nil {
		t.Fatal(err)
	}
	encCPU, _ := enc.To(device.CPU())
	e := tensor.MustSlice[float32](encCPU)
	D := m.Cfg.DModel
	if enc.Dim(0) != 19 || enc.Dim(1) != D {
		t.Fatalf("enc shape %v", enc.Shape())
	}
	approxRow(t, "enc[0]", e[0:6], []float32{-0.08325, -0.10395, 0.06469, 0.48413, -0.16898, 0.16005}, 1e-2)
	approxRow(t, "enc[10]", e[10*D:10*D+6], []float32{-0.03951, 0.37931, 0.07388, 0.19839, -0.09277, 0.26020}, 1e-2)
	approxRow(t, "enc[18]", e[18*D:18*D+6], []float32{-0.38457, -0.03998, -0.04586, -0.26954, 0.14263, -0.28893}, 1e-2)
	var sum float64
	for _, v := range e {
		sum += float64(v)
	}
	if math.Abs(sum-20.564) > 0.5 {
		t.Errorf("enc sum %.3f, want 20.564", sum)
	}
	audioEmb, err := m.Project(enc)
	if err != nil {
		t.Fatal(err)
	}
	pCPU, _ := audioEmb.To(device.CPU())
	if pCPU.DType() != dtype.F32 {
		pCPU, _ = castTo(pCPU, dtype.F32)
	}
	p := tensor.MustSlice[float32](pCPU)
	O := m.Cfg.OutputDim
	ptol := 2e-3
	if lmDT == dtype.BF16 {
		ptol = 1e-2
	}
	approxRow(t, "proj[0]", p[0:6], []float32{-0.05024, 0.00208, 0.07123, -0.00376, -0.00028, -0.03084}, ptol)
	approxRow(t, "proj[18]", p[18*O:18*O+6], []float32{-0.00041, -0.00961, -0.00862, 0.01840, 0.00673, -0.02133}, ptol)
	t.Logf("encode+project %.0fms", float64(time.Since(t1).Milliseconds()))

	t2 := time.Now()
	cache := NewKVCache(m.LM.Cfg.NumLayers)
	row, err := m.Prefill(ids, off, audioEmb, cache)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("prefill %.0fms", float64(time.Since(t2).Milliseconds()))
	got := top5(row)
	want := []struct {
		id    int
		logit float32
	}{{11528, 37.7779}, {4128, 24.6275}, {42824, 20.9654}, {102064, 19.7198}, {118732, 19.2004}}
	ltol := 0.05
	if lmDT == dtype.BF16 {
		ltol = 0.6
	}
	for i, w := range want {
		if got[i] != w.id {
			t.Errorf("step0 top-%d: got %d (%.4f) want %d (%.4f)", i+1, got[i], row[got[i]], w.id, w.logit)
		}
		if d := math.Abs(float64(row[w.id] - w.logit)); d > ltol {
			t.Errorf("step0 logit[%d] = %.4f want %.4f", w.id, row[w.id], w.logit)
		}
	}

	t3 := time.Now()
	gen, err := m.TranscribeAudio(audioEmb, nil, nil, 32)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("greedy %d tokens %.0fms: %v %q", len(gen), float64(time.Since(t3).Milliseconds()), gen, tok.Decode(toInt32(gen)))
	if !slices.Equal(gen, qwen3ASRRefGen) {
		t.Errorf("greedy ids %v, want %v", gen, qwen3ASRRefGen)
	}
	// Forced language: prefill "language English<asr_text>" and the model
	// emits only the text.
	prefill := append(toInt64(tok.Encode("language English")), m.Cfg.ASRTextID)
	gen, err = m.TranscribeAudio(audioEmb, nil, prefill, 32)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(gen, qwen3ASRRefGen[3:]) {
		t.Errorf("forced-language ids %v, want %v", gen, qwen3ASRRefGen[3:])
	}
}

func castTo(x *tensor.Tensor, dt dtype.DType) (*tensor.Tensor, error) {
	return ops.Cast(x, dt)
}

func toInt32(ids []int64) []int32 {
	out := make([]int32, len(ids))
	for i, v := range ids {
		out[i] = int32(v)
	}
	return out
}

func toInt64(ids []int32) []int64 {
	out := make([]int64, len(ids))
	for i, v := range ids {
		out[i] = int64(v)
	}
	return out
}

// The CPU/F32 variant needs ~12 GB of host RAM (8 GB F32 decoder + the
// BF16 source during load) and runs for minutes; it is opt-in so that
// `go test ./...` never loads it beside other model tests.
func TestQwen3ASRMatchesTransformersCPU(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 to run the 2B F32 decoder on the CPU")
	}
	testQwen3ASR(t, device.CPU(), dtype.F32)
}

func TestQwen3ASRCUDA(t *testing.T) {
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	testQwen3ASR(t, dev, dtype.BF16)
}
