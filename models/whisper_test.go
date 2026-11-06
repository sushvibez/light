package models

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/audio"
	"light/device"
	"light/hub"
	"light/tensor"
	"light/tokenizer"
)

const whisperRepo = "openai/whisper-tiny"

func whisperCached(t *testing.T) {
	t.Helper()
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--openai--whisper-tiny/snapshots/*/model.safetensors")); len(m) == 0 {
		t.Skip("openai/whisper-tiny not in HF cache")
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
}

func approxRow(t *testing.T, name string, got []float32, want []float32, tol float64) {
	t.Helper()
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > tol {
			t.Errorf("%s[%d] = %.5f, want %.5f (tol %g)", name, i, got[i], want[i], tol)
		}
	}
}

// Reference values: scripts/verify/ref_whisper.py (transformers 5.7, F32).
func testWhisperTiny(t *testing.T, dev device.Device) {
	samples, rate, err := audio.ReadWAV("../testdata/whisper/front_center_16k.wav")
	if err != nil {
		t.Fatal(err)
	}
	if rate != 16000 || len(samples) != 22849 {
		t.Fatalf("wav: rate %d, %d samples", rate, len(samples))
	}
	filters, err := LoadWhisperMelFilters(whisperRepo)
	if err != nil {
		t.Fatal(err)
	}
	mel, err := audio.LogMel(samples, filters)
	if err != nil {
		t.Fatal(err)
	}
	F := audio.WhisperFrames
	approxRow(t, "mel[0]", mel[0:6], []float32{-0.72754, -0.59202, -0.19451, -0.35730, -0.14826, 0.03731}, 2e-3)
	approxRow(t, "mel[40][100:]", mel[40*F+100:40*F+106], []float32{0.79388, 0.38610, 0.18774, -0.00183, -0.05535, -0.05922}, 2e-3)
	approxRow(t, "mel[79][140:]", mel[79*F+140:79*F+146], []float32{-0.72754, -0.72754, -0.72754, -0.72754, -0.72754, -0.72754}, 2e-3)
	var sum float64
	for _, v := range mel {
		sum += float64(v)
	}
	if math.Abs(sum+169069.812) > 30 {
		t.Errorf("mel sum %.3f, want -169069.812", sum)
	}

	m, err := LoadWhisperFromHF(whisperRepo, dev)
	if err != nil {
		t.Fatal(err)
	}
	if n := m.ParamCount(); n < 37_000_000 || n > 39_000_000 {
		t.Fatalf("param count %d", n)
	}
	melT, err := tensor.New(mel, []int{1, 80, F}, dev)
	if err != nil {
		t.Fatal(err)
	}
	enc, err := m.Encode(melT)
	if err != nil {
		t.Fatal(err)
	}
	encCPU, _ := enc.To(device.CPU())
	e := tensor.MustSlice[float32](encCPU)
	D := m.Cfg.DModel
	approxRow(t, "enc[0]", e[0:6], []float32{0.13737, 0.02052, 0.15564, 0.16854, 0.04277, 0.06231}, 1e-2)
	approxRow(t, "enc[70]", e[70*D:70*D+6], []float32{1.02572, 2.11026, 0.63304, -0.69588, -0.42404, -0.62291}, 1e-2)
	approxRow(t, "enc[1499]", e[1499*D:1499*D+6], []float32{0.22203, -0.10930, -1.38305, 0.57577, -0.57192, 4.18603}, 1e-2)

	cache, err := m.NewDecoderCache(enc)
	if err != nil {
		t.Fatal(err)
	}
	row, err := m.decodeRow([]int64{50258, 50259, 50359, 50363}, enc, cache)
	if err != nil {
		t.Fatal(err)
	}
	got := top5(row)
	want := []struct {
		id    int
		logit float32
	}{{17348, 14.3810}, {1868, 12.2400}, {22812, 11.7869}, {479, 11.2500}, {503, 11.1976}}
	for i, w := range want {
		if got[i] != w.id {
			t.Errorf("step0 top-%d: got %d (%.4f) want %d", i+1, got[i], row[got[i]], w.id)
		}
		if d := math.Abs(float64(row[w.id] - w.logit)); d > 2e-2 {
			t.Errorf("step0 logit[%d] = %.4f want %.4f", w.id, row[w.id], w.logit)
		}
	}

	ids, err := m.Transcribe(melT, TranscribeOpts{Language: "en"})
	if err != nil {
		t.Fatal(err)
	}
	wantIDs := []int64{17348, 5169, 13}
	if len(ids) != len(wantIDs) {
		t.Fatalf("ids %v want %v", ids, wantIDs)
	}
	for i := range ids {
		if ids[i] != wantIDs[i] {
			t.Fatalf("ids %v want %v", ids, wantIDs)
		}
	}
	tokPath, err := hub.Download(whisperRepo, "main", "tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	f, _ := os.Open(tokPath)
	bpe, err := tokenizer.LoadBPE(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	text := bpe.Decode([]int32{17348, 5169, 13})
	if text != " Front Center." {
		t.Fatalf("decoded %q", text)
	}
	t.Logf("transcript %q", text)

	// Language detection path (no Language given) must land on <|en|>.
	ids, err = m.Transcribe(melT, TranscribeOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 3 || ids[0] != 17348 {
		t.Fatalf("auto-language ids %v", ids)
	}

	// The original 48 kHz file through our resampler must transcribe the same.
	if raw, rate, err := audio.ReadWAV("/usr/share/sounds/alsa/Front_Center.wav"); err == nil {
		s16 := audio.Resample(raw, rate, 16000)
		mel2, err := audio.LogMel(s16, filters)
		if err != nil {
			t.Fatal(err)
		}
		melT2, _ := tensor.New(mel2, []int{1, 80, F}, dev)
		ids, err := m.Transcribe(melT2, TranscribeOpts{Language: "en"})
		if err != nil {
			t.Fatal(err)
		}
		if got := bpe.Decode(toI32(ids)); got != " Front Center." {
			t.Errorf("48 kHz via Resample: %q", got)
		}
	}
}

func toI32(ids []int64) []int32 {
	out := make([]int32, len(ids))
	for i, v := range ids {
		out[i] = int32(v)
	}
	return out
}

func TestWhisperTinyCPU(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1: the CPU encoder pass takes ~3 minutes")
	}
	whisperCached(t)
	testWhisperTiny(t, device.CPU())
}

func TestWhisperTinyCUDA(t *testing.T) {
	whisperCached(t)
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	testWhisperTiny(t, dev)
}
