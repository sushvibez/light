package models

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
)

// Tiny-random Qwen3.5 text model written by
// `scripts/verify/ref_qwen35.py --make-tiny`, reference values from
// `ref_qwen35.py testdata/qwen35/tiny f32 --tiny-ids` (transformers 5.21,
// eager, CPU). Layers: DeltaNet, full, DeltaNet, full; 2 key / 4 value
// heads in the DeltaNet layers; partial RoPE over 4 of 16 dims; untied head;
// norm weights and decay parameters perturbed away from their init values.
// The 70-token prompt makes the chunked delta rule cross a 64-token block.
const qwen35TinyDir = "testdata/qwen35/tiny"

var (
	qwen35TinyRef = []struct {
		id    int
		logit float32
	}{{309, 0.4402}, {34, 0.4399}, {259, 0.4161}, {254, 0.4030}, {278, 0.3841}}
	qwen35TinyGreedy = []int64{309, 59, 136, 157, 366, 115, 456, 5}
)

func qwen35TinyIDs() []int64 {
	ids := make([]int64, 70)
	for i := range ids {
		ids[i] = int64((37*i + 11) % 512)
	}
	return ids
}

func TestQwen35TinyMatchesTransformers(t *testing.T) {
	cpu := device.CPU()
	m, err := LoadQwen35FromHF(qwen35TinyDir, cpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if !slices.Equal(c.LinearLayers, []bool{true, false, true, false}) || c.LinearKHeads != 2 || c.LinearVHeads != 4 ||
		c.rotaryDim() != 4 || !c.AttnOutputGate || c.TieEmbed || c.RopeBase != 1e7 || c.ConvKernel != 4 {
		t.Fatalf("config %+v", c)
	}
	ids := qwen35TinyIDs()
	full := lastLogits(t, m, ids, NewKVCache(m.NumLayers()), cpu)
	got := top5(full)
	for i, w := range qwen35TinyRef {
		if i < 3 && got[i] != w.id {
			t.Errorf("top-%d: got %d (%.4f), want %d (%.4f)", i+1, got[i], full[got[i]], w.id, w.logit)
		}
		if d := full[w.id] - w.logit; d > 2e-3 || d < -2e-3 {
			t.Errorf("logit[%d] = %.4f, want %.4f", w.id, full[w.id], w.logit)
		}
	}

	// Greedy decode: one-token steps through the recurrent delta rule and
	// the carried conv / delta-rule / K-V state.
	cache := NewKVCache(m.NumLayers())
	step := lastLogits(t, m, ids, cache, cpu)
	var greedy []int64
	for len(greedy) < len(qwen35TinyGreedy) {
		next := int64(top5(step)[0])
		greedy = append(greedy, next)
		step = lastLogits(t, m, []int64{next}, cache, cpu)
	}
	if !slices.Equal(greedy, qwen35TinyGreedy) {
		t.Errorf("greedy %v, want %v", greedy, qwen35TinyGreedy)
	}

	// Prefill in two pieces (the first crossing a chunk boundary), then
	// single tokens == single shot.
	cache = NewKVCache(m.NumLayers())
	lastLogits(t, m, ids[:66], cache, cpu)
	var chunked []float32
	for i := 66; i < len(ids); i++ {
		chunked = lastLogits(t, m, ids[i:i+1], cache, cpu)
	}
	if d := maxAbsDiff(full, chunked); d > 1e-4 {
		t.Errorf("chunked vs single-shot differ by %.2e", d)
	}

	if !device.CUDAAvailable() {
		return
	}
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	mg, err := LoadQwen35FromHF(qwen35TinyDir, gpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(full, lastLogits(t, mg, ids, NewKVCache(mg.NumLayers()), gpu)); d > 1e-3 {
		t.Errorf("CUDA vs CPU logits differ by %.2e", d)
	}
}

// Qwen3.5-0.8B (hybrid DeltaNet / gated attention, text path of the
// multimodal checkpoint) against transformers 5.21 F32 eager on CPU:
//
//	python scripts/verify/ref_qwen35.py Qwen/Qwen3.5-0.8B f32 --chat
//
// The prompt is the chat-template rendering of the ref_phi4.py messages.
const qwen35SmallRepo = "Qwen/Qwen3.5-0.8B"

var (
	qwen35SmallChatIDs = []int64{248045, 8678, 198, 2523, 513, 264, 61446, 17313, 13, 248046, 198, 248045, 846, 198, 814, 20139,
		303, 799, 11316, 3069, 279, 12515, 369, 6105, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271}
	qwen35SmallRef = []struct {
		id    int
		logit float32
	}{{760, 19.0254}, {28232, 18.2804}, {13512, 18.0704}, {29108, 17.5299}, {42373, 17.1720}}
	qwen35SmallGreedy = []int64{760, 12515, 7701, 6105, 1521, 37728, 1100, 9872, 974, 279, 8964, 579, 16078, 11, 440, 22602}
)

func TestQwen35SmallMatchesTransformersF32(t *testing.T) {
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/config.json")); len(m) == 0 {
		t.Skip(qwen35SmallRepo + " not in HF cache")
	}
	devs := []device.Device{device.CPU()}
	if device.CUDAAvailable() {
		if gpu, err := device.CUDA(0); err == nil {
			devs = append(devs, gpu)
		}
	}
	for _, dev := range devs {
		t.Run(dev.String(), func(t *testing.T) {
			m, err := LoadQwen35FromHF(qwen35SmallRepo, dev, dtype.F32)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("params %.3f B", float64(m.ParamCount())/1e9)
			cache := NewKVCache(m.NumLayers())
			row := lastLogits(t, m, qwen35SmallChatIDs, cache, dev)
			got := top5(row)
			for i, w := range qwen35SmallRef {
				if i < 3 && got[i] != w.id {
					t.Errorf("top-%d: got %d (%.4f), want %d (%.4f)", i+1, got[i], row[got[i]], w.id, w.logit)
				}
				if d := row[w.id] - w.logit; d > 0.02 || d < -0.02 {
					t.Errorf("logit[%d] = %.4f, want %.4f", w.id, row[w.id], w.logit)
				}
			}
			t.Logf("top-5 %v logits %.4f %.4f %.4f %.4f %.4f", got, row[got[0]], row[got[1]], row[got[2]], row[got[3]], row[got[4]])
			var greedy []int64
			for len(greedy) < len(qwen35SmallGreedy) {
				next := int64(top5(row)[0])
				greedy = append(greedy, next)
				row = lastLogits(t, m, []int64{next}, cache, dev)
			}
			if !slices.Equal(greedy, qwen35SmallGreedy) {
				t.Errorf("greedy %v, want %v", greedy, qwen35SmallGreedy)
			}
		})
	}
}

// BF16 weights and activations against transformers BF16 eager on CPU
// (`ref_qwen35.py Qwen/Qwen3.5-0.8B bf16 --chat`). BF16 logits near 19
// are spaced 0.125 apart, so the tolerance is a few steps; the greedy
// continuation is compared token by token and must agree on its prefix.
// On CPU this takes a few minutes (BF16 matmuls are not vectorised).
var (
	qwen35SmallRefBF16 = []struct {
		id    int
		logit float32
	}{{760, 19.0}, {28232, 18.25}, {13512, 18.0}, {29108, 17.5}, {42373, 17.25}}
	qwen35SmallGreedyBF16 = []int64{760, 12515, 7701, 6105, 1521, 12870, 60845, 69377, 10814, 22602, 89661, 314, 6105, 3000, 310, 43000}
)

func TestQwen35SmallMatchesTransformersBF16(t *testing.T) {
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/config.json")); len(m) == 0 {
		t.Skip(qwen35SmallRepo + " not in HF cache")
	}
	devs := []device.Device{device.CPU()}
	if device.CUDAAvailable() {
		if gpu, err := device.CUDA(0); err == nil {
			devs = append(devs, gpu)
		}
	}
	for _, dev := range devs {
		t.Run(dev.String(), func(t *testing.T) {
			m, err := LoadQwen35FromHF(qwen35SmallRepo, dev, dtype.BF16)
			if err != nil {
				t.Fatal(err)
			}
			cache := NewKVCache(m.NumLayers())
			row := lastLogits(t, m, qwen35SmallChatIDs, cache, dev)
			got := top5(row)
			for i, w := range qwen35SmallRefBF16 {
				if i < 2 && got[i] != w.id {
					t.Errorf("top-%d: got %d (%.3f), want %d (%.3f)", i+1, got[i], row[got[i]], w.id, w.logit)
				}
				if d := row[w.id] - w.logit; d > 0.4 || d < -0.4 {
					t.Errorf("logit[%d] = %.3f, want %.3f (BF16 tol 0.4)", w.id, row[w.id], w.logit)
				}
			}
			t.Logf("top-5 %v logits %.3f %.3f %.3f %.3f %.3f", got, row[got[0]], row[got[1]], row[got[2]], row[got[3]], row[got[4]])
			var greedy []int64
			for len(greedy) < len(qwen35SmallGreedyBF16) {
				next := int64(top5(row)[0])
				greedy = append(greedy, next)
				row = lastLogits(t, m, []int64{next}, cache, dev)
			}
			agree := 0
			for agree < len(greedy) && greedy[agree] == qwen35SmallGreedyBF16[agree] {
				agree++
			}
			t.Logf("greedy %v (first %d of %d agree with transformers BF16)", greedy, agree, len(greedy))
			if agree < 5 {
				t.Errorf("greedy %v diverges from transformers BF16 %v after %d tokens", greedy, qwen35SmallGreedyBF16, agree)
			}
		})
	}
}
