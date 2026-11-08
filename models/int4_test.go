package models

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"light/device"
	"light/dtype"
)

// The official 4-bit Qwen2.5-0.5B-Instruct checkpoints (GPTQ-Int4: symmetric,
// group 128, ordered groups; AWQ: GEMM layout with zero points, group 128),
// loaded packed (quant.Int4) and run on the validation-bar chat prompt.
// Reference: `scripts/verify/ref_int4.py <repo> {gptq|awq} --chat` —
// transformers F32 on weights dequantized by an independent PyTorch
// unpacking of the same files (transformers itself needs GPU kernel
// packages for these formats). Same dequantized values on both sides, so
// F32 agrees to float rounding.
var int4Cases = []struct {
	repo string
	ids  []int64 // nil = int4ChatIDs (Qwen2.5)
	ref  []struct {
		id    int
		logit float32
	}
	greedy []int64
}{
	{"Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", nil,
		[]struct {
			id    int
			logit float32
		}{{785, 22.6532}, {4498, 16.2008}, {2121, 15.4965}, {641, 15.4459}, {1249, 15.4160}},
		// "The sky is blue because it reflects sunlight, which scatters colors of the Earth"
		[]int64{785, 12884, 374, 6303, 1576, 432, 25963, 39020, 11, 892, 1136, 10175, 7987, 315, 279, 9237}},
	{"Qwen/Qwen2.5-0.5B-Instruct-AWQ", nil,
		[]struct {
			id    int
			logit float32
		}{{785, 21.8347}, {4498, 15.3062}, {10331, 14.7970}, {69769, 14.7519}, {2121, 14.4423}},
		// "The sky appears blue because it reflects sunlight, which is a mixture of all colors"
		[]int64{785, 12884, 7952, 6303, 1576, 432, 25963, 39020, 11, 892, 374, 264, 20980, 315, 678, 7987}},
	// Act-order GPTQ (desc_act true, TheBloke's AutoGPTQ export with
	// all-zero biases on every linear): the loader sorts K by group and the
	// matmul gathers x's columns to match.
	{"TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
		[]int64{529, 29989, 5205, 29989, 29958, 13, 3492, 526, 263, 3022, 895, 20255, 29889, 2, 13, 29966, 29989, 1792,
			29989, 29958, 13, 9544, 7420, 297, 697, 10541, 2020, 278, 14744, 338, 7254, 29889, 2, 13, 29966, 29989, 465,
			22137, 29989, 29958, 13},
		[]struct {
			id    int
			logit float32
		}{{1576, 18.1317}, {797, 16.5232}, {29903, 15.2174}, {6716, 13.7784}, {11008, 12.8105}},
		// "The sky is blue because it is a color that is created by the reflection of"
		[]int64{1576, 14744, 338, 7254, 1363, 372, 338, 263, 2927, 393, 338, 2825, 491, 278, 17842, 310}},
}

var int4ChatIDs = []int64{151644, 8948, 198, 2610, 525, 264, 63594, 17847, 13, 151645, 198, 151644, 872, 198, 840,
	20772, 304, 825, 11652, 3170, 279, 12884, 374, 6303, 13, 151645, 198, 151644, 77091, 198}

func TestInt4CheckpointsMatchDequantizedReference(t *testing.T) {
	home, _ := os.UserHomeDir()
	for _, tc := range int4Cases {
		t.Run(tc.repo, func(t *testing.T) {
			cacheDir := "models--" + strings.ReplaceAll(tc.repo, "/", "--")
			if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub", cacheDir, "snapshots/*/model.safetensors")); len(m) == 0 {
				t.Skip(tc.repo + " not in HF cache")
			}
			os.Setenv("HF_HUB_OFFLINE", "1")
			devs := []device.Device{device.CPU()}
			if device.CUDAAvailable() {
				if g, err := device.CUDA(0); err == nil {
					devs = append(devs, g)
				}
			}
			for _, dev := range devs {
				for _, dt := range []dtype.DType{dtype.F32, dtype.BF16} {
					if dev.Kind() == device.KindCPU && dt == dtype.BF16 {
						continue // CPU BF16 matmuls are slow; CUDA covers BF16
					}
					m, err := LoadCausalLMFromHF(tc.repo, dev, dt)
					if err != nil {
						t.Fatal(err)
					}
					if m.Blocks[0].QProj.W4 == nil || m.Blocks[0].DownProj.W4 == nil {
						t.Fatalf("%v %v: projections not loaded as 4-bit", dev, dt)
					}
					cache := NewKVCache(m.NumLayers())
					ids := tc.ids
					if ids == nil {
						ids = int4ChatIDs
					}
					row := lastLogits(t, m, ids, cache, dev)
					got := top5(row)
					tol := float32(1e-3)
					if dt == dtype.BF16 {
						tol = 0.5
					}
					for i, w := range tc.ref {
						if i < 2 && got[i] != w.id {
							t.Errorf("%v %v top-%d: got %d (%.4f), want %d (%.4f)", dev, dt, i+1, got[i], row[got[i]], w.id, w.logit)
						}
						if d := row[w.id] - w.logit; d > tol || d < -tol {
							t.Errorf("%v %v logit[%d] = %.4f, want %.4f (tol %g)", dev, dt, w.id, row[w.id], w.logit, tol)
						}
					}
					var greedy []int64
					for len(greedy) < len(tc.greedy) {
						next := int64(top5(row)[0])
						greedy = append(greedy, next)
						row = lastLogits(t, m, []int64{next}, cache, dev)
					}
					agree := 0
					for agree < len(greedy) && greedy[agree] == tc.greedy[agree] {
						agree++
					}
					t.Logf("%v %v: top-5 %v, greedy %d / %d", dev, dt, got, agree, len(greedy))
					want := len(greedy)
					if dt == dtype.BF16 {
						want = 8
					}
					if agree < want {
						t.Errorf("%v %v: greedy %v diverges from %v after %d tokens", dev, dt, greedy, tc.greedy, agree)
					}
					if dt == dtype.F32 && !slices.Equal(greedy, tc.greedy) {
						t.Errorf("%v F32: greedy %v, want %v", dev, greedy, tc.greedy)
					}
				}
			}
		})
	}
}

// A tiny Qwen3-MoE in GPTQ format with 4-bit attention and per-expert 4-bit
// experts (`scripts/verify/make_int4_tiny.py qwen3moe`), against
// transformers on the independently dequantized weights
// (`ref_int4.py testdata/int4/tiny-qwen3moe-gptq gptq --ids …`): the MoE path
// expands only the routed experts (ops.MoEInt4).
func TestInt4MoETinyMatchesDequantizedReference(t *testing.T) {
	const dir = "testdata/int4/tiny-qwen3moe-gptq"
	ids := []int64{5, 17, 300, 42, 8, 499, 13, 77, 256, 3, 64, 250, 19, 6, 400, 41, 2, 90}
	ref := []struct {
		id    int
		logit float32
	}{{191, 0.4325}, {115, 0.3887}, {505, 0.3585}, {422, 0.3476}, {88, 0.3448}}
	wantGreedy := []int64{191, 497, 411, 25, 268, 145, 152, 321}
	devs := []device.Device{device.CPU()}
	if device.CUDAAvailable() {
		if g, err := device.CUDA(0); err == nil {
			devs = append(devs, g)
		}
	}
	var cpuRow []float32
	for _, dev := range devs {
		m, err := LoadCausalLMFromHF(dir, dev, dtype.F32)
		if err != nil {
			t.Fatal(err)
		}
		b := m.Blocks[0]
		if b.MoE == nil || len(b.MoE.GateUpQ4) != 4 || b.MoE.GateUp != nil || b.QProj.W4 == nil || b.MoE.Gate.W == nil {
			t.Fatalf("%v: 4-bit MoE not installed (experts %d, q_proj 4-bit %v)", dev, len(b.MoE.GateUpQ4), b.QProj.W4 != nil)
		}
		cache := NewKVCache(m.NumLayers())
		row := lastLogits(t, m, ids, cache, dev)
		if dev.Kind() != device.KindCPU {
			if d := maxAbsDiff(cpuRow, row); d > 1e-4 {
				t.Errorf("CUDA vs CPU logits differ by %.2e", d)
			}
			continue
		}
		cpuRow = slices.Clone(row)
		got := top5(row)
		for i, w := range ref {
			if i < 3 && got[i] != w.id {
				t.Errorf("top-%d: got %d (%.4f), want %d (%.4f)", i+1, got[i], row[got[i]], w.id, w.logit)
			}
			if d := row[w.id] - w.logit; d > 2e-4 || d < -2e-4 {
				t.Errorf("logit[%d] = %.5f, want %.4f", w.id, row[w.id], w.logit)
			}
		}
		var greedy []int64
		for len(greedy) < len(wantGreedy) {
			next := int64(top5(row)[0])
			greedy = append(greedy, next)
			row = lastLogits(t, m, []int64{next}, cache, dev)
		}
		if !slices.Equal(greedy, wantGreedy) {
			t.Errorf("greedy %v, want %v", greedy, wantGreedy)
		}
	}
}
