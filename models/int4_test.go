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
	ref  []struct {
		id    int
		logit float32
	}
	greedy []int64
}{
	{"Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
		[]struct {
			id    int
			logit float32
		}{{785, 22.6532}, {4498, 16.2008}, {2121, 15.4965}, {641, 15.4459}, {1249, 15.4160}},
		// "The sky is blue because it reflects sunlight, which scatters colors of the Earth"
		[]int64{785, 12884, 374, 6303, 1576, 432, 25963, 39020, 11, 892, 1136, 10175, 7987, 315, 279, 9237}},
	{"Qwen/Qwen2.5-0.5B-Instruct-AWQ",
		[]struct {
			id    int
			logit float32
		}{{785, 21.8347}, {4498, 15.3062}, {10331, 14.7970}, {69769, 14.7519}, {2121, 14.4423}},
		// "The sky appears blue because it reflects sunlight, which is a mixture of all colors"
		[]int64{785, 12884, 7952, 6303, 1576, 432, 25963, 39020, 11, 892, 374, 264, 20980, 315, 678, 7987}},
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
					row := lastLogits(t, m, int4ChatIDs, cache, dev)
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
