package quant_test

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/quant"
	"light/safetensors"
	"light/tensor"
)

func hfFile(t *testing.T, repo, file string) string {
	t.Helper()
	home, _ := os.UserHomeDir()
	m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--"+filepath.FromSlash(
		replaceSlash(repo))+"/snapshots/*/"+file))
	if len(m) == 0 {
		t.Skip(repo + " not in HF cache")
	}
	return m[0]
}

func replaceSlash(repo string) string {
	out := []byte(repo)
	for i, c := range out {
		if c == '/' {
			return string(out[:i]) + "--" + string(out[i+1:])
		}
	}
	return repo
}

func load(t *testing.T, path string, names ...string) map[string]*tensor.Tensor {
	t.Helper()
	all, _, err := safetensors.LoadFiles([]string{path}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]*tensor.Tensor{}
	for _, n := range names {
		x, ok := all[n]
		if !ok {
			t.Fatalf("missing %s", n)
		}
		out[n] = x
	}
	return out
}

func f32(t *testing.T, x *tensor.Tensor) *tensor.Tensor {
	t.Helper()
	c, err := ops.Cast(x, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// relErr is ‖W − ref‖ / ‖ref‖ for row-major (N, K) slices.
func relErr(w, ref []float32) float64 {
	var num, den float64
	for i := range w {
		d := float64(w[i]) - float64(ref[i])
		num += d * d
		den += float64(ref[i]) * float64(ref[i])
	}
	return math.Sqrt(num / den)
}

// The real Qwen2.5-0.5B-Instruct GPTQ-Int4 and AWQ layer-3 q_proj,
// converted and dequantized, against the BF16 original: the relative errors
// must equal the ones an independent PyTorch unpacking of the same tensors
// gives (0.1615 / 0.3087; a wrong nibble order or zero offset gives
// 1.49 / 0.46), which pins both layouts to the published checkpoints.
func TestInt4FromRealCheckpoints(t *testing.T) {
	const L = "model.layers.3.self_attn.q_proj."
	orig := load(t, hfFile(t, "Qwen/Qwen2.5-0.5B-Instruct", "model.safetensors"), L+"weight")[L+"weight"]
	ref := tensor.MustSlice[float32](f32(t, orig))

	g := load(t, hfFile(t, "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", "model.safetensors"),
		L+"qweight", L+"qzeros", L+"scales", L+"g_idx")
	wg, err := quant.FromGPTQ(g[L+"qweight"], g[L+"qzeros"], f32(t, g[L+"scales"]), g[L+"g_idx"], 128, false)
	if err != nil {
		t.Fatal(err)
	}
	if wg.N() != orig.Dim(0) || wg.K() != orig.Dim(1) {
		t.Fatalf("GPTQ shape (%d, %d), original %v", wg.N(), wg.K(), orig.Shape())
	}
	dg, err := wg.Dequant()
	if err != nil {
		t.Fatal(err)
	}
	if e := relErr(dg, ref); math.Abs(e-0.1615) > 5e-4 {
		t.Errorf("GPTQ relative error %.4f, want 0.1615", e)
	}

	a := load(t, hfFile(t, "Qwen/Qwen2.5-0.5B-Instruct-AWQ", "model.safetensors"), L+"qweight", L+"qzeros", L+"scales")
	wa, err := quant.FromAWQ(a[L+"qweight"], a[L+"qzeros"], f32(t, a[L+"scales"]), 128)
	if err != nil {
		t.Fatal(err)
	}
	da, err := wa.Dequant()
	if err != nil {
		t.Fatal(err)
	}
	if e := relErr(da, ref); math.Abs(e-0.3087) > 5e-4 {
		t.Errorf("AWQ relative error %.4f, want 0.3087", e)
	}
}
