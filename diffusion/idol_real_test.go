package diffusion

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
	"light/models"
	"light/nn"
	"light/safetensors"
	"light/tensor"
)

// TestIdolConditionRealMatchesSDK compares Idol's DINOv3 image tokens on
// the real weights with idol-sdk's for the same preprocessed pixels
// (scripts/colab/idol_check.py's ref.safetensors in LIGHT_IDOL_REF; the
// weights folder in LIGHT_IDOL_DIR): Light as light idol runs it (F16
// weights, F32 GEMMs, FP16 flash attention on CUDA) against the SDK's
// default BF16 tokens and its F32 ones. Light's error against F32 must stay
// within a tenth of the SDK BF16 tokens' own.
func TestIdolConditionRealMatchesSDK(t *testing.T) {
	refDir, wDir := os.Getenv("LIGHT_IDOL_REF"), os.Getenv("LIGHT_IDOL_DIR")
	if refDir == "" || wDir == "" {
		t.Skip("LIGHT_IDOL_REF / LIGHT_IDOL_DIR not set")
	}
	ref, _, err := safetensors.LoadFiles([]string{filepath.Join(refDir, "ref.safetensors")}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if ref["feature1_f32"] == nil {
		t.Skip("ref.safetensors has no feature1_f32 (rerun idol_check.py, COND_ONLY=1 suffices)")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Skip("no CUDA device")
	}
	p, err := LoadIdolPipeline(wDir, dev, IdolLoadOptions{FluxVAE: "black-forest-labs/FLUX.2-klein-4B", Storage16: true})
	if err != nil {
		t.Fatal(err)
	}
	px := tensor.MustSlice[float32](ref["pixels"])
	f1, _, err := p.Condition(px, ref["pixels"].Dim(2), nil)
	if err != nil {
		t.Fatal(err)
	}
	h, err := f1.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](h)
	bf16, f32 := tensor.MustSlice[float32](ref["feature1"]), tensor.MustSlice[float32](ref["feature1_f32"])
	if len(got) != len(f32) {
		t.Fatalf("feature1: %d values, want %d", len(got), len(f32))
	}
	rel := func(a, b []float32) (rms, maxd float64) {
		var se, sw float64
		for i := range a {
			d := float64(a[i] - b[i])
			se += d * d
			sw += float64(b[i]) * float64(b[i])
			maxd = math.Max(maxd, math.Abs(d))
		}
		return math.Sqrt(se / sw), maxd
	}
	e32, m32 := rel(got, f32)
	e16, m16 := rel(got, bf16)
	sdk, sdkMax := rel(bf16, f32)
	t.Logf("DINOv3 tokens: rms rel err vs sdk f32 %.3g (max %.3g), vs sdk bf16 %.3g (max %.3g); sdk bf16 vs f32 %.3g (max %.3g)",
		e32, m32, e16, m16, sdk, sdkMax)
	// For comparison: F32 SDPA (and BF16 compute throughout, as the SDK).
	var exact *models.DINOv3
	ddir, err := fetchFiles(p.Model.Cfg.ImageEncoders["dinov3"], "config.json", "model.safetensors")
	if err == nil {
		err = nn.WithWeightStorage16(dtype.F16, func() (err error) { exact, err = models.LoadDINOv3(ddir, dev); return })
	}
	var b16 *models.DINOv3
	if err == nil {
		err = nn.WithWeightStorage16Mixed(dtype.BF16, func() (err error) { b16, err = models.LoadDINOv3(ddir, dev); return })
	}
	for _, c := range []struct {
		name string
		m    *models.DINOv3
	}{{"F32 SDPA", exact}, {"BF16 GEMMs", b16}} {
		if c.m == nil {
			continue
		}
		q := &IdolPipeline{DINO: c.m, VAE: p.VAE, Dev: dev}
		if f, _, err := q.Condition(px, ref["pixels"].Dim(2), nil); err == nil {
			if h, err := f.To(device.CPU()); err == nil {
				e, m := rel(tensor.MustSlice[float32](h), f32)
				t.Logf("(%s: vs sdk f32 %.3g, max %.3g)", c.name, e, m)
			}
		}
	}
	if e32 > 0.1*sdk {
		t.Errorf("DINOv3 tokens: error vs F32 %.3g exceeds a tenth of the SDK's own BF16 error %.3g", e32, sdk)
	}
}
