package models

import (
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/safetensors"
	"light/tensor"
)

// TestIdolRealMatchesSDK compares Idol's denoiser and Gaussian decoder on
// the real speridlabs/idol weights with idol-sdk (scripts/colab/idol_check.py
// writes <LIGHT_IDOL_REF>/ref.safetensors; LIGHT_IDOL_DIR is the weights
// folder): one forward, the 50-step CFG sample and the Gaussian decoder, as
// light idol runs them (16-bit weights, FP16 GEMMs and FP16 attention on
// CUDA). Each output is scored by its RMS error relative to the reference's
// RMS against the SDK in F32 (the exact result) and in FP16 (as shipped),
// next to the SDK FP16 run's own error against F32: Light must stay within
// 1.5x upstream's own FP16 error (2x for the 50-step sample, where FP16
// rounding moves a few tokens by up to ~3 in either implementation).
func TestIdolRealMatchesSDK(t *testing.T) {
	refDir, wDir := os.Getenv("LIGHT_IDOL_REF"), os.Getenv("LIGHT_IDOL_DIR")
	if refDir == "" || wDir == "" {
		t.Skip("LIGHT_IDOL_REF / LIGHT_IDOL_DIR not set")
	}
	ref, _, err := safetensors.LoadFiles([]string{filepath.Join(refDir, "ref.safetensors")}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Skip("no CUDA device")
	}
	var m *Idol
	if err := nn.WithWeightStorage16Mixed(dtype.F16, func() (err error) { m, err = LoadIdol(wDir, dev); return }); err != nil {
		t.Fatal(err)
	}
	host := func(x *tensor.Tensor) []float32 {
		t.Helper()
		x, err := x.To(device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		if x.DType() != dtype.F32 {
			if x, err = ops.Cast(x, dtype.F32); err != nil {
				t.Fatal(err)
			}
		}
		if x, err = x.Contiguous(); err != nil {
			t.Fatal(err)
		}
		return tensor.MustSlice[float32](x)
	}
	up := func(name string) *tensor.Tensor {
		t.Helper()
		x, err := ref[name].To(dev)
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	rel := func(got, want []float32) (rms, maxd float64) {
		var se, sw float64
		for i := range got {
			d := float64(got[i] - want[i])
			se += d * d
			sw += float64(want[i]) * float64(want[i])
			maxd = math.Max(maxd, math.Abs(d))
		}
		return math.Sqrt(se / math.Max(sw, 1e-30)), maxd
	}
	// check scores got against both references; factor bounds Light's error
	// against F32 by that many times the SDK FP16 run's own.
	check := func(name string, got *tensor.Tensor, key string, factor float64) {
		t.Helper()
		g, f32, f16 := host(got), host(ref[key+"_f32"]), host(ref[key+"_f16"])
		if len(g) != len(f32) {
			t.Fatalf("%s: %d values, want %d", name, len(g), len(f32))
		}
		e32, m32 := rel(g, f32)
		e16, m16 := rel(g, f16)
		sdk, _ := rel(f16, f32)
		t.Logf("%s: rms rel err vs sdk f32 %.3g (max %.3g), vs sdk f16 %.3g (max %.3g); sdk f16 vs f32 %.3g",
			name, e32, m32, e16, m16, sdk)
		if e32 > factor*math.Max(sdk, 1e-4) {
			t.Errorf("%s: error vs F32 %.3g exceeds %g x the SDK's own FP16 error %.3g", name, e32, factor, sdk)
		}
	}
	f1, f2 := up("feature1"), up("feature2")
	lat0, cam0 := up("latent0"), up("camera0")

	vl, vc, err := m.Denoiser.Forward(IdolInputs{Latent: lat0, Camera: cam0, T: []float32{700}, Feature1: f1, Feature2: f2})
	if err != nil {
		t.Fatal(err)
	}
	check("forward latent (t = 700)", vl, "fwd_latent", 1.5)
	check("forward camera (t = 700)", vc, "fwd_camera", 1.5)

	t0 := time.Now()
	sl, sc, err := IdolSample(m.Denoiser, lat0, cam0, f1, f2, IdolSampleOptions{Steps: 50, CFG: 3, Shift: 3})
	if err != nil {
		t.Fatal(err)
	}
	host(sc)
	t.Logf("sample: 50 steps in %.1f s", time.Since(t0).Seconds())
	check("sample latent (50 steps)", sl, "sample_latent", 2)
	check("sample camera (50 steps)", sc, "sample_camera", 2)

	gs, err := m.Gaussian.Forward(up("anchors"), up("sample_latent_f16"))
	if err != nil {
		t.Fatal(err)
	}
	check("gaussian decoder", gs, "gs", 1.5)
}
