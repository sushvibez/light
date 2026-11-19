package diffusion

import (
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"light/device"
	"light/dtype"
	"light/models"
	"light/ops"
	"light/safetensors"
	"light/splat"
	"light/tensor"
)

// TRELLIS-image-large stage 2 on its real weights against TRELLIS's own run
// with its real spconv / xformers backends (scripts/colab/trellis_check.py
// records it in <work_dir>): one SLat flow forward on the reference noise,
// the SLat sample from the same noise, tokens and coords, the Gaussian
// decoder on the reference SLat (its outputs, the activated Gaussians and
// save_ply's rows), and Light's own end-to-end run from the reference
// noises (written to <work_dir>/light.ply and light_views.png). Runs when
// LIGHT_TRELLIS_DIR names that folder and its ref.safetensors has the stage-2
// keys; on CUDA when built with it (LIGHT_TRELLIS_CPU=1 forces the CPU).
func TestTrellisGaussiansRealMatchesReference(t *testing.T) {
	dir := os.Getenv("LIGHT_TRELLIS_DIR")
	if dir == "" {
		t.Skip("LIGHT_TRELLIS_DIR not set")
	}
	ref, _, err := safetensors.LoadFiles([]string{filepath.Join(dir, "ref.safetensors")}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := ref["slat"]; !ok {
		t.Skip("ref.safetensors has no stage-2 keys (written by trellis_ss_check.py, not trellis_check.py)")
	}
	var meta struct {
		Seed      int
		FP16      bool
		Occupied  int
		Gaussians int
		Backends  map[string]string
		Check     map[string]float64 `json:"backend_check"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "ref.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
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
	dev := device.CPU()
	if os.Getenv("LIGHT_TRELLIS_CPU") != "1" {
		if g, err := device.CUDA(0); err == nil {
			dev = g
		}
	}
	t.Logf("device %v; reference seed %d, fp16 %v, %d voxels, %d Gaussians, backends %v, stand-in check %v",
		dev, meta.Seed, meta.FP16, meta.Occupied, meta.Gaussians, meta.Backends, meta.Check)
	o := TrellisLoadOptions{}
	if d := filepath.Join(dir, "dinov2-with-registers-large"); trellisDirExists(d) {
		o.DINOv2 = d
	}
	p, err := LoadTrellisPipeline(filepath.Join(dir, "TRELLIS-image-large"), dev, o)
	if err != nil {
		t.Fatal(err)
	}
	// stat compares got with want: max diff relative to max |want|, and the
	// relative RMS error.
	stat := func(name string, got, want []float32, tol float64) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
		}
		md, mr, se, sw := 0.0, 0.0, 0.0, 0.0
		for i := range got {
			d := float64(got[i] - want[i])
			md = math.Max(md, math.Abs(d))
			mr = math.Max(mr, math.Abs(float64(want[i])))
			se += d * d
			sw += float64(want[i]) * float64(want[i])
		}
		rms := math.Sqrt(se / math.Max(sw, 1e-30))
		t.Logf("%s: max diff %.3g (max |ref| %.3g, rel %.3g), rms rel %.3g", name, md, mr, md/math.Max(mr, 1e-30), rms)
		if md > tol*math.Max(1, mr) {
			t.Errorf("%s differs by %.3g", name, md)
		}
	}
	up := func(name string) *tensor.Tensor {
		x, err := ref[name].To(dev)
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	cv := tensor.MustSlice[int32](ref["coords"])
	coords := make([][4]int32, len(cv)/4)
	for i := range coords {
		copy(coords[i][:], cv[4*i:4*i+4])
	}
	g, err := models.NewTrellisGrid(coords, dev)
	if err != nil {
		t.Fatal(err)
	}
	cond := up("cond")
	tol := 1.0 // relative to F32: TRELLIS_FP16 references drift ~10x further
	if meta.FP16 {
		tol = 10
	}

	// One flow forward at t = 1000 on the reference noise.
	v0, err := p.SLatFlow.Forward(g, up("slat_noise"), []float32{1000}, cond)
	if err != nil {
		t.Fatal(err)
	}
	stat("slat flow forward (t = 1000)", host(v0), host(ref["slat_v0"]), 2e-3*tol)

	// The sample from the reference noise, tokens and coords.
	so := p.SLatSampler
	so.Step = func(i, n int) { t.Logf("slat step %d / %d", i+1, n) }
	_, rawz, slat, err := p.SampleSLat(cond, coords, host(ref["slat_noise"]), so)
	if err != nil {
		t.Fatal(err)
	}
	stat("slat sample", host(rawz), host(ref["slat_raw"]), 2e-2*tol)
	stat("slat (un-normalised)", host(slat), host(ref["slat"]), 2e-2*tol)

	// The Gaussian decoder on the reference SLat.
	feats, gs, err := p.DecodeGaussians(g, up("slat"))
	if err != nil {
		t.Fatal(err)
	}
	stat("decoder outputs", host(feats), host(ref["gs_feats"]), 2e-3*tol)
	stat("gaussian xyz", gs.XYZ, host(ref["gs_xyz"]), 2e-3*tol)
	stat("gaussian features_dc", gs.FeaturesDC, host(ref["gs_features_dc"]), 2e-3*tol)
	stat("gaussian scaling", gs.Scaling, host(ref["gs_scaling"]), 2e-3*tol)
	stat("gaussian rotation", gs.Rotation, host(ref["gs_rotation"]), 2e-3*tol)
	stat("gaussian opacity", gs.Opacity, host(ref["gs_opacity"]), 2e-3*tol)
	// save_ply's rows; near-zero quaternion components carry F32 rounding
	// noise amplified by matrix_to_quaternion's square roots (~1e-3 abs).
	rows, want := splat.PLYRows(gs), host(ref["gs_ply"])
	for c, name := range splat.PLYProperties {
		if name == "nx" || name == "ny" || name == "nz" {
			continue
		}
		var a, b []float32
		for i := c; i < len(rows); i += len(splat.PLYProperties) {
			a, b = append(a, rows[i]), append(b, want[i])
		}
		lim := 2e-3 * tol
		if strings.HasPrefix(name, "rot") {
			lim = 5e-3 * tol
		}
		if name == "opacity" {
			// The file stores logit(opacity): near opacity 1 a 1e-5
			// difference becomes a large one in logit space, so compare
			// the opacities themselves (the column's meaning).
			for i := range a {
				a[i] = float32(1 / (1 + math.Exp(-float64(a[i]))))
				b[i] = float32(1 / (1 + math.Exp(-float64(b[i]))))
			}
			name = "opacity (sigmoid)"
		}
		stat("save_ply "+name, a, b, lim)
	}

	// Light end to end (its own preprocessing and tokens) from the reference
	// noises; the SLat noise only fits when the coords come out the same.
	img := trellisReadPNG(t, filepath.Join(dir, "object.png"))
	opts := TrellisOptions{Image: img, Noise: host(ref["noise"])}
	res, err := p.SparseStructure(opts)
	if err != nil {
		t.Fatal(err)
	}
	same := len(res.Coords) == len(coords)
	for i := 0; same && i < len(coords); i++ {
		same = res.Coords[i] == coords[i]
	}
	if same {
		opts.SLatNoise = host(ref["slat_noise"])
	} else {
		t.Logf("end to end: %d coords (reference %d) differ; drawing Light's own SLat noise", len(res.Coords), len(coords))
	}
	full, err := p.Generate(opts)
	if err != nil {
		t.Fatal(err)
	}
	if same {
		stat("end-to-end slat", full.SLat, host(ref["slat"]), 5e-2*tol)
	}
	out := filepath.Join(dir, "light.ply")
	if err := splat.WritePLY(out, full.Gaussians); err != nil {
		t.Fatal(err)
	}
	im := splat.Turntable(full.Gaussians, splat.Camera{Pitch: 0.25, Radius: 2, FOV: 40}, 4, 384, 384, color.NRGBA{255, 255, 255, 255})
	if err := savePNGTest(filepath.Join(dir, "light_views.png"), im); err != nil {
		t.Fatal(err)
	}
	t.Logf("end to end: %d voxels, %d Gaussians; wrote %s and light_views.png", len(full.Coords), len(full.Gaussians.Opacity), out)
}

func savePNGTest(path string, im image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(f, im); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
