package diffusion

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
	"light/models"
	"light/ops"
	"light/safetensors"
	"light/tensor"
)

// TRELLIS-image-large stage 1 on its real weights against TRELLIS's own
// run (scripts/colab/trellis_ss_check.py records it in <work_dir>): the
// preprocessed image, the DINOv2 tokens, the sampled latent from the same
// noise and tokens, the decoder's logits on the reference latent, and the
// coords of Light's own end-to-end run. Runs when LIGHT_TRELLIS_DIR names
// that folder; on CUDA when built with it (LIGHT_TRELLIS_CPU=1 forces the
// CPU, which takes a while).
func TestTrellisRealMatchesReference(t *testing.T) {
	dir := os.Getenv("LIGHT_TRELLIS_DIR")
	if dir == "" {
		t.Skip("LIGHT_TRELLIS_DIR not set")
	}
	var meta struct {
		Seed     int
		FP16     bool
		Occupied int
	}
	raw, err := os.ReadFile(filepath.Join(dir, "ref.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	ref, _, err := safetensors.LoadFiles([]string{filepath.Join(dir, "ref.safetensors")}, device.CPU())
	if err != nil {
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
		return tensor.MustSlice[float32](x)
	}
	dev := device.CPU()
	if os.Getenv("LIGHT_TRELLIS_CPU") != "1" {
		if g, err := device.CUDA(0); err == nil {
			dev = g
		}
	}
	t.Logf("device %v; reference seed %d, fp16 %v, %d occupied", dev, meta.Seed, meta.FP16, meta.Occupied)
	o := TrellisLoadOptions{SparseStructureOnly: true}
	if d := filepath.Join(dir, "dinov2-with-registers-large"); trellisDirExists(d) {
		o.DINOv2 = d
	}
	p, err := LoadTrellisPipeline(filepath.Join(dir, "TRELLIS-image-large"), dev, o)
	if err != nil {
		t.Fatal(err)
	}
	S := p.ImageSize()
	// stat compares got with want: max diff relative to max |want|.
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
		t.Logf("%s: max diff %.3g (max |ref| %.3g, rel %.3g), rms rel %.3g", name, md, mr, md/math.Max(mr, 1e-30), math.Sqrt(se/math.Max(sw, 1e-30)))
		if md > tol*math.Max(1, mr) {
			t.Errorf("%s differs by %.3g", name, md)
		}
	}
	iou := func(a [][4]int32, b []int32) (float64, int) {
		set := map[[4]int32]bool{}
		for i := 0; i+3 < len(b); i += 4 {
			set[[4]int32{b[i], b[i+1], b[i+2], b[i+3]}] = true
		}
		inter := 0
		for _, c := range a {
			if set[c] {
				inter++
			}
		}
		union := len(a) + len(set) - inter
		return float64(inter) / math.Max(1, float64(union)), len(set)
	}
	if ref["coords"].DType() != dtype.I32 {
		t.Fatalf("coords stored as %v, want I32", ref["coords"].DType())
	}
	refCoords := tensor.MustSlice[int32](ref["coords"])

	// Preprocessing (Lanczos rounding differs slightly from Pillow's).
	img := trellisReadPNG(t, filepath.Join(dir, "object.png"))
	px, _, err := trellisPreprocess(img, S)
	if err != nil {
		t.Fatal(err)
	}
	refPx := host(ref["pixels"])
	var sum float64
	for i := range px {
		sum += math.Abs(float64(px[i] - refPx[i]))
	}
	t.Logf("preprocess: mean |diff| %.5f (of 1)", sum/float64(len(px)))
	if sum/float64(len(px)) > 0.003 {
		t.Errorf("preprocessed image differs: mean %.5f", sum/float64(len(px)))
	}

	// DINOv2 tokens of the reference pixels.
	cond, err := p.Condition(refPx, S)
	if err != nil {
		t.Fatal(err)
	}
	stat("cond (DINOv2 tokens)", host(cond), host(ref["cond"]), 2e-3)

	// The sample from the reference noise and tokens.
	refCond, err := ref["cond"].To(dev)
	if err != nil {
		t.Fatal(err)
	}
	so := p.Sampler
	so.Step = func(i, n int) { t.Logf("step %d / %d", i+1, n) }
	lat, logits, coords, err := p.SampleSparseStructure(refCond, host(ref["noise"]), so)
	if err != nil {
		t.Fatal(err)
	}
	latTol := 1e-2
	if meta.FP16 {
		latTol = 0.1
	}
	stat("latent", host(lat), host(ref["latent"]), latTol)
	stat("logits (own latent)", host(logits), host(ref["logits"]), 10*latTol)
	j, n := iou(coords, refCoords)
	t.Logf("coords (reference tokens and noise): %d, reference %d, IoU %.4f", len(coords), n, j)
	if j < 0.98 {
		t.Errorf("coords IoU %.4f", j)
	}

	// The decoder on the reference latent.
	rl, err := ref["latent"].To(dev)
	if err != nil {
		t.Fatal(err)
	}
	dl, err := p.Decoder.Forward(rl)
	if err != nil {
		t.Fatal(err)
	}
	stat("decoder logits (reference latent)", host(dl), host(ref["logits"]), 1e-3)
	dc, err := models.TrellisCoords(dl)
	if err != nil {
		t.Fatal(err)
	}
	j, _ = iou(dc, refCoords)
	t.Logf("coords (reference latent): %d, IoU %.4f", len(dc), j)
	if j < 0.999 {
		t.Errorf("decoder coords IoU %.4f", j)
	}

	// Light end to end: its own preprocessing and tokens, the reference noise.
	res, err := p.SparseStructure(TrellisOptions{Image: img, Noise: host(ref["noise"])})
	if err != nil {
		t.Fatal(err)
	}
	j, _ = iou(res.Coords, refCoords)
	t.Logf("coords (end to end): %d, IoU %.4f", len(res.Coords), j)
	if j < 0.95 {
		t.Errorf("end-to-end coords IoU %.4f", j)
	}
}

func trellisDirExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}
