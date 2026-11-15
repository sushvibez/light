package diffusion

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/safetensors"
	"light/splat"
	"light/tensor"
)

// The tiny pipeline end to end through stage 2 (scripts/verify/ref_trellis_slat.py
// ran TrellisImageTo3DPipeline's run() on the same models): stage 1 from
// the reference pixels and noise, sample_slat on its coords from the
// recorded noise, decode_slat's Gaussians and save_ply's rows, a written
// PLY read back, and Generate's plumbing on an RGBA image.
func TestTrellisPipelineTinyGaussians(t *testing.T) {
	raw, err := os.ReadFile("../models/testdata/trellis-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref1 struct {
		DINOv2 struct {
			Pixels []float32
			Size   int
		} `json:"dinov2"`
		Sample struct{ Noise []float32 } `json:"sample"`
	}
	if err := json.Unmarshal(raw, &ref1); err != nil {
		t.Fatal(err)
	}
	ref, _, err := safetensors.LoadFiles([]string{"../models/testdata/trellis-tiny-slat.safetensors"}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	f32 := func(name string) []float32 { return tensor.MustSlice[float32](ref[name]) }
	cmp := func(t *testing.T, name string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
		}
		md, mr := 0.0, 0.0
		for i := range got {
			md = math.Max(md, math.Abs(float64(got[i]-want[i])))
			mr = math.Max(mr, math.Abs(float64(want[i])))
		}
		t.Logf("%s: max diff %.3g (max |ref| %.3g)", name, md, mr)
		if md > 1e-4*math.Max(1, mr) {
			t.Errorf("%s differs by %.3g", name, md)
		}
	}
	host := func(t *testing.T, x *tensor.Tensor) []float32 {
		t.Helper()
		h, err := x.To(device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		return tensor.MustSlice[float32](h)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		t.Run(fmt.Sprint(dev), func(t *testing.T) {
			p, err := LoadTrellisPipeline(trellisTiny, dev, TrellisLoadOptions{DINOv2: trellisTiny + "/dinov2"})
			if err != nil {
				t.Fatal(err)
			}
			if s := p.SLatSampler; s.Steps != 25 || s.CFG != 5 || s.Interval != [2]float64{0.5, 1} || s.RescaleT != 3 {
				t.Fatalf("slat sampler params %+v", s)
			}
			cond, err := p.Condition(ref1.DINOv2.Pixels, ref1.DINOv2.Size)
			if err != nil {
				t.Fatal(err)
			}
			_, _, coords, err := p.SampleSparseStructure(cond, ref1.Sample.Noise, p.Sampler)
			if err != nil {
				t.Fatal(err)
			}
			rc := tensor.MustSlice[int32](ref["sample.coords"])
			if len(coords)*4 != len(rc) {
				t.Fatalf("%d coords, reference %d", len(coords), len(rc)/4)
			}
			for i, c := range coords {
				if c != [4]int32{rc[4*i], rc[4*i+1], rc[4*i+2], rc[4*i+3]} {
					t.Fatalf("coord %d: %v, reference %v", i, c, rc[4*i:4*i+4])
				}
			}
			g, rawz, slat, err := p.SampleSLat(cond, coords, f32("sample.noise"), p.SLatSampler)
			if err != nil {
				t.Fatal(err)
			}
			cmp(t, "slat sample", host(t, rawz), f32("sample.raw"))
			cmp(t, "slat", host(t, slat), f32("sample.slat"))
			feats, gs, err := p.DecodeGaussians(g, slat)
			if err != nil {
				t.Fatal(err)
			}
			cmp(t, "decoder outputs", host(t, feats), f32("dec.feats"))
			cmp(t, "save_ply rows", splat.PLYRows(gs), f32("gs.ply"))

			path := filepath.Join(t.TempDir(), "tiny.ply")
			if err := splat.WritePLY(path, gs); err != nil {
				t.Fatal(err)
			}
			back, err := splat.ReadPLY(path)
			if err != nil {
				t.Fatal(err)
			}
			cmp(t, "PLY round trip xyz", back.XYZ, gs.XYZ)
			cmp(t, "PLY round trip opacity", back.Opacity, gs.Opacity)
			cmp(t, "PLY round trip scaling", back.Scaling, gs.Scaling)
			worst := 0.0
			for i := range back.Opacity {
				d := 0.0
				for k := range 4 {
					d += float64(back.Rotation[4*i+k]) * float64(gs.Rotation[4*i+k])
				}
				worst = math.Max(worst, 1-math.Abs(d))
			}
			t.Logf("PLY round trip rotation: max 1 - |<q, q'>| %.3g", worst)
			if worst > 1e-6 {
				t.Errorf("rotations differ after a PLY round trip: %.3g", worst)
			}

			res, err := p.Generate(TrellisOptions{Image: trellisReadPNG(t, trellisTiny+"/object.png"), Seed: 3, Steps: 4, SLatSteps: 4})
			if err != nil {
				t.Fatal(err)
			}
			if n := len(res.Gaussians.Opacity); n == 0 || n != len(res.Coords)*p.SLatDecoder.Args.Rep.NumGaussians {
				t.Errorf("Generate: %d Gaussians for %d voxels", n, len(res.Coords))
			}
			t.Logf("Generate: %d voxels, %d Gaussians", len(res.Coords), len(res.Gaussians.Opacity))
		})
	}
}
