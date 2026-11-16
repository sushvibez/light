package diffusion

import (
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"testing"

	"light/device"
	"light/tensor"
)

const trellisTiny = "../models/testdata/trellis-tiny"

func trellisReadPNG(t *testing.T, path string) image.Image {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

// TRELLIS's preprocess_image on an RGBA object against Pillow
// (scripts/verify/ref_trellis_ss.py): the crop matches, Lanczos rounding
// differs slightly.
func TestTrellisPreprocessMatchesPIL(t *testing.T) {
	want := trellisReadPNG(t, trellisTiny+"/preprocessed.png")
	S := want.Bounds().Dx()
	px, _, err := trellisPreprocess(trellisReadPNG(t, trellisTiny+"/object.png"), S, nil)
	if err != nil {
		t.Fatal(err)
	}
	var sum, mx float64
	far := 0
	for y := range S {
		for x := range S {
			r, g, b, _ := want.At(x, y).RGBA()
			for c, v := range []uint32{r, g, b} {
				d := math.Abs(float64(px[c*S*S+y*S+x]) - float64(v>>8)/255)
				sum += d
				mx = math.Max(mx, d)
				if d > 2.5/255 {
					far++
				}
			}
		}
	}
	mean := sum / float64(3*S*S)
	t.Logf("preprocess (%d x %d): mean |diff| %.5f, max %.4f (of 1); %d of %d values off by more than 2/255", S, S, mean, mx, far, 3*S*S)
	if mean > 0.002 || mx > 0.15 || far > 3*S*S/100 {
		t.Errorf("preprocessed image differs: mean %.5f, max %.4f", mean, mx)
	}
}

// The tiny stage-1 pipeline loaded from its folder (pipeline.json, ckpts/,
// a DINOv2 folder): encode_image's tokens, then sample_sparse_structure's
// latent and coords from the recorded noise with pipeline.json's sampler.
func TestTrellisPipelineTiny(t *testing.T) {
	raw, err := os.ReadFile("../models/testdata/trellis-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		DINOv2 struct {
			Pixels, Tokens []float32
			Size           int
		} `json:"dinov2"`
		Sample struct {
			Noise, Latent, Logits []float32
			Coords                []int32
		} `json:"sample"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	cmp := func(t *testing.T, name string, x *tensor.Tensor, want []float32) {
		t.Helper()
		h, err := x.To(device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		got := tensor.MustSlice[float32](h)
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
	for _, dev := range devs {
		t.Run(fmt.Sprint(dev), func(t *testing.T) {
			p, err := LoadTrellisPipeline(trellisTiny, dev, TrellisLoadOptions{DINOv2: trellisTiny + "/dinov2"})
			if err != nil {
				t.Fatal(err)
			}
			if s := p.Sampler; s.Steps != 25 || s.CFG != 5 || s.Interval != [2]float64{0.5, 1} || s.RescaleT != 3 {
				t.Fatalf("sampler params %+v", s)
			}
			cond, err := p.Condition(ref.DINOv2.Pixels, ref.DINOv2.Size)
			if err != nil {
				t.Fatal(err)
			}
			cmp(t, "cond", cond, ref.DINOv2.Tokens)
			lat, logits, coords, err := p.SampleSparseStructure(cond, ref.Sample.Noise, p.Sampler)
			if err != nil {
				t.Fatal(err)
			}
			cmp(t, "latent", lat, ref.Sample.Latent)
			cmp(t, "logits", logits, ref.Sample.Logits)
			R := logits.Dim(2)
			want := map[[4]int32]bool{}
			for i := 0; i+3 < len(ref.Sample.Coords); i += 4 {
				c := ref.Sample.Coords
				want[[4]int32{c[i], c[i+1], c[i+2], c[i+3]}] = true
			}
			bad := 0
			for _, c := range coords {
				if !want[c] {
					bad++
				}
			}
			bad += len(want) - (len(coords) - bad)
			t.Logf("coords: %d (reference %d) of %d³, %d differ", len(coords), len(want), R, bad)
			if bad > 0 {
				t.Errorf("%d coords differ", bad)
			}
		})
	}
}
