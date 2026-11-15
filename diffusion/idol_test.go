package diffusion

import (
	"encoding/json"
	"fmt"
	"image/png"
	"math"
	"os"
	"testing"

	"light/device"
	"light/models"
	"light/tensor"
)

// Idol's preprocessing and image conditioning on tiny encoders
// (scripts/verify/ref_idol_pipe.py): the cropped, composited image against
// Pillow (Lanczos resampling differs slightly), and the DINOv3 / FLUX.2 VAE
// tokens against idol-sdk's ImageEncoder for the same pixels and VAE noise.
func TestIdolConditioningMatchesSDK(t *testing.T) {
	raw, err := os.ReadFile("testdata/idol-pipe.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Size                            int
		Margin                          float64
		Pixels, Eps, Feature1, Feature2 []float32
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open("testdata/idol-pipe/object.png")
	if err != nil {
		t.Fatal(err)
	}
	img, err := png.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	px, _, err := idolPreprocess(img, ref.Size, ref.Margin)
	if err != nil {
		t.Fatal(err)
	}
	var sum, mx float64
	for i := range px {
		d := math.Abs(float64(px[i] - ref.Pixels[i]))
		sum += d
		mx = math.Max(mx, d)
	}
	t.Logf("preprocess: mean |diff| %.4f, max %.4f (of 1)", sum/float64(len(px)), mx)
	if sum/float64(len(px)) > 0.01 || mx > 0.25 {
		t.Errorf("preprocessed image differs: mean %.4f, max %.4f", sum/float64(len(px)), mx)
	}

	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		t.Run(fmt.Sprint(dev), func(t *testing.T) {
			p := &IdolPipeline{Dev: dev}
			if p.DINO, err = models.LoadDINOv3("testdata/idol-pipe/dinov3", dev); err != nil {
				t.Fatal(err)
			}
			if p.VAE, err = models.LoadFluxVAE("testdata/idol-pipe/vae", dev); err != nil {
				t.Fatal(err)
			}
			f1, f2, err := p.Condition(ref.Pixels, ref.Size, ref.Eps)
			if err != nil {
				t.Fatal(err)
			}
			for _, c := range []struct {
				name string
				x    *tensor.Tensor
				want []float32
			}{{"feature1 (DINOv3)", f1, ref.Feature1}, {"feature2 (FLUX.2 VAE)", f2, ref.Feature2}} {
				h, _ := c.x.To(device.CPU())
				got := tensor.MustSlice[float32](h)
				if len(got) != len(c.want) {
					t.Fatalf("%s: %d values, want %d", c.name, len(got), len(c.want))
				}
				md := 0.0
				for i := range got {
					md = math.Max(md, math.Abs(float64(got[i]-c.want[i])))
				}
				t.Logf("%s: max diff %.3g", c.name, md)
				if md > 1e-4 {
					t.Errorf("%s differs by %.3g", c.name, md)
				}
			}
		})
	}
}
