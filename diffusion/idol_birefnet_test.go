package diffusion

import (
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"light/device"
	"light/models"
)

// Idol's preprocessing of photos without alpha, BiRefNet included, against
// idol-sdk's preprocess_image with the real BiRefNet (CPU): set
// LIGHT_BIREFNET_WEIGHTS to speridlabs/idol's birefnet.safetensors and
// LIGHT_BIREFNET_PRE to a folder of <name>_src.png (the photo) and
// <name>_pre.png (the SDK's 1024² composite, crop margin 1.7).
func TestIdolPreprocessBiRefNetMatchesSDK(t *testing.T) {
	wpath, dir := os.Getenv("LIGHT_BIREFNET_WEIGHTS"), os.Getenv("LIGHT_BIREFNET_PRE")
	if wpath == "" || dir == "" {
		t.Skip("set LIGHT_BIREFNET_WEIGHTS and LIGHT_BIREFNET_PRE")
	}
	srcs, err := filepath.Glob(filepath.Join(dir, "*_src.png"))
	if err != nil || len(srcs) == 0 {
		t.Fatalf("no *_src.png in %s", dir)
	}
	m, err := models.LoadBiRefNet(wpath, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	read := func(p string) image.Image {
		f, err := os.Open(p)
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
	for _, src := range srcs {
		name := strings.TrimSuffix(src, "_src.png")
		t0 := time.Now()
		px, _, err := idolPreprocess(read(src), 1024, 1.7, m.RemoveBackground)
		if err != nil {
			t.Fatal(err)
		}
		dt := time.Since(t0)
		ref := read(name + "_pre.png")
		var sum, mx float64
		n := 1024 * 1024
		for y := range 1024 {
			for x := range 1024 {
				r, g, b, _ := ref.At(x, y).RGBA()
				for c, v := range [3]uint32{r, g, b} {
					d := math.Abs(float64(px[c*n+y*1024+x]) - float64(v>>8)/255)
					sum += d
					mx = math.Max(mx, d)
				}
			}
		}
		t.Logf("%s: preprocess %.1f s, mean |diff| %.4f, max %.4f (of 1)", filepath.Base(name), dt.Seconds(), sum/float64(3*n), mx)
		if sum/float64(3*n) > 0.01 {
			t.Errorf("%s: preprocessed image differs: mean %.4f", name, sum/float64(3*n))
		}
	}
}
