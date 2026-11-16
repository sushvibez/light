package diffusion

import (
	"image"
	"image/color"
	"math"
	"os"
	"testing"
	"time"

	"light/device"
	"light/models"
)

// An opaque image goes through the matte: the result is the same as
// preprocessing the RGBA image carrying that matte as its alpha, and
// without a matte it is refused. Images over 1024 on their long side are
// shrunk before matting (TRELLIS does so before rembg).
func TestTrellisPreprocessMatte(t *testing.T) {
	obj := trellisReadPNG(t, trellisTiny+"/object.png")
	b := obj.Bounds()
	w, h := b.Dx(), b.Dy()
	opaque := image.NewNRGBA(image.Rect(0, 0, w, h))
	alpha := make([]uint8, w*h)
	for y := range h {
		for x := range w {
			c := color.NRGBAModel.Convert(obj.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
			alpha[y*w+x] = c.A
			c.A = 255
			opaque.SetNRGBA(x, y, c)
		}
	}
	withAlpha := image.NewNRGBA(opaque.Rect)
	copy(withAlpha.Pix, opaque.Pix)
	for i, a := range alpha {
		withAlpha.Pix[4*i+3] = a
	}
	if HasAlpha(opaque) || !HasAlpha(withAlpha) {
		t.Fatal("HasAlpha")
	}
	if _, _, err := trellisPreprocess(opaque, 64, nil); err == nil {
		t.Fatal("an opaque image without a matte was accepted")
	}
	calls := 0
	matte := func(rgb []float32, mw, mh int) ([]uint8, error) {
		calls++
		if mw != w || mh != h || len(rgb) != 3*w*h {
			t.Fatalf("matte got %d x %d (%d values), want %d x %d", mw, mh, len(rgb), w, h)
		}
		return alpha, nil
	}
	got, _, err := trellisPreprocess(opaque, 64, matte)
	if err != nil {
		t.Fatal(err)
	}
	want, _, err := trellisPreprocess(withAlpha, 64, nil)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("matte called %d times", calls)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("value %d: %v, want %v", i, got[i], want[i])
		}
	}

	big := image.NewNRGBA(image.Rect(0, 0, 2050, 1000))
	for i := range big.Pix {
		big.Pix[i] = 255
	}
	var mw, mh int
	_, _, err = trellisPreprocess(big, 64, func(rgb []float32, w, h int) ([]uint8, error) {
		mw, mh = w, h
		a := make([]uint8, w*h)
		for y := h / 4; y < 3*h/4; y++ {
			for x := w / 4; x < 3*w/4; x++ {
				a[y*w+x] = 255
			}
		}
		return a, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if mw != 1024 || mh != 499 {
		t.Errorf("matte ran at %d x %d, want 1024 x 499 (PIL's int(1000 * 1024 / 2050))", mw, mh)
	}
}

// The real BiRefNet (LIGHT_BIREFNET_WEIGHTS, speridlabs/idol's
// birefnet.safetensors) cutting out a TRELLIS example object composited on
// white (LIGHT_TRELLIS_MATTE_IMAGE, an RGBA cut-out): TRELLIS's 518² input
// comes out close to the one from the original alpha.
func TestTrellisPreprocessBiRefNetReal(t *testing.T) {
	wpath, ipath := os.Getenv("LIGHT_BIREFNET_WEIGHTS"), os.Getenv("LIGHT_TRELLIS_MATTE_IMAGE")
	if wpath == "" || ipath == "" {
		t.Skip("set LIGHT_BIREFNET_WEIGHTS and LIGHT_TRELLIS_MATTE_IMAGE")
	}
	m, err := models.LoadBiRefNet(wpath, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	obj := trellisReadPNG(t, ipath)
	b := obj.Bounds()
	onWhite := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := range b.Dy() {
		for x := range b.Dx() {
			c := color.NRGBAModel.Convert(obj.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
			a := float64(c.A) / 255
			mix := func(v uint8) uint8 { return uint8(math.Round(float64(v)*a + 255*(1-a))) }
			onWhite.SetNRGBA(x, y, color.NRGBA{mix(c.R), mix(c.G), mix(c.B), 255})
		}
	}
	const S = 518
	t0 := time.Now()
	got, gimg, err := trellisPreprocess(onWhite, S, m.RemoveBackground)
	if err != nil {
		t.Fatal(err)
	}
	dt := time.Since(t0)
	want, _, err := trellisPreprocess(obj, S, nil)
	if err != nil {
		t.Fatal(err)
	}
	var sum float64
	far := 0
	for i := range want {
		d := math.Abs(float64(got[i] - want[i]))
		sum += d
		if d > 0.1 {
			far++
		}
	}
	mean := sum / float64(len(want))
	t.Logf("BiRefNet cut-out in %.1f s: mean |diff| %.4f vs the original alpha, %d of %d values off by > 0.1", dt.Seconds(), mean, far, len(want))
	if out := os.Getenv("LIGHT_TRELLIS_MATTE_OUT"); out != "" {
		if err := savePNGTest(out, gimg); err != nil {
			t.Fatal(err)
		}
	}
	if mean > 0.02 || far > len(want)/50 {
		t.Errorf("cut-out differs from the original alpha: mean %.4f, %d far", mean, far)
	}
}
