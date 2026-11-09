package diffusion

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"

	"light/device"
	"light/tensor"
)

// Image wraps a pipeline's output: (1, 3, H, W) values in [0, 1].
type Image struct {
	Pixels *tensor.Tensor
}

// RGBA converts the image to 8-bit RGBA (round to nearest, as diffusers'
// numpy_to_pil: (x * 255).round()).
func (im Image) RGBA() (*image.RGBA, error) {
	t, err := im.Pixels.To(device.CPU())
	if err != nil {
		return nil, err
	}
	if t.Rank() != 4 || t.Dim(0) != 1 || t.Dim(1) != 3 {
		return nil, fmt.Errorf("diffusion.Image: want (1, 3, H, W), got %v", t.Shape())
	}
	H, W := t.Dim(2), t.Dim(3)
	px := tensor.MustSlice[float32](t)
	out := image.NewRGBA(image.Rect(0, 0, W, H))
	to8 := func(v float32) uint8 { return uint8(math.Round(math.Min(math.Max(float64(v), 0), 1) * 255)) }
	for y := range H {
		for x := range W {
			out.SetRGBA(x, y, color.RGBA{to8(px[y*W+x]), to8(px[H*W+y*W+x]), to8(px[2*H*W+y*W+x]), 255})
		}
	}
	return out, nil
}

// SavePNG writes the image to path.
func (im Image) SavePNG(path string) error {
	rgba, err := im.RGBA()
	if err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(f, rgba); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
