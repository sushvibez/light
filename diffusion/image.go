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

// RGBA converts the image to 8-bit, non-premultiplied RGBA (round to
// nearest, as diffusers' numpy_to_pil: (x * 255).round()). Pixels is
// (1, 3, H, W), or (1, 4, H, W) with alpha (Qwen-Image 2.1's VAE).
func (im Image) RGBA() (*image.NRGBA, error) {
	t, err := im.Pixels.To(device.CPU())
	if err != nil {
		return nil, err
	}
	if t.Rank() != 4 || t.Dim(0) != 1 || (t.Dim(1) != 3 && t.Dim(1) != 4) {
		return nil, fmt.Errorf("diffusion.Image: want (1, 3 or 4, H, W), got %v", t.Shape())
	}
	C, H, W := t.Dim(1), t.Dim(2), t.Dim(3)
	px := tensor.MustSlice[float32](t)
	out := image.NewNRGBA(image.Rect(0, 0, W, H))
	to8 := func(v float32) uint8 { return uint8(math.Round(math.Min(math.Max(float64(v), 0), 1) * 255)) }
	for y := range H {
		for x := range W {
			i := y*W + x
			a := uint8(255)
			if C == 4 {
				a = to8(px[3*H*W+i])
			}
			out.SetNRGBA(x, y, color.NRGBA{to8(px[i]), to8(px[H*W+i]), to8(px[2*H*W+i]), a})
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
