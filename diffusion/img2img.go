package diffusion

import (
	"fmt"
	"image"

	"light/device"
	"light/models"
	"light/ops"
	"light/tensor"
)

// Img2Img holds the image-to-image inputs shared by the flow-matching
// pipelines (FLUX.1, Z-Image): the init image and how much of the schedule
// to run.
type Img2Img struct {
	Image    image.Image
	Strength float32 // 0..1: the fraction of the schedule that runs (1 = from pure noise)
}

// startStep is diffusers' get_timesteps: the schedule is cut so only
// strength · n steps run.
func startStep(n int, strength float32) int {
	init := min(float64(n)*float64(strength), float64(n))
	return int(max(float64(n)-init, 0))
}

// imageLatents encodes img at (w, h) pixels to normalised latents
// (1, C, h/s, w/s): the VAE's mean, then (z − shift) · scale. The image is
// resized when its size differs (Pillow bicubic; diffusers uses Lanczos).
func imageLatents(vae *models.FluxVAE, img image.Image, w, h int, dev device.Device) (*tensor.Tensor, error) {
	if vae.Encoder == nil {
		return nil, fmt.Errorf("img2img: the VAE has no encoder weights")
	}
	rgb, iw, ih := models.ImageRGB(img)
	if iw != w || ih != h {
		rgb = models.ResizeRGB(rgb, iw, ih, w, h)
	}
	px := make([]float32, 3*h*w)
	for c := range 3 {
		for i := range h * w {
			px[c*h*w+i] = float32(rgb[i*3+c])/255*2 - 1
		}
	}
	x, err := tensor.New(px, []int{1, 3, h, w}, dev)
	if err != nil {
		return nil, err
	}
	z, err := vae.Encoder.Mode(x)
	if err != nil {
		return nil, err
	}
	sh, _ := tensor.New([]float32{vae.ShiftFactor}, []int{1}, dev)
	sc, _ := tensor.New([]float32{vae.ScalingFactor}, []int{1}, dev)
	if z, err = ops.Sub(z, sh); err != nil {
		return nil, err
	}
	return ops.Mul(z, sc)
}

// noised is the scheduler's scale_noise: sigma · noise + (1 − sigma) · z.
func noised(z, noise *tensor.Tensor, sigma float32) (*tensor.Tensor, error) {
	dev := z.Device()
	s, _ := tensor.New([]float32{sigma}, []int{1}, dev)
	r, _ := tensor.New([]float32{1 - sigma}, []int{1}, dev)
	a, err := ops.Mul(noise, s)
	if err != nil {
		return nil, err
	}
	b, err := ops.Mul(z, r)
	if err != nil {
		return nil, err
	}
	return ops.Add(a, b)
}

// defaultSize is an init image's size cut to a multiple of unit, for
// pipelines whose width / height default to the image's.
func defaultSize(img image.Image, unit int) (int, int) {
	b := img.Bounds()
	return (b.Dx() / unit) * unit, (b.Dy() / unit) * unit
}
