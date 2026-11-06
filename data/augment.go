package data

import (
	"math/rand/v2"
)

// CIFAR10Mean / CIFAR10Std are the per-channel statistics computed on the
// 50,000-image training set, expressed in F32 [0,1] pixel space. These
// match the values used by torchvision and the fastai / kmldas Kaggle
// notebook ("CIFAR10 ResNet 90% Accuracy Less Than 5 Min").
var (
	CIFAR10Mean = [3]float32{0.4914, 0.4822, 0.4465}
	CIFAR10Std  = [3]float32{0.2470, 0.2435, 0.2616}
)

// NormalizeCIFAR10 standardises an in-place flat NCHW F32 batch by
// subtracting the per-channel mean and dividing by the per-channel std.
// `imgs` is expected to be `B*3*H*W` floats in row-major NCHW order.
//
// Centring inputs around zero with unit variance speeds up convergence
// significantly on ResNet-style networks; expect ~+1–2% test accuracy
// over un-normalised [0, 1] pixels.
func NormalizeCIFAR10(imgs []float32, batch, h, w int) {
	if len(imgs) != batch*3*h*w {
		panic("NormalizeCIFAR10: bad slice length")
	}
	hw := h * w
	for b := range batch {
		for c := range 3 {
			off := (b*3 + c) * hw
			m, s := CIFAR10Mean[c], CIFAR10Std[c]
			invS := 1 / s
			for i := off; i < off+hw; i++ {
				imgs[i] = (imgs[i] - m) * invS
			}
		}
	}
}

// RandomCropPad augments an in-place NCHW F32 batch by reflecting each
// image's content into a (H+2pad, W+2pad) zero-padded canvas, then
// cropping back a (H, W) window at a uniformly random offset in
// [0, 2pad]² independently per image.
//
// This is the canonical CIFAR-10 augmentation (`torchvision.transforms.
// RandomCrop(32, padding=4)`); combined with horizontal flip it brings a
// well-trained ResNet from ~83% to ~91% test accuracy.
func RandomCropPad(imgs []float32, batch, h, w, pad int, rng *rand.Rand) {
	if len(imgs) != batch*3*h*w {
		panic("RandomCropPad: bad slice length")
	}
	if pad <= 0 {
		return
	}
	ph := h + 2*pad
	pw := w + 2*pad
	scratch := make([]float32, 3*ph*pw)
	for b := range batch {
		base := b * 3 * h * w
		// Zero the scratch and copy the original image into the centre.
		for i := range scratch {
			scratch[i] = 0
		}
		for c := range 3 {
			for y := range h {
				srcRow := base + (c*h+y)*w
				dstRow := (c*ph+(y+pad))*pw + pad
				copy(scratch[dstRow:dstRow+w], imgs[srcRow:srcRow+w])
			}
		}
		// Random crop offset.
		oy := rng.IntN(2*pad + 1)
		ox := rng.IntN(2*pad + 1)
		for c := range 3 {
			for y := range h {
				dstRow := base + (c*h+y)*w
				srcRow := (c*ph+(y+oy))*pw + ox
				copy(imgs[dstRow:dstRow+w], scratch[srcRow:srcRow+w])
			}
		}
	}
}

// RandomHorizontalFlip flips each image left-to-right with probability
// `p` (typically 0.5). In-place on a flat NCHW F32 batch.
func RandomHorizontalFlip(imgs []float32, batch, h, w int, p float64, rng *rand.Rand) {
	if len(imgs) != batch*3*h*w {
		panic("RandomHorizontalFlip: bad slice length")
	}
	for b := range batch {
		if rng.Float64() >= p {
			continue
		}
		base := b * 3 * h * w
		for c := range 3 {
			for y := range h {
				row := base + (c*h+y)*w
				for x := 0; x < w/2; x++ {
					imgs[row+x], imgs[row+w-1-x] = imgs[row+w-1-x], imgs[row+x]
				}
			}
		}
	}
}
