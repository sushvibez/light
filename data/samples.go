package data

import (
	"fmt"

	"light/device"
	"light/tensor"
)

// CIFAR10ClassNames maps the 10 class indices to human-readable labels in
// the order used by torchvision and the upstream `cifar-10-binary` files.
var CIFAR10ClassNames = [10]string{
	"airplane", "automobile", "bird", "cat", "deer",
	"dog", "frog", "horse", "ship", "truck",
}

// MNISTClassNames is just the digit names — included for symmetry.
var MNISTClassNames = [10]string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

// CIFAR10TestSamples returns the first `n` test-set images and labels as
// fresh tensors on `dev`. Useful for quick inference demos that don't need
// the whole 10k-image test split in memory.
func CIFAR10TestSamples(cacheDir string, n int, dev device.Device) (*tensor.Tensor, *tensor.Tensor, error) {
	if n <= 0 || n > 10000 {
		return nil, nil, fmt.Errorf("CIFAR10TestSamples: n must be in [1, 10000], got %d", n)
	}
	cifar, err := LoadCIFAR10(cacheDir, device.CPU())
	if err != nil {
		return nil, nil, err
	}
	imgs, err := cifar.TestImages.Narrow(0, 0, n)
	if err != nil {
		return nil, nil, err
	}
	imgsC, err := imgs.Contiguous()
	if err != nil {
		return nil, nil, err
	}
	lbls, err := cifar.TestLabels.Narrow(0, 0, n)
	if err != nil {
		return nil, nil, err
	}
	lblsC, err := lbls.Contiguous()
	if err != nil {
		return nil, nil, err
	}
	if !dev.SameAs(device.CPU()) {
		imgsC, err = imgsC.To(dev)
		if err != nil {
			return nil, nil, err
		}
		lblsC, err = lblsC.To(dev)
		if err != nil {
			return nil, nil, err
		}
	}
	return imgsC, lblsC, nil
}

// MNISTTestSamples returns the first `n` test-set images (as F32 in [0,1])
// and labels (I64) for inference demos.
func MNISTTestSamples(cacheDir string, n int, dev device.Device) (*tensor.Tensor, *tensor.Tensor, error) {
	if n <= 0 || n > 10000 {
		return nil, nil, fmt.Errorf("MNISTTestSamples: n must be in [1, 10000], got %d", n)
	}
	mnist, err := LoadMNIST(cacheDir, device.CPU())
	if err != nil {
		return nil, nil, err
	}
	imgs, err := mnist.TestImages.Narrow(0, 0, n)
	if err != nil {
		return nil, nil, err
	}
	imgsC, err := imgs.Contiguous()
	if err != nil {
		return nil, nil, err
	}
	lbls, err := mnist.TestLabels.Narrow(0, 0, n)
	if err != nil {
		return nil, nil, err
	}
	lblsC, err := lbls.Contiguous()
	if err != nil {
		return nil, nil, err
	}
	if !dev.SameAs(device.CPU()) {
		imgsC, err = imgsC.To(dev)
		if err != nil {
			return nil, nil, err
		}
		lblsC, err = lblsC.To(dev)
		if err != nil {
			return nil, nil, err
		}
	}
	return imgsC, lblsC, nil
}
