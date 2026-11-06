package data

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"light/device"
	"light/tensor"
)

// CIFAR10 is the 32×32 RGB image dataset, 10 classes, 50k train + 10k test.
type CIFAR10 struct {
	TrainImages *tensor.Tensor // (50000, 3, 32, 32) F32, pixels in [0, 1]
	TrainLabels *tensor.Tensor // (50000,)            I64
	TestImages  *tensor.Tensor // (10000, 3, 32, 32) F32
	TestLabels  *tensor.Tensor // (10000,)            I64
}

// LoadCIFAR10 reads (and optionally downloads into `cacheDir`) the CIFAR-10
// binary distribution. The tarball contains 6 files — data_batch_{1..5}.bin
// and test_batch.bin — each with 10,000 records of 1 label byte + 3072
// channel-major image bytes (1024 R, 1024 G, 1024 B).
func LoadCIFAR10(cacheDir string, dev device.Device) (*CIFAR10, error) {
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return nil, err
	}
	tarPath := filepath.Join(cacheDir, "cifar-10-binary.tar.gz")
	if _, err := os.Stat(tarPath); os.IsNotExist(err) {
		if err := download("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", tarPath); err != nil {
			return nil, fmt.Errorf("cifar-10: download: %w", err)
		}
	}
	// Extract on-demand to cacheDir/cifar-10-batches-bin/.
	extDir := filepath.Join(cacheDir, "cifar-10-batches-bin")
	if _, err := os.Stat(extDir); os.IsNotExist(err) {
		if err := extractTGZ(tarPath, cacheDir); err != nil {
			return nil, fmt.Errorf("cifar-10: extract: %w", err)
		}
	}

	var (
		trainImgs = make([]float32, 0, 50000*3*32*32)
		trainLbls = make([]int64, 0, 50000)
	)
	for i := 1; i <= 5; i++ {
		imgs, lbls, err := readCIFAR10Batch(filepath.Join(extDir, fmt.Sprintf("data_batch_%d.bin", i)))
		if err != nil {
			return nil, err
		}
		trainImgs = append(trainImgs, imgs...)
		trainLbls = append(trainLbls, lbls...)
	}
	testImgs, testLbls, err := readCIFAR10Batch(filepath.Join(extDir, "test_batch.bin"))
	if err != nil {
		return nil, err
	}

	xtr, err := tensor.New(trainImgs, []int{50000, 3, 32, 32}, dev)
	if err != nil {
		return nil, err
	}
	ytr, err := tensor.New(trainLbls, []int{50000}, dev)
	if err != nil {
		return nil, err
	}
	xte, err := tensor.New(testImgs, []int{10000, 3, 32, 32}, dev)
	if err != nil {
		return nil, err
	}
	yte, err := tensor.New(testLbls, []int{10000}, dev)
	if err != nil {
		return nil, err
	}
	return &CIFAR10{TrainImages: xtr, TrainLabels: ytr, TestImages: xte, TestLabels: yte}, nil
}

// readCIFAR10Batch returns (flattened-image-F32-in-[0,1], labels-I64) from
// one CIFAR-10 binary batch file. Each record is 1 label byte + 3072 image
// bytes (channel-major RGB). Output layout matches PyTorch's
// torchvision.datasets.CIFAR10: shape (N, 3, 32, 32).
func readCIFAR10Batch(path string) ([]float32, []int64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	const recordSize = 1 + 3*32*32
	if len(raw)%recordSize != 0 {
		return nil, nil, fmt.Errorf("cifar-10 %s: bad size %d", path, len(raw))
	}
	n := len(raw) / recordSize
	imgs := make([]float32, n*3*32*32)
	lbls := make([]int64, n)
	for i := range n {
		off := i * recordSize
		lbls[i] = int64(raw[off])
		src := raw[off+1 : off+recordSize] // already channel-major (1024 R, 1024 G, 1024 B)
		dst := imgs[i*3*32*32 : (i+1)*3*32*32]
		for k, b := range src {
			dst[k] = float32(b) / 255.0
		}
	}
	return imgs, lbls, nil
}

// extractTGZ unpacks a .tar.gz into destDir.
func extractTGZ(src, destDir string) error {
	f, err := os.Open(src)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		target := filepath.Join(destDir, filepath.Clean(hdr.Name))
		if !strings.HasPrefix(target, filepath.Clean(destDir)+string(os.PathSeparator)) {
			return fmt.Errorf("tar: unsafe path %q", hdr.Name)
		}
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			out, err := os.Create(target)
			if err != nil {
				return err
			}
			if _, err := io.Copy(out, tr); err != nil {
				out.Close()
				return err
			}
			out.Close()
		}
	}
	return nil
}
