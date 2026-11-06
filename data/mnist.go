// Package data provides dataset loaders for common benchmarks. The MVP
// ships MNIST (IDX ubyte format) with optional on-demand download from a
// public mirror. More datasets (CIFAR-10, Fashion-MNIST) follow the same
// pattern.
package data

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// MNIST is the 28×28 handwritten-digit dataset, 10 classes, 60k train + 10k test.
type MNIST struct {
	TrainImages *tensor.Tensor // (60000, 1, 28, 28) F32, pixels scaled to [0, 1]
	TrainLabels *tensor.Tensor // (60000,)            I64
	TestImages  *tensor.Tensor // (10000, 1, 28, 28) F32
	TestLabels  *tensor.Tensor // (10000,)            I64
}

// LoadMNIST reads (and optionally downloads into `cacheDir`) the four MNIST
// files. Tensors land on `dev` — host-side scaling to F32 happens on CPU and
// transfers over if `dev` is CUDA.
func LoadMNIST(cacheDir string, dev device.Device) (*MNIST, error) {
	files := []struct {
		name, url string
	}{
		{"train-images-idx3-ubyte.gz", "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"},
		{"train-labels-idx1-ubyte.gz", "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"},
		{"t10k-images-idx3-ubyte.gz", "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"},
		{"t10k-labels-idx1-ubyte.gz", "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"},
	}
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return nil, err
	}
	for _, f := range files {
		p := filepath.Join(cacheDir, f.name)
		if _, err := os.Stat(p); os.IsNotExist(err) {
			if err := download(f.url, p); err != nil {
				return nil, fmt.Errorf("mnist: download %s: %w", f.name, err)
			}
		}
	}

	xtr, err := readIDXImages(filepath.Join(cacheDir, "train-images-idx3-ubyte.gz"), dev)
	if err != nil {
		return nil, err
	}
	ytr, err := readIDXLabels(filepath.Join(cacheDir, "train-labels-idx1-ubyte.gz"), dev)
	if err != nil {
		return nil, err
	}
	xte, err := readIDXImages(filepath.Join(cacheDir, "t10k-images-idx3-ubyte.gz"), dev)
	if err != nil {
		return nil, err
	}
	yte, err := readIDXLabels(filepath.Join(cacheDir, "t10k-labels-idx1-ubyte.gz"), dev)
	if err != nil {
		return nil, err
	}
	return &MNIST{TrainImages: xtr, TrainLabels: ytr, TestImages: xte, TestLabels: yte}, nil
}

// readIDXImages reads an IDX3 gzip-compressed image file. Magic 0x00000803,
// dims u32 big-endian: (n, rows, cols). Returns tensor of shape (n, 1, rows, cols) F32.
func readIDXImages(path string, dev device.Device) (*tensor.Tensor, error) {
	r, err := openGzip(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	var hdr [4]uint32
	if err := binary.Read(r, binary.BigEndian, &hdr); err != nil {
		return nil, err
	}
	if hdr[0] != 0x00000803 {
		return nil, fmt.Errorf("idx images: bad magic %#x", hdr[0])
	}
	n, rows, cols := int(hdr[1]), int(hdr[2]), int(hdr[3])
	raw := make([]byte, n*rows*cols)
	if _, err := io.ReadFull(r, raw); err != nil {
		return nil, err
	}
	// Scale u8 pixels [0, 255] → F32 [0, 1].
	data := make([]float32, n*rows*cols)
	for i, b := range raw {
		data[i] = float32(b) / 255.0
	}
	return tensor.New(data, []int{n, 1, rows, cols}, dev)
}

// readIDXLabels reads an IDX1 gzip-compressed label file. Magic 0x00000801,
// dims u32 big-endian: (n). Returns (n,) I64.
func readIDXLabels(path string, dev device.Device) (*tensor.Tensor, error) {
	r, err := openGzip(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	var hdr [2]uint32
	if err := binary.Read(r, binary.BigEndian, &hdr); err != nil {
		return nil, err
	}
	if hdr[0] != 0x00000801 {
		return nil, fmt.Errorf("idx labels: bad magic %#x", hdr[0])
	}
	n := int(hdr[1])
	raw := make([]byte, n)
	if _, err := io.ReadFull(r, raw); err != nil {
		return nil, err
	}
	data := make([]int64, n)
	for i, b := range raw {
		data[i] = int64(b)
	}
	return tensor.New(data, []int{n}, dev)
}

// openGzip opens a gzip-compressed file as a ReadCloser of decompressed bytes.
func openGzip(path string) (io.ReadCloser, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	gz, err := gzip.NewReader(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	return &gzipCloser{gz: gz, f: f}, nil
}

type gzipCloser struct {
	gz *gzip.Reader
	f  *os.File
}

func (g *gzipCloser) Read(p []byte) (int, error) { return g.gz.Read(p) }
func (g *gzipCloser) Close() error {
	_ = g.gz.Close()
	return g.f.Close()
}

// download fetches url and writes it to path (streamed, single-connection).
func download(url, path string) error {
	fmt.Printf("downloading %s...\n", url)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("http %d", resp.StatusCode)
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

// Ensure dtype stays imported.
var _ = dtype.F32
