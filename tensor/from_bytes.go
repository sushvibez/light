package tensor

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/shape"
	"light/storage"
)

// FromBytes constructs a tensor from a raw byte buffer interpreted as
// `dt`-typed, row-major-contiguous values in the given shape. The bytes are
// copied (so the caller may reuse the input slice). On CUDA devices, the
// bytes are uploaded via an intermediate CPU staging tensor.
//
// Primarily used by loaders (safetensors, GGUF) to turn mmap'd file bytes
// into tensors on an arbitrary device.
func FromBytes(data []byte, dt dtype.DType, dims []int, dev device.Device) (*Tensor, error) {
	sh := shape.New(dims...)
	expected := dt.ByteSize(sh.NumElements())
	if expected != len(data) {
		return nil, fmt.Errorf("tensor.FromBytes: got %d bytes for shape %v dtype %v, want %d", len(data), sh, dt, expected)
	}
	s := storage.NewCPU(dt, sh.NumElements())
	copy(s.CPUBytes(), data)
	cpuT := newTensor(s, sh, shape.DefaultStrides(sh), 0)
	if dev.Kind() == device.KindCPU {
		return cpuT, nil
	}
	return cpuT.To(dev)
}

// FromStorage wraps a CPU storage holding row-major contiguous elements as a
// tensor of the given shape, without copying (the loaders' mapped files).
func FromStorage(s *storage.Storage, dims []int) (*Tensor, error) {
	sh := shape.New(dims...)
	if s.Len() != sh.NumElements() {
		return nil, fmt.Errorf("tensor.FromStorage: storage holds %d elements, shape %v wants %d", s.Len(), sh, sh.NumElements())
	}
	return newTensor(s, sh, shape.DefaultStrides(sh), 0), nil
}
