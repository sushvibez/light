//go:build !nocuda && cgo

package cuda

import (
	"light/device"
	"light/internal/cudart"
	"light/tensor"
)

func init() {
	tensor.RegisterCUDACopy(cudaCopyImpl)
	tensor.RegisterCUDACopyFromHost(cudaCopyHostImpl)
}

// cudaCopyImpl is a straight cudaMemcpyDtoD between two contiguous tensors
// on the same device.
func cudaCopyImpl(dst, src *tensor.Tensor) error {
	bytes := uintptr(src.NumElements()) * uintptr(src.DType().Size())
	return cudart.MemcpyDtoD(dst.Storage().CUDAPtr(), src.Storage().CUDAPtr(), bytes)
}

// cudaCopyHostImpl streams `src`'s CPU bytes into `dst`'s CUDA storage in
// place — no transient GPU tensor is allocated, so the caller can load a
// 3B+ checkpoint without piling up per-layer transients waiting on GC.
func cudaCopyHostImpl(dst, src *tensor.Tensor) error {
	if err := cudart.SetDevice(dst.Device().Ordinal()); err != nil {
		return err
	}
	nbytes := src.DType().ByteSize(src.NumElements())
	srcBytes := src.Storage().CPUBytes()
	if nbytes > len(srcBytes) {
		nbytes = len(srcBytes)
	}
	return cudart.MemcpyHtoD(dst.Storage().CUDAPtr(), srcBytes[:nbytes])
}

func init() {
	tensor.RegisterPitchedCopy(device.KindCUDA, cudaPitchedCopyImpl)
}

// cudaPitchedCopyImpl is tensor.Assign's CUDA path: one cudaMemcpy2D writes
// `rows` runs of `width` elements from contiguous src into dst's storage,
// `pitch` elements apart (a KV-cache append into a capacity buffer).
func cudaPitchedCopyImpl(dst, src *tensor.Tensor, rows, width, pitch int) error {
	es := uintptr(dst.DType().Size())
	d := dst.Storage().CUDAPtr() + uintptr(dst.Offset())*es
	return cudart.Memcpy2DDtoD(d, uintptr(pitch)*es, src.Storage().CUDAPtr(), uintptr(width)*es, uintptr(width)*es, uintptr(rows))
}
