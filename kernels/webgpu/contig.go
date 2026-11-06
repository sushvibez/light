//go:build webgpu

package webgpu

import (
	"fmt"

	"light/device"
	"light/internal/wgpudev"
	"light/tensor"
)

func init() {
	tensor.RegisterContig(device.KindWebGPU, copyStridedToContig)
	tensor.RegisterCopy(device.KindWebGPU, copyDeviceToDevice)
	tensor.RegisterCopyFromHost(device.KindWebGPU, copyFromHost)
}

// copyStridedToContig implements tensor.Contiguous on WebGPU: gather src
// (arbitrary strides/offset) into the packed dst. The kernel moves 32-bit
// words, so 4-byte dtypes map directly and 8-byte dtypes get a trailing
// axis of size 2 (strides doubled). 1- and 2-byte dtypes are not supported.
func copyStridedToContig(dst, src *tensor.Tensor) error {
	if !dst.IsContiguous() || dst.Offset() != 0 {
		return fmt.Errorf("webgpu Contiguous: dst must be contiguous")
	}
	if !dst.Shape().Equal(src.Shape()) {
		return fmt.Errorf("webgpu Contiguous: shape mismatch")
	}
	if dst.DType() != src.DType() {
		return fmt.Errorf("webgpu Contiguous: dtype mismatch")
	}
	d, err := dev()
	if err != nil {
		return err
	}
	dims := src.Shape().Dims()
	str := src.Strides().Slice()
	off := src.Offset()
	n := src.NumElements()
	if n == 0 {
		return nil
	}
	switch src.DType().Size() {
	case 4:
	case 8:
		dims = append(append([]int(nil), dims...), 2)
		s2 := make([]int, len(str)+1)
		for i, s := range str {
			s2[i] = 2 * s
		}
		s2[len(str)] = 1
		str = s2
		off *= 2
		n *= 2
	default:
		return fmt.Errorf("webgpu Contiguous: dtype %v (%d-byte) not supported; only 4- and 8-byte element types", src.DType(), src.DType().Size())
	}
	meta, err := d.StorageInit(i32Bytes(dims, str))
	if err != nil {
		return err
	}
	defer meta.Release()
	params, err := d.Uniform(uint32(n), uint32(len(dims)), uint32(off), 0)
	if err != nil {
		return err
	}
	defer params.Release()
	return run1D(d, "copy_strided", []wgpudev.Binding{rd(raw(src)), rw(raw(dst)), rd(meta), uni(params)}, n)
}

// copyDeviceToDevice is a whole-buffer copy between two contiguous tensors.
func copyDeviceToDevice(dst, src *tensor.Tensor) error {
	d, err := dev()
	if err != nil {
		return err
	}
	n := uint64(src.DType().ByteSize(src.NumElements()))
	n = (n + 3) &^ 3
	return d.Copy(raw(src), uint64(src.Offset()*src.DType().Size()), raw(dst), 0, n)
}

// copyFromHost streams a contiguous CPU tensor's bytes into dst's existing
// device buffer (no transient allocation).
func copyFromHost(dst, src *tensor.Tensor) error {
	d, err := dev()
	if err != nil {
		return err
	}
	nbytes := src.DType().ByteSize(src.NumElements())
	srcBytes := src.Storage().CPUBytes()
	if nbytes > len(srcBytes) {
		nbytes = len(srcBytes)
	}
	return d.Write(raw(dst), 0, srcBytes[:nbytes])
}
