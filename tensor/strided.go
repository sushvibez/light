package tensor

import (
	"fmt"

	"light/device"
)

// Assign copies src into the receiver in place, where the receiver may be
// a strided view (a narrowed block of a larger buffer) as long as it is a
// stack of contiguous rows: trailing axes packed, leading axes collapsing
// to one row pitch. A KV-cache append, (B, KH, T, HD) into columns
// [len, len+T) of a (B, KH, cap, HD) buffer, is exactly that shape, so it
// is one cudaMemcpy2D on CUDA and one loop of copies on CPU. src must be
// contiguous (it is contiguified otherwise) and match shape, dtype and
// device.
func (t *Tensor) Assign(src *Tensor) error {
	if !t.Shape().Equal(src.Shape()) {
		return fmt.Errorf("tensor.Assign: shape mismatch (%v vs %v)", t.Shape(), src.Shape())
	}
	if t.DType() != src.DType() {
		return fmt.Errorf("tensor.Assign: dtype mismatch (%v vs %v)", t.DType(), src.DType())
	}
	if !t.Device().SameAs(src.Device()) {
		return fmt.Errorf("tensor.Assign: device mismatch (%v vs %v)", t.Device(), src.Device())
	}
	if t.IsContiguous() && t.offset == 0 {
		return t.CopyFrom(src)
	}
	src, err := src.Contiguous()
	if err != nil {
		return err
	}
	rows, width, pitch, ok := t.rowLayout()
	if !ok {
		return fmt.Errorf("tensor.Assign: destination view %v / strides %v is not a stack of contiguous rows", t.Shape(), t.Strides())
	}
	es := t.DType().Size()
	if es == 0 {
		return fmt.Errorf("tensor.Assign: quantized dtype %v", t.DType())
	}
	if t.NumElements() == 0 {
		return nil
	}
	switch kind := t.Device().Kind(); kind {
	case device.KindCPU:
		dst := t.storage.CPUBytes()
		s := src.storage.CPUBytes()
		for r := 0; r < rows; r++ {
			d := (t.offset + r*pitch) * es
			copy(dst[d:d+width*es], s[r*width*es:(r+1)*width*es])
		}
		return nil
	default:
		fn := pitchedCopyFns[kind]
		if fn == nil {
			return fmt.Errorf("tensor.Assign: no %v pitched-copy hook registered (build with the matching tag and import ops)", kind)
		}
		return fn(t, src, rows, width, pitch)
	}
}

// rowLayout collapses the receiver's view into `rows` runs of `width`
// packed elements whose starts are `pitch` elements apart. ok is false when
// the strides do not have that shape.
func (t *Tensor) rowLayout() (rows, width, pitch int, ok bool) {
	dims := t.shape.Dims()
	st := t.strides.Slice()
	width = 1
	i := len(dims) - 1
	// Packed suffix.
	for ; i >= 0; i-- {
		if dims[i] == 1 {
			continue
		}
		if st[i] != width {
			break
		}
		width *= dims[i]
	}
	if i < 0 {
		return 1, width, width, true
	}
	// Everything above axis i must collapse to one pitch.
	pitch = st[i]
	rows = dims[i]
	for j := i - 1; j >= 0; j-- {
		if dims[j] == 1 {
			continue
		}
		if st[j] != rows*pitch {
			return 0, 0, 0, false
		}
		rows *= dims[j]
	}
	if pitch < width {
		return 0, 0, 0, false
	}
	return rows, width, pitch, true
}

// PitchedCopyFunc writes `rows` runs of `width` elements from contiguous
// src into dst's storage at dst.Offset() + r·pitch (elements).
type PitchedCopyFunc func(dst, src *Tensor, rows, width, pitch int) error

var pitchedCopyFns = map[device.Kind]PitchedCopyFunc{}

// RegisterPitchedCopy wires the device implementation of Assign's strided
// destination copy for a device kind.
func RegisterPitchedCopy(kind device.Kind, fn PitchedCopyFunc) { pitchedCopyFns[kind] = fn }
