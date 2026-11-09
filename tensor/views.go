package tensor

import (
	"fmt"

	"light/device"
	"light/shape"
)

// Reshape returns a new Tensor with the requested shape, sharing storage
// with the receiver. Supports a single -1 in dims (infer). Requires the
// tensor to be contiguous; non-contiguous tensors must call Contiguous() first.
func (t *Tensor) Reshape(dims ...int) (*Tensor, error) {
	if !t.IsContiguous() || t.offset != 0 {
		return nil, fmt.Errorf("Reshape: tensor is not contiguous; call Contiguous() first")
	}
	// Resolve -1.
	inferIdx := -1
	prod := 1
	for i, d := range dims {
		switch {
		case d == -1:
			if inferIdx != -1 {
				return nil, fmt.Errorf("Reshape: only one dim may be -1")
			}
			inferIdx = i
		case d < 0:
			return nil, fmt.Errorf("Reshape: negative dim %d at axis %d", d, i)
		default:
			prod *= d
		}
	}
	n := t.NumElements()
	if inferIdx != -1 {
		if prod == 0 || n%prod != 0 {
			return nil, fmt.Errorf("Reshape: cannot infer dim from %d elements and shape %v", n, dims)
		}
		dims[inferIdx] = n / prod
	} else if prod != n {
		return nil, fmt.Errorf("Reshape: new shape %v has %d elements, want %d", dims, prod, n)
	}
	newShape := shape.New(dims...)
	return newTensor(t.storage, newShape, shape.DefaultStrides(newShape), t.offset), nil
}

// Permute returns a new Tensor whose axes are reordered according to perm.
// perm must be a permutation of [0..Rank).
func (t *Tensor) Permute(perm ...int) (*Tensor, error) {
	r := t.Rank()
	if len(perm) != r {
		return nil, fmt.Errorf("Permute: perm length %d != rank %d", len(perm), r)
	}
	seen := make(map[int]bool, r)
	for _, p := range perm {
		if p < 0 || p >= r || seen[p] {
			return nil, fmt.Errorf("Permute: invalid perm %v", perm)
		}
		seen[p] = true
	}
	newDims := make([]int, r)
	for i, p := range perm {
		newDims[i] = t.shape.Dim(p)
	}
	newSh := shape.New(newDims...)
	newStrides := t.strides.Permute(perm)
	return newTensor(t.storage, newSh, newStrides, t.offset), nil
}

// Transpose is the 2-axis form of Permute; convenient for matrix transpose.
func (t *Tensor) Transpose(i, j int) (*Tensor, error) {
	r := t.Rank()
	pi, err := t.shape.NormalizeAxis(i)
	if err != nil {
		return nil, err
	}
	pj, err := t.shape.NormalizeAxis(j)
	if err != nil {
		return nil, err
	}
	perm := make([]int, r)
	for k := range perm {
		perm[k] = k
	}
	perm[pi], perm[pj] = pj, pi
	return t.Permute(perm...)
}

// Contiguous returns a new Tensor whose storage is freshly-allocated and
// packed in row-major order, copying elements from the receiver through its
// strides. If the receiver is already contiguous with zero offset, returns
// the receiver unchanged.
func (t *Tensor) Contiguous() (*Tensor, error) {
	if t.IsContiguous() && t.offset == 0 {
		return t, nil
	}
	newT, err := Empty(t.DType(), t.shape.Dims(), t.Device())
	if err != nil {
		return nil, err
	}
	kind := t.Device().Kind()
	if kind == device.KindCPU {
		if err := copyStrided(newT, t); err != nil {
			return nil, err
		}
		return newT, nil
	}
	fn := contigFns[kind]
	if fn == nil {
		if kind == device.KindCUDA {
			return nil, errContigNoCUDA
		}
		return nil, fmt.Errorf("tensor.Contiguous: no %v handler registered (import ops or the kernels package)", kind)
	}
	if err := fn(newT, t); err != nil {
		return nil, err
	}
	return newT, nil
}

// copyStrided copies elements from src (with its strides) into dst (assumed
// contiguous, same shape & dtype). CPU-only for Phase 1.
func copyStrided(dst, src *Tensor) error {
	if dst.Device().Kind() != device.KindCPU {
		return fmt.Errorf("copyStrided: only CPU in Phase 1")
	}
	if dst.DType() != src.DType() {
		return fmt.Errorf("copyStrided: dtype mismatch %v vs %v", dst.DType(), src.DType())
	}
	if !dst.Shape().Equal(src.Shape()) {
		return fmt.Errorf("copyStrided: shape mismatch %v vs %v", dst.Shape(), src.Shape())
	}
	elemSize := dst.DType().Size()
	if elemSize == 0 {
		return fmt.Errorf("copyStrided: zero element size (quantized not supported)")
	}
	dstBuf := dst.storage.CPUBytes()
	srcBuf := src.storage.CPUBytes()
	dims := src.Shape().Dims()
	srcStrides := src.strides.Slice()
	// Iterate all logical indices, compute source byte offset via strides.
	idx := make([]int, len(dims))
	dstByteOff := 0
	for {
		// Compute source byte offset from strides.
		srcElemOff := src.offset
		for a, ix := range idx {
			srcElemOff += ix * srcStrides[a]
		}
		srcByteOff := srcElemOff * elemSize
		copy(dstBuf[dstByteOff:dstByteOff+elemSize], srcBuf[srcByteOff:srcByteOff+elemSize])
		dstByteOff += elemSize
		// Increment multi-index.
		a := len(dims) - 1
		for a >= 0 {
			idx[a]++
			if idx[a] < dims[a] {
				break
			}
			idx[a] = 0
			a--
		}
		if a < 0 {
			break
		}
	}
	return nil
}
