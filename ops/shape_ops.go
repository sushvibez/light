package ops

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// Pad zero-pads `dim` with `left` elements before and `right` elements after
// (constant padding, value 0). Negative dims count from the back.
//
//	(…, n, …) → (…, left + n + right, …)
func Pad(a *tensor.Tensor, dim, left, right int) (*tensor.Tensor, error) {
	d, err := a.Shape().NormalizeAxis(dim)
	if err != nil {
		return nil, fmt.Errorf("Pad: %w", err)
	}
	if left < 0 || right < 0 {
		return nil, fmt.Errorf("Pad: left/right must be >= 0 (got %d, %d)", left, right)
	}
	src, err := ensureContig(a)
	if err != nil {
		return nil, err
	}
	outDims := src.Shape().Dims()
	outDims[d] += left + right
	out, err := tensor.Zeros(src.DType(), outDims, src.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "pad", Dev: src.Device().Kind(), DType: src.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{src},
		Output: out,
		Attrs:  map[string]any{"dim": d, "left": left, "right": right},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// Flip reverses the order of elements along each listed dim. Negative dims
// count from the back; listing a dim twice is an error.
func Flip(a *tensor.Tensor, dims ...int) (*tensor.Tensor, error) {
	seen := make(map[int]bool, len(dims))
	norm := make([]int, 0, len(dims))
	for _, dim := range dims {
		d, err := a.Shape().NormalizeAxis(dim)
		if err != nil {
			return nil, fmt.Errorf("Flip: %w", err)
		}
		if seen[d] {
			return nil, fmt.Errorf("Flip: dim %d listed twice", d)
		}
		seen[d] = true
		norm = append(norm, d)
	}
	src, err := ensureContig(a)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(src.DType(), src.Shape().Dims(), src.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "flip", Dev: src.Device().Kind(), DType: src.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{src},
		Output: out,
		Attrs:  map[string]any{"dims": norm},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// Roll circularly shifts elements along `dim` by `shift` positions (a
// positive shift moves elements toward higher indices; elements that fall
// off the end wrap around to the front).
func Roll(a *tensor.Tensor, shift, dim int) (*tensor.Tensor, error) {
	d, err := a.Shape().NormalizeAxis(dim)
	if err != nil {
		return nil, fmt.Errorf("Roll: %w", err)
	}
	src, err := ensureContig(a)
	if err != nil {
		return nil, err
	}
	out, err := tensor.Zeros(src.DType(), src.Shape().Dims(), src.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "roll", Dev: src.Device().Kind(), DType: src.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{src},
		Output: out,
		Attrs:  map[string]any{"dim": d, "shift": shift},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// Sort orders each row along the last axis and returns (values, indices):
// values has the dtype of `a`, indices is I64 holding the original position
// of every sorted element. The sort is stable.
//
// CUDA runs a bitonic network (shared memory for rows ≤ 1024, global
// multi-launch beyond) for F32/F64/I32/I64; ties are ordered by original
// index so results are bit-identical to the CPU stable sort.
func Sort(a *tensor.Tensor, descending bool) (*tensor.Tensor, *tensor.Tensor, error) {
	if a.Rank() < 1 {
		return nil, nil, fmt.Errorf("Sort: rank must be >= 1")
	}
	src, err := ensureContig(a)
	if err != nil {
		return nil, nil, err
	}
	dims := src.Shape().Dims()
	vals, err := tensor.Zeros(src.DType(), dims, src.Device())
	if err != nil {
		return nil, nil, err
	}
	idx, err := tensor.Zeros(dtype.I64, dims, src.Device())
	if err != nil {
		return nil, nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: "sort", Dev: src.Device().Kind(), DType: src.DType()})
	if err != nil {
		return nil, nil, err
	}
	if err := k(&registry.Context{
		Inputs: []*tensor.Tensor{src, idx},
		Output: vals,
		Attrs:  map[string]any{"descending": descending},
	}); err != nil {
		return nil, nil, err
	}
	return vals, idx, nil
}

// Norm computes the Lp norm (sum |x|^p)^(1/p) over the axes in opts
// (all elements → scalar when opts.Axes is empty). F32/F64 only; p must be
// positive and finite.
//
// On CUDA this lowers to Abs → Pow → Sum → Pow using the existing kernels
// (p=1 skips the pows) rather than a dedicated kernel.
func Norm(a *tensor.Tensor, p float64, opts ReduceOptions) (*tensor.Tensor, error) {
	if a.DType() != dtype.F32 && a.DType() != dtype.F64 {
		return nil, fmt.Errorf("Norm: dtype %v not supported (F32/F64 only)", a.DType())
	}
	if !(p > 0) || p > 1e300 {
		return nil, fmt.Errorf("Norm: p must be a positive finite number (got %v)", p)
	}
	src, err := ensureContig(a)
	if err != nil {
		return nil, err
	}
	if device.IsCUDA(src.Device()) {
		return normCUDALowered(src, p, opts)
	}
	return reduceDispatchAttrs("lpnorm", src, opts, map[string]any{"p": p})
}

// normCUDALowered expresses the Lp norm with elementwise + reduce ops that
// already have CUDA kernels.
func normCUDALowered(a *tensor.Tensor, p float64, opts ReduceOptions) (*tensor.Tensor, error) {
	absA, err := Abs(a)
	if err != nil {
		return nil, err
	}
	powd := absA
	if p != 1 {
		pT, err := fullLike(p, absA)
		if err != nil {
			return nil, err
		}
		if powd, err = Pow(absA, pT); err != nil {
			return nil, err
		}
	}
	s, err := Sum(powd, opts)
	if err != nil {
		return nil, err
	}
	if p == 1 {
		return s, nil
	}
	invT, err := fullLike(1/p, s)
	if err != nil {
		return nil, err
	}
	return Pow(s, invT)
}

// fullLike builds a tensor with the shape/dtype/device of `like` filled
// with v (F32 or F64).
func fullLike(v float64, like *tensor.Tensor) (*tensor.Tensor, error) {
	dims := like.Shape().Dims()
	switch like.DType() {
	case dtype.F32:
		return tensor.Full(float32(v), dims, like.Device())
	case dtype.F64:
		return tensor.Full(v, dims, like.Device())
	}
	return nil, fmt.Errorf("fullLike: unsupported dtype %v", like.DType())
}

// reduceDispatchAttrs is reduceDispatch with extra kernel attributes merged
// into the context (used by parameterised reductions such as the Lp norm).
func reduceDispatchAttrs(op string, a *tensor.Tensor, opts ReduceOptions, extra map[string]any) (*tensor.Tensor, error) {
	rank := a.Rank()
	axes := append([]int(nil), opts.Axes...)
	if len(axes) == 0 {
		axes = make([]int, rank)
		for i := range axes {
			axes[i] = i
		}
	}
	seen := make(map[int]bool, len(axes))
	for i, ax := range axes {
		if ax < 0 {
			ax += rank
		}
		if ax < 0 || ax >= rank {
			return nil, fmt.Errorf("%s: axis %d out of range for rank %d", op, opts.Axes[i], rank)
		}
		if seen[ax] {
			return nil, fmt.Errorf("%s: duplicate axis %d", op, ax)
		}
		seen[ax] = true
		axes[i] = ax
	}
	inDims := a.Shape().Dims()
	outDims := []int{}
	for ax := range rank {
		switch {
		case seen[ax] && opts.KeepDims:
			outDims = append(outDims, 1)
		case seen[ax]:
		default:
			outDims = append(outDims, inDims[ax])
		}
	}
	out, err := tensor.Zeros(a.DType(), outDims, a.Device())
	if err != nil {
		return nil, err
	}
	attrs := map[string]any{"axes": axes, "keepdims": opts.KeepDims}
	for k, v := range extra {
		attrs[k] = v
	}
	k, err := registry.Lookup(registry.Key{Op: op, Dev: a.Device().Kind(), DType: a.DType()})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{Inputs: []*tensor.Tensor{a}, Output: out, Attrs: attrs}); err != nil {
		return nil, err
	}
	return out, nil
}

// Unfold extracts sliding windows of `size` every `step` along `dim`:
// the window count replaces `dim` and the window itself becomes a new last
// axis, as in Candle/PyTorch `unfold`. (…, n, …) → (…, (n-size)/step+1, …, size).
func Unfold(a *tensor.Tensor, dim, size, step int) (*tensor.Tensor, error) {
	d, err := a.Shape().NormalizeAxis(dim)
	if err != nil {
		return nil, fmt.Errorf("Unfold: %w", err)
	}
	n := a.Dim(d)
	if size <= 0 || step <= 0 || size > n {
		return nil, fmt.Errorf("Unfold: size %d / step %d invalid for length %d", size, step, n)
	}
	count := (n-size)/step + 1
	windows := make([]*tensor.Tensor, count)
	for i := range count {
		w, err := a.Narrow(d, i*step, size)
		if err != nil {
			return nil, err
		}
		windows[i] = w
	}
	// Stack → (…, count, size, …); move `size` to the end.
	st, err := Stack(windows, d)
	if err != nil {
		return nil, err
	}
	rank := st.Rank()
	perm := make([]int, 0, rank)
	for i := range rank {
		if i != d+1 {
			perm = append(perm, i)
		}
	}
	perm = append(perm, d+1)
	p, err := st.Permute(perm...)
	if err != nil {
		return nil, err
	}
	return ensureContig(p)
}

// PixelShuffle rearranges (N, C·r², H, W) → (N, C, H·r, W·r).
func PixelShuffle(a *tensor.Tensor, r int) (*tensor.Tensor, error) {
	if a.Rank() != 4 || r <= 0 || a.Dim(1)%(r*r) != 0 {
		return nil, fmt.Errorf("PixelShuffle: want (N, C·r², H, W) with r=%d, got %v", r, a.Shape())
	}
	N, C, H, W := a.Dim(0), a.Dim(1)/(r*r), a.Dim(2), a.Dim(3)
	src, err := ensureContig(a)
	if err != nil {
		return nil, err
	}
	x, err := src.Reshape(N, C, r, r, H, W)
	if err != nil {
		return nil, err
	}
	if x, err = x.Permute(0, 1, 4, 2, 5, 3); err != nil { // (N, C, H, r, W, r)
		return nil, err
	}
	if x, err = ensureContig(x); err != nil {
		return nil, err
	}
	return x.Reshape(N, C, H*r, W*r)
}

// PixelUnshuffle is the inverse: (N, C, H·r, W·r) → (N, C·r², H, W).
func PixelUnshuffle(a *tensor.Tensor, r int) (*tensor.Tensor, error) {
	if a.Rank() != 4 || r <= 0 || a.Dim(2)%r != 0 || a.Dim(3)%r != 0 {
		return nil, fmt.Errorf("PixelUnshuffle: want (N, C, H·r, W·r) with r=%d, got %v", r, a.Shape())
	}
	N, C, H, W := a.Dim(0), a.Dim(1), a.Dim(2)/r, a.Dim(3)/r
	src, err := ensureContig(a)
	if err != nil {
		return nil, err
	}
	x, err := src.Reshape(N, C, H, r, W, r)
	if err != nil {
		return nil, err
	}
	if x, err = x.Permute(0, 1, 3, 5, 2, 4); err != nil { // (N, C, r, r, H, W)
		return nil, err
	}
	if x, err = ensureContig(x); err != nil {
		return nil, err
	}
	return x.Reshape(N, C*r*r, H, W)
}

// ReplicationPad2d pads the last two axes of (N, C, H, W) by repeating the
// edge rows/columns (PyTorch `ReplicationPad2d((left, right, top, bottom))`).
func ReplicationPad2d(a *tensor.Tensor, left, right, top, bottom int) (*tensor.Tensor, error) {
	if a.Rank() != 4 {
		return nil, fmt.Errorf("ReplicationPad2d: want (N, C, H, W), got %v", a.Shape())
	}
	if left < 0 || right < 0 || top < 0 || bottom < 0 {
		return nil, fmt.Errorf("ReplicationPad2d: negative padding")
	}
	x, err := replicateEdge(a, 3, left, right)
	if err != nil {
		return nil, err
	}
	return replicateEdge(x, 2, top, bottom)
}

// replicateEdge concatenates `before` copies of the first slice and `after`
// copies of the last slice along dim.
func replicateEdge(a *tensor.Tensor, dim, before, after int) (*tensor.Tensor, error) {
	if before == 0 && after == 0 {
		return ensureContig(a)
	}
	n := a.Dim(dim)
	edge := func(start, count int) (*tensor.Tensor, error) {
		s, err := a.Narrow(dim, start, 1)
		if err != nil {
			return nil, err
		}
		dims := s.Shape().Dims()
		dims[dim] = count
		e, err := s.Expand(dims...)
		if err != nil {
			return nil, err
		}
		return ensureContig(e)
	}
	var parts []*tensor.Tensor
	if before > 0 {
		b, err := edge(0, before)
		if err != nil {
			return nil, err
		}
		parts = append(parts, b)
	}
	mid, err := ensureContig(a)
	if err != nil {
		return nil, err
	}
	parts = append(parts, mid)
	if after > 0 {
		e, err := edge(n-1, after)
		if err != nil {
			return nil, err
		}
		parts = append(parts, e)
	}
	return Concat(parts, dim)
}
