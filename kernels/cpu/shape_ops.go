package cpu

import (
	"math"
	"sort"

	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	reg := func(op string, dt dtype.DType, k registry.Kernel) {
		registry.Register(registry.Key{Op: op, Dev: device.KindCPU, DType: dt}, k)
	}
	// Index-mapping ops: every scalar dtype with a Go counterpart.
	reg("pad", dtype.F32, padKernel[float32])
	reg("pad", dtype.F64, padKernel[float64])
	reg("pad", dtype.I32, padKernel[int32])
	reg("pad", dtype.I64, padKernel[int64])
	reg("pad", dtype.U8, padKernel[uint8])
	reg("pad", dtype.I8, padKernel[int8])

	reg("flip", dtype.F32, flipKernel[float32])
	reg("flip", dtype.F64, flipKernel[float64])
	reg("flip", dtype.I32, flipKernel[int32])
	reg("flip", dtype.I64, flipKernel[int64])
	reg("flip", dtype.U8, flipKernel[uint8])
	reg("flip", dtype.I8, flipKernel[int8])

	reg("roll", dtype.F32, rollKernel[float32])
	reg("roll", dtype.F64, rollKernel[float64])
	reg("roll", dtype.I32, rollKernel[int32])
	reg("roll", dtype.I64, rollKernel[int64])
	reg("roll", dtype.U8, rollKernel[uint8])
	reg("roll", dtype.I8, rollKernel[int8])

	reg("sort", dtype.F32, sortKernel[float32])
	reg("sort", dtype.F64, sortKernel[float64])
	reg("sort", dtype.I32, sortKernel[int32])
	reg("sort", dtype.I64, sortKernel[int64])

	reg("lpnorm", dtype.F32, lpNormKernel[float32])
	reg("lpnorm", dtype.F64, lpNormKernel[float64])
}

// splitAroundDim decomposes a contiguous row-major shape into
// (outer, size, inner) around `dim`: outer = prod(dims[:dim]),
// size = dims[dim], inner = prod(dims[dim+1:]). A flat index then reads
// idx = (o*size + d)*inner + i.
func splitAroundDim(dims []int, dim int) (outer, size, inner int) {
	outer, inner = 1, 1
	for i := 0; i < dim; i++ {
		outer *= dims[i]
	}
	for i := dim + 1; i < len(dims); i++ {
		inner *= dims[i]
	}
	return outer, dims[dim], inner
}

// padKernel zero-pads `dim` with `left` elements before and `right` after.
//
//	Inputs[0]: contiguous input
//	Output:    contiguous, dims[dim] = in + left + right (pre-zeroed)
//	Attrs:     "dim" int, "left" int, "right" int
func padKernel[T dtype.Numeric](ctx *registry.Context) error {
	in, out := ctx.Inputs[0], ctx.Output
	dim := ctx.Attrs["dim"].(int)
	left := ctx.Attrs["left"].(int)
	inS := asSlice[T](in)
	outS := asSlice[T](out)
	outer, size, inner := splitAroundDim(in.Shape().Dims(), dim)
	outSize := out.Dim(dim)
	for o := range outer {
		for d := range size {
			src := (o*size + d) * inner
			dst := (o*outSize + d + left) * inner
			copy(outS[dst:dst+inner], inS[src:src+inner])
		}
	}
	return nil
}

// flipKernel reverses the input along every dim listed in Attrs["dims"].
func flipKernel[T dtype.Numeric](ctx *registry.Context) error {
	in, out := ctx.Inputs[0], ctx.Output
	flipDims := ctx.Attrs["dims"].([]int)
	inS := asSlice[T](in)
	outS := asSlice[T](out)
	dims := in.Shape().Dims()
	rank := len(dims)
	flip := make([]bool, rank)
	for _, d := range flipDims {
		flip[d] = true
	}
	strides := make([]int, rank)
	s := 1
	for k := rank - 1; k >= 0; k-- {
		strides[k] = s
		s *= dims[k]
	}
	n := in.NumElements()
	idx := make([]int, rank)
	for i := range n {
		o := 0
		for k := range rank {
			c := idx[k]
			if flip[k] {
				c = dims[k] - 1 - c
			}
			o += c * strides[k]
		}
		outS[o] = inS[i]
		for k := rank - 1; k >= 0; k-- {
			idx[k]++
			if idx[k] < dims[k] {
				break
			}
			idx[k] = 0
		}
	}
	return nil
}

// rollKernel circularly shifts along Attrs["dim"] by Attrs["shift"].
func rollKernel[T dtype.Numeric](ctx *registry.Context) error {
	in, out := ctx.Inputs[0], ctx.Output
	dim := ctx.Attrs["dim"].(int)
	shift := ctx.Attrs["shift"].(int)
	inS := asSlice[T](in)
	outS := asSlice[T](out)
	outer, size, inner := splitAroundDim(in.Shape().Dims(), dim)
	if size == 0 {
		return nil
	}
	shift = ((shift % size) + size) % size
	for o := range outer {
		for d := range size {
			nd := (d + shift) % size
			src := (o*size + d) * inner
			dst := (o*size + nd) * inner
			copy(outS[dst:dst+inner], inS[src:src+inner])
		}
	}
	return nil
}

// sortKernel sorts each last-axis row. Stable, so equal keys keep their
// original relative order (and the index output is deterministic).
//
//	Inputs[0]: contiguous input (..., n)
//	Inputs[1]: I64 index output (..., n) — written by the kernel
//	Output:    sorted values (..., n)
//	Attrs:     "descending" bool
func sortKernel[T dtype.Numeric](ctx *registry.Context) error {
	in, idxT, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	desc := ctx.Attrs["descending"].(bool)
	inS := asSlice[T](in)
	outS := asSlice[T](out)
	idxS := asSlice[int64](idxT)
	cols := in.Dim(-1)
	if cols == 0 {
		return nil
	}
	rows := in.NumElements() / cols
	order := make([]int, cols)
	for r := range rows {
		base := r * cols
		row := inS[base : base+cols]
		for j := range order {
			order[j] = j
		}
		if desc {
			sort.SliceStable(order, func(a, b int) bool { return row[order[a]] > row[order[b]] })
		} else {
			sort.SliceStable(order, func(a, b int) bool { return row[order[a]] < row[order[b]] })
		}
		for j, o := range order {
			outS[base+j] = row[o]
			idxS[base+j] = int64(o)
		}
	}
	return nil
}

// lpNormKernel computes (sum |x|^p)^(1/p) over the reduced axes
// (Attrs["axes"], "keepdims" as in the other reductions; Attrs["p"] float64).
func lpNormKernel[T dtype.Numeric](ctx *registry.Context) error {
	p := ctx.Attrs["p"].(float64)
	out := ctx.Output
	outS := asSlice[T](out)
	acc := make([]float64, len(outS))
	reduceWalk[T](ctx, func(outIdx int, v T) {
		acc[outIdx] += math.Pow(math.Abs(float64(v)), p)
	})
	for i, a := range acc {
		outS[i] = T(math.Pow(a, 1.0/p))
	}
	return nil
}
