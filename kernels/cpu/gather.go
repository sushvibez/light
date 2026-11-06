package cpu

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	reg := func(dt dtype.DType, k registry.Kernel) {
		registry.Register(registry.Key{Op: "gather", Dev: device.KindCPU, DType: dt}, k)
		registry.Register(registry.Key{Op: "scatter_add", Dev: device.KindCPU, DType: dt}, scatterAddKernelFor(dt))
	}
	reg(dtype.F32, gatherKernel[float32])
	reg(dtype.F64, gatherKernel[float64])
	reg(dtype.I32, gatherKernel[int32])
	reg(dtype.I64, gatherKernel[int64])
}

// gatherKernel: out[i0…] = src[i0, …, indices[i0…], …, iN-1]
// where `dim` is the axis being gathered, and indices has the same shape as out.
func gatherKernel[T any](ctx *registry.Context) error {
	src := ctx.Inputs[0]
	indices := ctx.Inputs[1]
	out := ctx.Output
	dim := ctx.Attrs["dim"].(int)

	if indices.DType() != dtype.I64 {
		return fmt.Errorf("gather: indices must be I64, got %v", indices.DType())
	}

	srcS := asSlice[T](src)
	idxS := asSlice[int64](indices)
	outS := asSlice[T](out)

	srcDims := src.Shape().Dims()
	outDims := out.Shape().Dims()
	rank := len(outDims)
	if rank != len(srcDims) {
		return fmt.Errorf("gather: src and out must have same rank")
	}

	// Compute strides.
	srcStr := make([]int, rank)
	outStr := make([]int, rank)
	s1 := 1
	s2 := 1
	for i := rank - 1; i >= 0; i-- {
		srcStr[i] = s1
		outStr[i] = s2
		s1 *= srcDims[i]
		s2 *= outDims[i]
	}

	n := out.NumElements()
	idx := make([]int, rank)
	for oi := range n {
		// Compute multi-index of oi in output.
		rem := oi
		for d := range rank {
			idx[d] = rem / outStr[d]
			rem %= outStr[d]
		}
		// Source index: same as out except along `dim`, use indices[oi].
		srcOff := 0
		for d := range rank {
			if d == dim {
				srcOff += int(idxS[oi]) * srcStr[d]
			} else {
				srcOff += idx[d] * srcStr[d]
			}
		}
		outS[oi] = srcS[srcOff]
	}
	return nil
}

// scatterAddKernelFor returns a kernel that computes:
//
//	out[…, indices[…, j, …], …] += src[…, j, …]
//
// along `dim`. `out` starts as a copy of another tensor (base) that the
// caller provides (ops.ScatterAdd handles copying before dispatch).
func scatterAddKernelFor(dt dtype.DType) registry.Kernel {
	switch dt {
	case dtype.F32:
		return scatterAddKernel[float32]
	case dtype.F64:
		return scatterAddKernel[float64]
	case dtype.I32:
		return scatterAddKernel[int32]
	case dtype.I64:
		return scatterAddKernel[int64]
	}
	return nil
}

func scatterAddKernel[T dtype.Numeric](ctx *registry.Context) error {
	src := ctx.Inputs[0]
	indices := ctx.Inputs[1]
	out := ctx.Output
	dim := ctx.Attrs["dim"].(int)

	if indices.DType() != dtype.I64 {
		return fmt.Errorf("scatter_add: indices must be I64")
	}

	srcS := asSlice[T](src)
	idxS := asSlice[int64](indices)
	outS := asSlice[T](out)

	srcDims := src.Shape().Dims()
	outDims := out.Shape().Dims()
	rank := len(srcDims)
	if rank != len(outDims) {
		return fmt.Errorf("scatter_add: rank mismatch")
	}

	srcStr := make([]int, rank)
	outStr := make([]int, rank)
	s1 := 1
	s2 := 1
	for i := rank - 1; i >= 0; i-- {
		srcStr[i] = s1
		outStr[i] = s2
		s1 *= srcDims[i]
		s2 *= outDims[i]
	}

	n := src.NumElements()
	idx := make([]int, rank)
	for si := range n {
		rem := si
		for d := range rank {
			idx[d] = rem / srcStr[d]
			rem %= srcStr[d]
		}
		outOff := 0
		for d := range rank {
			if d == dim {
				outOff += int(idxS[si]) * outStr[d]
			} else {
				outOff += idx[d] * outStr[d]
			}
		}
		outS[outOff] += srcS[si]
	}
	return nil
}

// ---- scatter (set), index_add, and BF16 / F16 for every index op (2026-10-28) ----

func init() {
	reg := func(op string, dt dtype.DType, k registry.Kernel) {
		registry.Register(registry.Key{Op: op, Dev: device.KindCPU, DType: dt}, k)
	}
	// gather / scatter only move bytes: any 2-, 4- or 8-byte dtype works.
	reg("gather", dtype.BF16, gatherKernel[uint16])
	reg("gather", dtype.F16, gatherKernel[uint16])
	for _, dt := range []dtype.DType{dtype.BF16, dtype.F16} {
		reg("scatter", dt, scatterKernel[uint16])
	}
	reg("scatter", dtype.F32, scatterKernel[float32])
	reg("scatter", dtype.F64, scatterKernel[float64])
	reg("scatter", dtype.I32, scatterKernel[int32])
	reg("scatter", dtype.I64, scatterKernel[int64])
	// Accumulating ops at 16-bit: add in F32, round once per update.
	bf := func(u uint16) float32 { return dtype.BFloat16(u).ToF32() }
	bfr := func(f float32) uint16 { return uint16(dtype.BF16FromF32(f)) }
	hf := func(u uint16) float32 { return dtype.Half(u).ToF32() }
	hfr := func(f float32) uint16 { return uint16(dtype.HalfFromF32(f)) }
	reg("scatter_add", dtype.BF16, scatterAddHalfKernel(bf, bfr, false))
	reg("scatter_add", dtype.F16, scatterAddHalfKernel(hf, hfr, false))
	reg("index_add", dtype.BF16, scatterAddHalfKernel(bf, bfr, true))
	reg("index_add", dtype.F16, scatterAddHalfKernel(hf, hfr, true))
	reg("index_add", dtype.F32, indexAddKernel[float32])
	reg("index_add", dtype.F64, indexAddKernel[float64])
	reg("index_add", dtype.I32, indexAddKernel[int32])
	reg("index_add", dtype.I64, indexAddKernel[int64])
}

// axisSplit returns (outer, axis, inner) for a contiguous tensor split at dim.
func axisSplit(dims []int, dim int) (outer, axis, inner int) {
	outer, inner = 1, 1
	for i := 0; i < dim; i++ {
		outer *= dims[i]
	}
	for i := dim + 1; i < len(dims); i++ {
		inner *= dims[i]
	}
	return outer, dims[dim], inner
}

// scatterKernel: out[o, idx[o, j, i], i] = src[o, j, i]; later writes win,
// as in PyTorch's scatter with duplicate indices on the CPU.
func scatterKernel[T any](ctx *registry.Context) error {
	src, indices, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	dim := ctx.Attrs["dim"].(int)
	srcS, idxS, outS := asSlice[T](src), asSlice[int64](indices), asSlice[T](out)
	outer, srcAxis, inner := axisSplit(src.Shape().Dims(), dim)
	outAxis := out.Dim(dim)
	for o := range outer {
		for j := range srcAxis {
			for i := range inner {
				si := (o*srcAxis+j)*inner + i
				outS[(o*outAxis+int(idxS[si]))*inner+i] = srcS[si]
			}
		}
	}
	return nil
}

// indexAddKernel: out[o, idx[j], i] += src[o, j, i] with a 1-D index.
func indexAddKernel[T dtype.Numeric](ctx *registry.Context) error {
	src, indices, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	dim := ctx.Attrs["dim"].(int)
	srcS, idxS, outS := asSlice[T](src), asSlice[int64](indices), asSlice[T](out)
	outer, srcAxis, inner := axisSplit(src.Shape().Dims(), dim)
	outAxis := out.Dim(dim)
	for o := range outer {
		for j := range srcAxis {
			row := (o*outAxis + int(idxS[j])) * inner
			srow := (o*srcAxis + j) * inner
			for i := range inner {
				outS[row+i] += srcS[srow+i]
			}
		}
	}
	return nil
}

// scatterAddHalfKernel covers scatter_add (indices shaped like src) and
// index_add (1-D indices) for 16-bit floats.
func scatterAddHalfKernel(toF32 func(uint16) float32, fromF32 func(float32) uint16, oneD bool) registry.Kernel {
	return func(ctx *registry.Context) error {
		src, indices, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
		dim := ctx.Attrs["dim"].(int)
		srcS, idxS, outS := asSlice[uint16](src), asSlice[int64](indices), asSlice[uint16](out)
		outer, srcAxis, inner := axisSplit(src.Shape().Dims(), dim)
		outAxis := out.Dim(dim)
		for o := range outer {
			for j := range srcAxis {
				for i := range inner {
					si := (o*srcAxis+j)*inner + i
					idx := idxS[j]
					if !oneD {
						idx = idxS[si]
					}
					oi := (o*outAxis+int(idx))*inner + i
					outS[oi] = fromF32(toF32(outS[oi]) + toF32(srcS[si]))
				}
			}
		}
		return nil
	}
}
