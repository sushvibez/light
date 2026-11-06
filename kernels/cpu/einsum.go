package cpu

import (
	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "einsum", Dev: device.KindCPU, DType: dtype.F32}, einsumKernel[float32])
	registry.Register(registry.Key{Op: "einsum", Dev: device.KindCPU, DType: dtype.F64}, einsumKernel[float64])
}

// einsumKernel is the general (any number of operands, repeated indices
// allowed) reference contraction. ops.Einsum parses the notation and hands
// the kernel a resolved plan via Attrs:
//
//	"subs"     [][]int — per operand, the letter id at each axis
//	"out"      []int   — letter ids in output order (free indices)
//	"sum"      []int   — letter ids that are contracted
//	"sizes"    []int   — size of every letter id
//
// Each operand is contiguous. The loops run over every output position and,
// for each, over the full summation space, accumulating in float64.
func einsumKernel[T dtype.Numeric](ctx *registry.Context) error {
	subs := ctx.Attrs["subs"].([][]int)
	outIdx := ctx.Attrs["out"].([]int)
	sumIdx := ctx.Attrs["sum"].([]int)
	sizes := ctx.Attrs["sizes"].([]int)
	nOps := len(ctx.Inputs)

	data := make([][]T, nOps)
	for i, t := range ctx.Inputs {
		data[i] = asSlice[T](t)
	}
	outS := asSlice[T](ctx.Output)

	// Per-operand stride of each letter (sum of strides of every axis the
	// letter occupies — handles repeated letters like "ii" for the diagonal).
	nLetters := len(sizes)
	letterStride := make([][]int, nOps)
	for op, sub := range subs {
		dims := ctx.Inputs[op].Shape().Dims()
		ls := make([]int, nLetters)
		s := 1
		for ax := len(sub) - 1; ax >= 0; ax-- {
			ls[sub[ax]] += s
			s *= dims[ax]
		}
		letterStride[op] = ls
	}

	coords := make([]int, nLetters)
	base := make([]int, nOps)
	outN := len(outS)
	sumTotal := 1
	for _, l := range sumIdx {
		sumTotal *= sizes[l]
	}
	for flat := range outN {
		// Decode output coordinates (row-major over outIdx).
		rem := flat
		for i := len(outIdx) - 1; i >= 0; i-- {
			l := outIdx[i]
			coords[l] = rem % sizes[l]
			rem /= sizes[l]
		}
		for op := range nOps {
			b := 0
			for _, l := range outIdx {
				b += coords[l] * letterStride[op][l]
			}
			base[op] = b
		}
		var acc float64
		for sf := range sumTotal {
			rem := sf
			for i := len(sumIdx) - 1; i >= 0; i-- {
				l := sumIdx[i]
				coords[l] = rem % sizes[l]
				rem /= sizes[l]
			}
			prod := 1.0
			for op := range nOps {
				off := base[op]
				for _, l := range sumIdx {
					off += coords[l] * letterStride[op][l]
				}
				prod *= float64(data[op][off])
			}
			acc += prod
		}
		outS[flat] = T(acc)
	}
	return nil
}
