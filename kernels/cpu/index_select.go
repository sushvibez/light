package cpu

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	reg := func(dt dtype.DType, k registry.Kernel) {
		registry.Register(registry.Key{Op: "index_select", Dev: device.KindCPU, DType: dt}, k)
	}
	reg(dtype.F32, indexSelectKernel[float32])
	reg(dtype.F64, indexSelectKernel[float64])
	reg(dtype.I32, indexSelectKernel[int32])
	reg(dtype.I64, indexSelectKernel[int64])
	// Row gathers that copy raw bytes: every GGUF block format (rows are
	// whole blocks, so a quantized embedding table can be looked up without
	// dequantizing the whole table) and the remaining fixed-width dtypes.
	for _, dt := range []dtype.DType{
		dtype.F16, dtype.BF16, dtype.F8E4M3, dtype.U8, dtype.I8, dtype.U16, dtype.I16, dtype.U32, dtype.U64,
		dtype.Q4_0, dtype.Q4_1, dtype.Q5_0, dtype.Q5_1, dtype.Q8_0, dtype.Q8_1,
		dtype.Q2_K, dtype.Q3_K, dtype.Q4_K, dtype.Q5_K, dtype.Q6_K, dtype.Q8_K,
	} {
		reg(dt, indexSelectBytesKernel)
	}
}

// indexSelectBytesKernel is indexSelectKernel on raw row bytes, for dtypes
// without a Go element type (block-quantized) or without a generic kernel.
func indexSelectBytesKernel(ctx *registry.Context) error {
	src, indices, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	if indices.DType() != dtype.I64 {
		return fmt.Errorf("index_select: indices must be I64, got %v", indices.DType())
	}
	if src.Offset() != 0 || out.Offset() != 0 {
		return fmt.Errorf("index_select: byte-row gather needs zero-offset tensors")
	}
	srcDims := src.Shape().Dims()
	numRows := srcDims[0]
	rowElems := 1
	for _, d := range srcDims[1:] {
		rowElems *= d
	}
	dt := src.DType()
	if dt.IsQuantized() && rowElems%dt.BlockElems() != 0 {
		return fmt.Errorf("index_select: %v row of %d elements is not whole blocks", dt, rowElems)
	}
	rowBytes := dt.ByteSize(rowElems)
	srcB, outB := src.Storage().CPUBytes(), out.Storage().CPUBytes()
	for i, idx := range asSlice[int64](indices) {
		if idx < 0 || int(idx) >= numRows {
			return fmt.Errorf("index_select: index %d out of range [0, %d)", idx, numRows)
		}
		copy(outB[i*rowBytes:(i+1)*rowBytes], srcB[int(idx)*rowBytes:(int(idx)+1)*rowBytes])
	}
	return nil
}

// indexSelectKernel selects rows along axis 0 by index. Input src is (N, …),
// indices is a 1-D I64 tensor of length K, output is (K, …) with the same
// trailing dims as src.
func indexSelectKernel[T dtype.Numeric](ctx *registry.Context) error {
	src := ctx.Inputs[0]
	indices := ctx.Inputs[1]
	out := ctx.Output
	if indices.DType() != dtype.I64 {
		return fmt.Errorf("index_select: indices must be I64, got %v", indices.DType())
	}
	srcS := asSlice[T](src)
	outS := asSlice[T](out)
	idxS := asSlice[int64](indices)

	srcDims := src.Shape().Dims()
	numRows := srcDims[0]
	rowSize := 1
	for _, d := range srcDims[1:] {
		rowSize *= d
	}

	for i, idx := range idxS {
		if idx < 0 || int(idx) >= numRows {
			return fmt.Errorf("index_select: index %d out of range [0, %d)", idx, numRows)
		}
		copy(outS[i*rowSize:(i+1)*rowSize], srcS[int(idx)*rowSize:(int(idx)+1)*rowSize])
	}
	return nil
}
