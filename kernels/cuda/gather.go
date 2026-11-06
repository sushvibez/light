//go:build !nocuda && cgo

package cuda

import (
	"fmt"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "gather", Dev: device.KindCUDA, DType: dtype.F32}, gatherF32Launcher)
	registry.Register(registry.Key{Op: "scatter_add", Dev: device.KindCUDA, DType: dtype.F32}, scatterAddF32Launcher)
	for _, dt := range []dtype.DType{
		dtype.F16, dtype.F8E4M3, dtype.U8, dtype.I8, dtype.U16, dtype.I16, dtype.U32, dtype.U64,
		dtype.Q4_0, dtype.Q4_1, dtype.Q5_0, dtype.Q5_1, dtype.Q8_0, dtype.Q8_1,
		dtype.Q2_K, dtype.Q3_K, dtype.Q4_K, dtype.Q5_K, dtype.Q6_K, dtype.Q8_K,
	} {
		registry.Register(registry.Key{Op: "index_select", Dev: device.KindCUDA, DType: dt}, indexSelectBytesLauncher)
	}
}

// indexSelectBytesLauncher gathers whole rows as bytes; see the CPU twin in
// kernels/cpu/index_select.go.
func indexSelectBytesLauncher(ctx *registry.Context) error {
	src, indices, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	if indices.DType() != dtype.I64 {
		return fmt.Errorf("cuda index_select: indices must be I64")
	}
	if src.Offset() != 0 || out.Offset() != 0 {
		return fmt.Errorf("cuda index_select: byte-row gather needs zero-offset tensors")
	}
	srcDims := src.Shape().Dims()
	rowElems := 1
	for _, d := range srcDims[1:] {
		rowElems *= d
	}
	dt := src.DType()
	if dt.IsQuantized() && rowElems%dt.BlockElems() != 0 {
		return fmt.Errorf("cuda index_select: %v row of %d elements is not whole blocks", dt, rowElems)
	}
	rowBytes := uint64(dt.ByteSize(rowElems))
	total := rowBytes * uint64(indices.NumElements())
	if total == 0 {
		return nil
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("index_select_bytes")
	if err != nil {
		return err
	}
	srcPtr, idxPtr, outPtr := src.Storage().CUDAPtr(), indices.Storage().CUDAPtr(), out.Storage().CUDAPtr()
	args := []unsafe.Pointer{unsafe.Pointer(&srcPtr), unsafe.Pointer(&idxPtr), unsafe.Pointer(&outPtr), unsafe.Pointer(&rowBytes), unsafe.Pointer(&total)}
	if err := Launch1D(fn, total, args); err != nil {
		return err
	}
	return nil
}

func scatterAddF32Launcher(ctx *registry.Context) error {
	src := ctx.Inputs[0]
	indices := ctx.Inputs[1]
	out := ctx.Output
	dim := ctx.Attrs["dim"].(int)

	if indices.DType() != dtype.I64 {
		return fmt.Errorf("cuda scatter_add: indices must be I64")
	}
	srcDims := src.Shape().Dims()
	outDims := out.Shape().Dims()
	if len(srcDims) != len(outDims) {
		return fmt.Errorf("cuda scatter_add: rank mismatch")
	}
	outer := 1
	for i := 0; i < dim; i++ {
		outer *= srcDims[i]
	}
	inner := 1
	for i := dim + 1; i < len(srcDims); i++ {
		inner *= srcDims[i]
	}
	srcAxis := uint64(srcDims[dim])
	outAxis := uint64(outDims[dim])
	outerU := uint64(outer)
	innerU := uint64(inner)

	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("scatter_add_f32")
	if err != nil {
		return err
	}
	srcPtr := src.Storage().CUDAPtr()
	idxPtr := indices.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&srcPtr),
		unsafe.Pointer(&idxPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&outerU),
		unsafe.Pointer(&srcAxis),
		unsafe.Pointer(&outAxis),
		unsafe.Pointer(&innerU),
	}
	n := outerU * srcAxis * innerU
	if err := Launch1D(fn, n, args); err != nil {
		return err
	}
	return nil
}

func gatherF32Launcher(ctx *registry.Context) error {
	src := ctx.Inputs[0]
	indices := ctx.Inputs[1]
	out := ctx.Output
	dim := ctx.Attrs["dim"].(int)

	if indices.DType() != dtype.I64 {
		return fmt.Errorf("cuda gather: indices must be I64")
	}
	srcDims := src.Shape().Dims()
	outDims := out.Shape().Dims()
	if len(srcDims) != len(outDims) {
		return fmt.Errorf("cuda gather: rank mismatch")
	}
	// outer = product of out dims before `dim`; inner = product after.
	outer := 1
	for i := 0; i < dim; i++ {
		outer *= outDims[i]
	}
	inner := 1
	for i := dim + 1; i < len(outDims); i++ {
		inner *= outDims[i]
	}
	outAxis := uint64(outDims[dim])
	srcAxis := uint64(srcDims[dim])
	outerU := uint64(outer)
	innerU := uint64(inner)

	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("gather_f32")
	if err != nil {
		return err
	}
	srcPtr := src.Storage().CUDAPtr()
	idxPtr := indices.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&srcPtr),
		unsafe.Pointer(&idxPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&outerU),
		unsafe.Pointer(&outAxis),
		unsafe.Pointer(&srcAxis),
		unsafe.Pointer(&innerU),
	}
	n := outerU * outAxis * innerU
	if err := Launch1D(fn, n, args); err != nil {
		return err
	}
	return nil
}

// ---- scatter / index_add and the wider dtype set (src/index_ops.cu) ----

func init() {
	sfx := map[dtype.DType]string{dtype.F32: "f32", dtype.F64: "f64", dtype.I32: "i32", dtype.I64: "i64", dtype.BF16: "bf16", dtype.F16: "f16"}
	for dt, s := range sfx {
		if dt != dtype.F32 { // F32 gather / scatter_add are registered above
			registry.Register(registry.Key{Op: "gather", Dev: device.KindCUDA, DType: dt}, indexOpLauncher("gather_"+s, true))
			registry.Register(registry.Key{Op: "scatter_add", Dev: device.KindCUDA, DType: dt}, indexOpLauncher("scatter_add_"+s, false))
		}
		registry.Register(registry.Key{Op: "scatter", Dev: device.KindCUDA, DType: dt}, indexOpLauncher("scatter_"+s, false))
		registry.Register(registry.Key{Op: "index_add", Dev: device.KindCUDA, DType: dt}, indexOpLauncher("index_add_"+s, false))
	}
}

// indexOpLauncher runs one of the (outer, axis, inner) index kernels.
// Inputs = [src, indices]; the thread count follows the source for scatter
// style ops and the output for gather (gatherShapedByOut).
func indexOpLauncher(name string, gatherShapedByOut bool) registry.Kernel {
	return func(ctx *registry.Context) error {
		src, indices, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
		dim := ctx.Attrs["dim"].(int)
		if indices.DType() != dtype.I64 {
			return fmt.Errorf("cuda %s: indices must be I64", name)
		}
		srcDims, outDims := src.Shape().Dims(), out.Shape().Dims()
		if len(srcDims) != len(outDims) {
			return fmt.Errorf("cuda %s: rank mismatch", name)
		}
		outer, inner := uint64(1), uint64(1)
		for i := 0; i < dim; i++ {
			outer *= uint64(srcDims[i])
		}
		for i := dim + 1; i < len(srcDims); i++ {
			inner *= uint64(srcDims[i])
		}
		srcAxis, outAxis := uint64(srcDims[dim]), uint64(outDims[dim])
		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(name)
		if err != nil {
			return err
		}
		srcPtr, idxPtr, outPtr := src.Storage().CUDAPtr(), indices.Storage().CUDAPtr(), out.Storage().CUDAPtr()
		// gather kernels take (outer, outAxis, srcAxis, inner); scatter-style
		// kernels take (outer, srcAxis, outAxis, inner). Both iterate over the
		// tensor whose axis comes first.
		first, second := srcAxis, outAxis
		if gatherShapedByOut {
			first, second = outAxis, srcAxis
		}
		args := []unsafe.Pointer{
			unsafe.Pointer(&srcPtr), unsafe.Pointer(&idxPtr), unsafe.Pointer(&outPtr),
			unsafe.Pointer(&outer), unsafe.Pointer(&first), unsafe.Pointer(&second), unsafe.Pointer(&inner),
		}
		if err := Launch1D(fn, outer*first*inner, args); err != nil {
			return err
		}
		return nil
	}
}
