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
	registry.Register(registry.Key{Op: "rmsnorm", Dev: device.KindCUDA, DType: dtype.F32}, rmsnormF32CUDA)
	registry.Register(registry.Key{Op: "rmsnorm", Dev: device.KindCUDA, DType: dtype.BF16}, rmsnormBF16CUDA)
	registry.Register(registry.Key{Op: "layernorm", Dev: device.KindCUDA, DType: dtype.F32}, layernormF32CUDA)
	registry.Register(registry.Key{Op: "layernorm", Dev: device.KindCUDA, DType: dtype.BF16}, layernormBF16CUDA)
	registry.Register(registry.Key{Op: "index_select", Dev: device.KindCUDA, DType: dtype.F32}, indexSelectF32CUDA)
	registry.Register(registry.Key{Op: "index_select", Dev: device.KindCUDA, DType: dtype.BF16}, indexSelectBF16CUDA)
}

func rmsnormF32CUDA(ctx *registry.Context) error {
	x, w, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	eps, _ := ctx.Attrs["eps"].(float64)
	epsF := float32(eps)

	cols := uint64(x.Dim(-1))
	rows := uint64(x.NumElements()) / cols

	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("rmsnorm_f32")
	if err != nil {
		return err
	}
	xPtr := x.Storage().CUDAPtr()
	wPtr := w.Storage().CUDAPtr()
	oPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&xPtr),
		unsafe.Pointer(&wPtr),
		unsafe.Pointer(&oPtr),
		unsafe.Pointer(&rows),
		unsafe.Pointer(&cols),
		unsafe.Pointer(&epsF),
	}
	const block = 256
	if err := cudart.LaunchKernel(fn, uint32(rows), 1, 1, block, 1, 1, block*4, 0, args); err != nil {
		return err
	}
	return nil
}

func layernormF32CUDA(ctx *registry.Context) error {
	x, g, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	eps, _ := ctx.Attrs["eps"].(float64)
	epsF := float32(eps)

	cols := uint64(x.Dim(-1))
	rows := uint64(x.NumElements()) / cols

	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("layernorm_f32")
	if err != nil {
		return err
	}
	xPtr := x.Storage().CUDAPtr()
	gPtr := g.Storage().CUDAPtr()
	bPtr := b.Storage().CUDAPtr()
	oPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&xPtr),
		unsafe.Pointer(&gPtr),
		unsafe.Pointer(&bPtr),
		unsafe.Pointer(&oPtr),
		unsafe.Pointer(&rows),
		unsafe.Pointer(&cols),
		unsafe.Pointer(&epsF),
	}
	const block = 256
	if err := cudart.LaunchKernel(fn, uint32(rows), 1, 1, block, 1, 1, block*4, 0, args); err != nil {
		return err
	}
	return nil
}

// layernormBF16CUDA: BF16 input/output, BF16 gamma/beta. Compute reductions
// in F32 for stability. Weight dtype matches input dtype — caller is
// expected to land everything at one dtype at load time.
func layernormBF16CUDA(ctx *registry.Context) error {
	x, g, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	eps, _ := ctx.Attrs["eps"].(float64)
	epsF := float32(eps)
	if g.DType() != dtype.BF16 || b.DType() != dtype.BF16 {
		return fmt.Errorf("cuda layernorm_bf16: gamma/beta must be BF16 (got %v / %v)", g.DType(), b.DType())
	}
	cols := uint64(x.Dim(-1))
	rows := uint64(x.NumElements()) / cols
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("layernorm_bf16")
	if err != nil {
		return err
	}
	xPtr := x.Storage().CUDAPtr()
	gPtr := g.Storage().CUDAPtr()
	bPtr := b.Storage().CUDAPtr()
	oPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&xPtr), unsafe.Pointer(&gPtr), unsafe.Pointer(&bPtr), unsafe.Pointer(&oPtr),
		unsafe.Pointer(&rows), unsafe.Pointer(&cols), unsafe.Pointer(&epsF),
	}
	const block = 256
	if err := cudart.LaunchKernel(fn, uint32(rows), 1, 1, block, 1, 1, 2*block*4, 0, args); err != nil {
		return err
	}
	return nil
}

// rmsnormBF16CUDA: BF16 input/output, BF16 weight. Compute reductions in
// F32 for stability and store BF16. Weight dtype matches input dtype —
// caller casts at load, not in the hot path.
func rmsnormBF16CUDA(ctx *registry.Context) error {
	x, w, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	eps, _ := ctx.Attrs["eps"].(float64)
	epsF := float32(eps)
	if w.DType() != dtype.BF16 {
		return fmt.Errorf("cuda rmsnorm_bf16: weight must be BF16 (got %v)", w.DType())
	}
	cols := uint64(x.Dim(-1))
	rows := uint64(x.NumElements()) / cols
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("rmsnorm_bf16")
	if err != nil {
		return err
	}
	xPtr := x.Storage().CUDAPtr()
	wPtr := w.Storage().CUDAPtr()
	oPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&xPtr),
		unsafe.Pointer(&wPtr),
		unsafe.Pointer(&oPtr),
		unsafe.Pointer(&rows),
		unsafe.Pointer(&cols),
		unsafe.Pointer(&epsF),
	}
	const block = 256
	if err := cudart.LaunchKernel(fn, uint32(rows), 1, 1, block, 1, 1, block*4, 0, args); err != nil {
		return err
	}
	return nil
}

func indexSelectBF16CUDA(ctx *registry.Context) error {
	src := ctx.Inputs[0]
	indices := ctx.Inputs[1]
	out := ctx.Output
	if indices.DType() != dtype.I64 {
		return fmt.Errorf("cuda index_select_bf16: indices must be I64, got %v", indices.DType())
	}
	srcDims := src.Shape().Dims()
	outRows := uint64(indices.Dim(0))
	rowSize := uint64(1)
	for _, d := range srcDims[1:] {
		rowSize *= uint64(d)
	}
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("index_select_dim0_bf16")
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
		unsafe.Pointer(&outRows),
		unsafe.Pointer(&rowSize),
	}
	n := outRows * rowSize
	if err := Launch1D(fn, n, args); err != nil {
		return err
	}
	return nil
}

func indexSelectF32CUDA(ctx *registry.Context) error {
	src := ctx.Inputs[0]
	indices := ctx.Inputs[1]
	out := ctx.Output
	if indices.DType() != dtype.I64 {
		return fmt.Errorf("cuda index_select: indices must be I64, got %v", indices.DType())
	}
	srcDims := src.Shape().Dims()
	outRows := uint64(indices.Dim(0))
	rowSize := uint64(1)
	for _, d := range srcDims[1:] {
		rowSize *= uint64(d)
	}

	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("index_select_dim0_f32")
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
		unsafe.Pointer(&outRows),
		unsafe.Pointer(&rowSize),
	}
	n := outRows * rowSize
	if err := Launch1D(fn, n, args); err != nil {
		return err
	}
	return nil
}
