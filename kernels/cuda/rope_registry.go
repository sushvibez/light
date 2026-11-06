//go:build !nocuda

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
	registry.Register(registry.Key{Op: "rope", Dev: device.KindCUDA, DType: dtype.F32}, ropeF32Launcher)
	registry.Register(registry.Key{Op: "rope", Dev: device.KindCUDA, DType: dtype.BF16}, ropeBF16Launcher)
	for _, dt := range []dtype.DType{dtype.F32, dtype.F64, dtype.BF16} {
		sfx := map[dtype.DType]string{dtype.F32: "f32", dtype.F64: "f64", dtype.BF16: "bf16"}[dt]
		registry.Register(registry.Key{Op: "rope_i", Dev: device.KindCUDA, DType: dt}, ropeILauncher("rope_i_"+sfx, dt))
	}
	registry.Register(registry.Key{Op: "dropout", Dev: device.KindCUDA, DType: dtype.F32}, dropoutF32Launcher)
	registry.Register(registry.Key{Op: "tril", Dev: device.KindCUDA, DType: dtype.F32}, triangleLauncher("tril_f32"))
	registry.Register(registry.Key{Op: "triu", Dev: device.KindCUDA, DType: dtype.F32}, triangleLauncher("triu_f32"))
}

func ropeBF16Launcher(ctx *registry.Context) error {
	x := ctx.Inputs[0]
	positions := ctx.Inputs[1]
	out := ctx.Output
	base := float32(ctx.Attrs["base"].(float64))
	if positions.DType() != dtype.I64 {
		return fmt.Errorf("cuda rope_bf16: positions must be I64")
	}
	headDim := uint64(x.Dim(-1))
	seqLen := uint64(x.Dim(-2))
	rows := uint64(x.NumElements()) / headDim
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("rope_bf16")
	if err != nil {
		return err
	}
	xPtr := x.Storage().CUDAPtr()
	posPtr := positions.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&xPtr),
		unsafe.Pointer(&posPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&rows),
		unsafe.Pointer(&seqLen),
		unsafe.Pointer(&headDim),
		unsafe.Pointer(&base),
	}
	halfDim := headDim / 2
	if err := Launch1D(fn, rows*halfDim, args); err != nil {
		return err
	}
	return nil
}

func ropeF32Launcher(ctx *registry.Context) error {
	x := ctx.Inputs[0]
	positions := ctx.Inputs[1]
	out := ctx.Output
	base := float32(ctx.Attrs["base"].(float64))
	if positions.DType() != dtype.I64 {
		return fmt.Errorf("cuda rope: positions must be I64")
	}
	headDim := uint64(x.Dim(-1))
	seqLen := uint64(x.Dim(-2))
	rows := uint64(x.NumElements()) / headDim
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("rope_f32")
	if err != nil {
		return err
	}
	xPtr := x.Storage().CUDAPtr()
	posPtr := positions.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&xPtr),
		unsafe.Pointer(&posPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&rows),
		unsafe.Pointer(&seqLen),
		unsafe.Pointer(&headDim),
		unsafe.Pointer(&base),
	}
	halfDim := headDim / 2
	if err := Launch1D(fn, rows*halfDim, args); err != nil {
		return err
	}
	return nil
}

func dropoutF32Launcher(ctx *registry.Context) error {
	x, out := ctx.Inputs[0], ctx.Output
	p := float32(ctx.Attrs["p"].(float64))
	scale := float32(ctx.Attrs["scale"].(float64))
	seed := ctx.Attrs["seed"].(uint64)
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("dropout_f32")
	if err != nil {
		return err
	}
	xPtr := x.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	n := uint64(x.NumElements())
	args := []unsafe.Pointer{
		unsafe.Pointer(&xPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&n),
		unsafe.Pointer(&p),
		unsafe.Pointer(&scale),
		unsafe.Pointer(&seed),
	}
	if err := Launch1D(fn, n, args); err != nil {
		return err
	}
	return nil
}

func triangleLauncher(kname string) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, out := ctx.Inputs[0], ctx.Output
		k := int32(ctx.Attrs["k"].(int))
		// Copy input to output first (tril/triu zero the opposite triangle in-place on `out`).
		elemBytes := uint64(out.NumElements()) * uint64(out.DType().Size())
		if err := cudart.MemcpyDtoD(out.Storage().CUDAPtr(), a.Storage().CUDAPtr(), uintptr(elemBytes)); err != nil {
			return err
		}
		rank := out.Rank()
		dims := out.Shape().Dims()
		rows := uint64(dims[rank-2])
		cols := uint64(dims[rank-1])
		batch := uint64(out.NumElements()) / (rows * cols)
		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(kname)
		if err != nil {
			return err
		}
		dataPtr := out.Storage().CUDAPtr()
		args := []unsafe.Pointer{
			unsafe.Pointer(&dataPtr),
			unsafe.Pointer(&batch),
			unsafe.Pointer(&rows),
			unsafe.Pointer(&cols),
			unsafe.Pointer(&k),
		}
		n := batch * rows * cols
		if err := Launch1D(fn, n, args); err != nil {
			return err
		}
		return nil
	}
}

// ropeILauncher mirrors ropeF32Launcher for the interleaved-pair kernels.
func ropeILauncher(name string, dt dtype.DType) registry.Kernel {
	return func(ctx *registry.Context) error {
		x, positions, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
		base := ctx.Attrs["base"].(float64)
		if positions.DType() != dtype.I64 {
			return fmt.Errorf("cuda %s: positions must be I64", name)
		}
		headDim := uint64(x.Dim(-1))
		seqLen := uint64(x.Dim(-2))
		rows := uint64(x.NumElements()) / headDim
		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(name)
		if err != nil {
			return err
		}
		xPtr := x.Storage().CUDAPtr()
		posPtr := positions.Storage().CUDAPtr()
		outPtr := out.Storage().CUDAPtr()
		base32 := float32(base)
		args := []unsafe.Pointer{
			unsafe.Pointer(&xPtr),
			unsafe.Pointer(&posPtr),
			unsafe.Pointer(&outPtr),
			unsafe.Pointer(&rows),
			unsafe.Pointer(&seqLen),
			unsafe.Pointer(&headDim),
		}
		if dt == dtype.F64 {
			args = append(args, unsafe.Pointer(&base))
		} else {
			args = append(args, unsafe.Pointer(&base32))
		}
		if err := Launch1D(fn, rows*(headDim/2), args); err != nil {
			return err
		}
		return nil
	}
}
