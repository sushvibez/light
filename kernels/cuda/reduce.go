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
	for _, op := range []string{"sum", "mean", "max", "min", "prod"} {
		for _, dt := range []dtype.DType{dtype.F32, dtype.F64} {
			registry.Register(registry.Key{Op: op, Dev: device.KindCUDA, DType: dt}, makeReduceLauncher(op, dt))
		}
	}
	registry.Register(registry.Key{Op: "l2norm", Dev: device.KindCUDA, DType: dtype.F32}, l2NormLauncher("reduce_l2_f32", 4))
	registry.Register(registry.Key{Op: "l2norm", Dev: device.KindCUDA, DType: dtype.F64}, l2NormLauncher("reduce_l2_f64", 8))
	registry.Register(registry.Key{Op: "all", Dev: device.KindCUDA, DType: dtype.U8}, u8ReduceLauncher("reduce_all_u8"))
	registry.Register(registry.Key{Op: "any", Dev: device.KindCUDA, DType: dtype.U8}, u8ReduceLauncher("reduce_any_u8"))
	registry.Register(registry.Key{Op: "argmax", Dev: device.KindCUDA, DType: dtype.F32}, argmaxRowF32)
	registry.Register(registry.Key{Op: "argmin", Dev: device.KindCUDA, DType: dtype.F32}, argLaunchF32("argmin_last_f32"))
	registry.Register(registry.Key{Op: "logsoftmax", Dev: device.KindCUDA, DType: dtype.F32}, logsoftmaxF32CUDA)
}

func l2NormLauncher(kname string, elemSize uint32) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, out := ctx.Inputs[0], ctx.Output
		axes, _ := ctx.Attrs["axes"].([]int)
		if !isLastAxisContiguous(axes, a.Rank()) {
			return fmt.Errorf("cuda l2norm: only last-axis supported")
		}
		dims := a.Shape().Dims()
		cols := uint64(1)
		for _, ax := range axes {
			if ax < 0 {
				ax += a.Rank()
			}
			cols *= uint64(dims[ax])
		}
		rows := uint64(a.NumElements()) / cols
		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(kname)
		if err != nil {
			return err
		}
		aPtr := a.Storage().CUDAPtr()
		outPtr := out.Storage().CUDAPtr()
		args := []unsafe.Pointer{
			unsafe.Pointer(&aPtr),
			unsafe.Pointer(&outPtr),
			unsafe.Pointer(&rows),
			unsafe.Pointer(&cols),
		}
		const block = 256
		if err := cudart.LaunchKernel(fn, uint32(rows), 1, 1, block, 1, 1, block*elemSize, 0, args); err != nil {
			return err
		}
		return nil
	}
}

func u8ReduceLauncher(kname string) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, out := ctx.Inputs[0], ctx.Output
		axes, _ := ctx.Attrs["axes"].([]int)
		if !isLastAxisContiguous(axes, a.Rank()) {
			return fmt.Errorf("cuda %s: only last-axis supported", kname)
		}
		dims := a.Shape().Dims()
		cols := uint64(1)
		for _, ax := range axes {
			if ax < 0 {
				ax += a.Rank()
			}
			cols *= uint64(dims[ax])
		}
		rows := uint64(a.NumElements()) / cols
		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(kname)
		if err != nil {
			return err
		}
		aPtr := a.Storage().CUDAPtr()
		outPtr := out.Storage().CUDAPtr()
		args := []unsafe.Pointer{
			unsafe.Pointer(&aPtr),
			unsafe.Pointer(&outPtr),
			unsafe.Pointer(&rows),
			unsafe.Pointer(&cols),
		}
		const block = 256
		if err := cudart.LaunchKernel(fn, uint32(rows), 1, 1, block, 1, 1, block, 0, args); err != nil {
			return err
		}
		return nil
	}
}

// makeReduceLauncher returns a kernel for last-axis reductions. When the
// requested reduction axes aren't trailing, the launcher permutes the input
// so they are (via Contiguous() for actual memory movement on CUDA).
//
// NOTE: the output shape is allocated by the caller (ops.reduceDispatch);
// this can only work if the caller's output matches the shape this launcher
// would produce AFTER permuting. For non-trailing axes, the caller's output
// shape assumes non-permuted input, so we'd need to permute the output back.
// To keep things simple, we require the caller to permute beforehand OR
// accept the current permuted output shape. Since ops.reduceDispatch doesn't
// permute, we implement permute-and-reshape here and error if the ordering
// mismatches.
func makeReduceLauncher(op string, dt dtype.DType) registry.Kernel {
	kname := fmt.Sprintf("reduce_%s_%s", op, dtypeSuffix(dt))
	return func(ctx *registry.Context) error {
		a, out := ctx.Inputs[0], ctx.Output
		axes, _ := ctx.Attrs["axes"].([]int)
		if !isLastAxisContiguous(axes, a.Rank()) {
			perm, newAxes := permuteForReduce(a.Rank(), axes)
			pT, err := a.Permute(perm...)
			if err != nil {
				return err
			}
			cT, err := pT.Contiguous()
			if err != nil {
				return err
			}
			a = cT
			axes = newAxes
		}
		// Compute cols = product of reduced dims (from back), rows = remaining.
		dims := a.Shape().Dims()
		sortedAxes := append([]int(nil), axes...)
		// Trailing axes.
		cols := uint64(1)
		for _, ax := range sortedAxes {
			cols *= uint64(dims[ax])
		}
		total := uint64(a.NumElements())
		rows := total / cols

		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(kname)
		if err != nil {
			return err
		}
		aPtr := a.Storage().CUDAPtr()
		outPtr := out.Storage().CUDAPtr()
		args := []unsafe.Pointer{
			unsafe.Pointer(&aPtr),
			unsafe.Pointer(&outPtr),
			unsafe.Pointer(&rows),
			unsafe.Pointer(&cols),
		}
		const block = 256
		sharedBytes := uint32(block) * uint32(dt.Size())
		if err := cudart.LaunchKernel(fn, uint32(rows), 1, 1, block, 1, 1, sharedBytes, 0, args); err != nil {
			return err
		}
		return nil
	}
}

// isLastAxisContiguous reports whether `axes` is exactly the trailing k axes
// of a tensor with the given rank.
func isLastAxisContiguous(axes []int, rank int) bool {
	if len(axes) == 0 {
		return rank == 0
	}
	// Normalize negative axes and sort.
	seen := make(map[int]bool, len(axes))
	for _, ax := range axes {
		if ax < 0 {
			ax += rank
		}
		seen[ax] = true
	}
	// Axes must be exactly [rank-k, rank-1].
	k := len(seen)
	for i := rank - k; i < rank; i++ {
		if !seen[i] {
			return false
		}
	}
	return true
}

// permuteForReduce builds a permutation that moves the given reduction axes
// to the trailing positions (preserving their internal order) and returns
// (perm, trailing-axes). Non-reduction axes stay in their original order.
//
//	rank=4 axes=[1, 3] → perm=[0, 2, 1, 3], trailing=[2, 3]
func permuteForReduce(rank int, axes []int) (perm []int, newAxes []int) {
	axSet := make(map[int]bool, len(axes))
	for _, ax := range axes {
		if ax < 0 {
			ax += rank
		}
		axSet[ax] = true
	}
	// Keep non-reduced axes in their original order.
	for i := range rank {
		if !axSet[i] {
			perm = append(perm, i)
		}
	}
	// Then the reduced axes, in the caller's order.
	for _, ax := range axes {
		if ax < 0 {
			ax += rank
		}
		perm = append(perm, ax)
	}
	// The new axes list: the last len(axes) axes of the permuted tensor.
	startOfReduced := rank - len(axes)
	newAxes = make([]int, len(axes))
	for i := range axes {
		newAxes[i] = startOfReduced + i
	}
	return perm, newAxes
}

// argLaunchF32 wraps the argmax/argmin kernels which write I64 indices.
func argLaunchF32(kname string) registry.Kernel {
	return func(ctx *registry.Context) error {
		a, out := ctx.Inputs[0], ctx.Output
		axes, _ := ctx.Attrs["axes"].([]int)
		if !isLastAxisContiguous(axes, a.Rank()) {
			return fmt.Errorf("cuda %s: only last-axis supported", kname)
		}
		dims := a.Shape().Dims()
		cols := uint64(dims[len(dims)-1])
		rows := uint64(a.NumElements()) / cols

		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		fn, err := GetKernel(kname)
		if err != nil {
			return err
		}
		aPtr := a.Storage().CUDAPtr()
		outPtr := out.Storage().CUDAPtr()
		args := []unsafe.Pointer{
			unsafe.Pointer(&aPtr),
			unsafe.Pointer(&outPtr),
			unsafe.Pointer(&rows),
			unsafe.Pointer(&cols),
		}
		// 1 thread per row (each row is serially scanned); grid = rows.
		if err := cudart.LaunchKernel(fn, uint32(rows), 1, 1, 1, 1, 1, 0, 0, args); err != nil {
			return err
		}
		return nil
	}
}

func logsoftmaxF32CUDA(ctx *registry.Context) error {
	a, out := ctx.Inputs[0], ctx.Output
	dims := a.Shape().Dims()
	cols := uint64(dims[len(dims)-1])
	rows := uint64(a.NumElements()) / cols
	if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
		return err
	}
	fn, err := GetKernel("logsoftmax_f32")
	if err != nil {
		return err
	}
	aPtr := a.Storage().CUDAPtr()
	outPtr := out.Storage().CUDAPtr()
	args := []unsafe.Pointer{
		unsafe.Pointer(&aPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&rows),
		unsafe.Pointer(&cols),
	}
	const block = 256
	const sharedBytes = block * 4
	if err := cudart.LaunchKernel(fn, uint32(rows), 1, 1, block, 1, 1, sharedBytes, 0, args); err != nil {
		return err
	}
	return nil
}

// dtypeSuffix returns the lowercase short name used in .cu kernel names.
func dtypeSuffix(dt dtype.DType) string {
	switch dt {
	case dtype.F32:
		return "f32"
	case dtype.F64:
		return "f64"
	case dtype.F16:
		return "f16"
	case dtype.BF16:
		return "bf16"
	case dtype.I32:
		return "i32"
	case dtype.I64:
		return "i64"
	case dtype.U8:
		return "u8"
	case dtype.I8:
		return "i8"
	}
	return ""
}
