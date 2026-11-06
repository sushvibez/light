//go:build !nocuda && cgo

package cuda

import (
	"fmt"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
	"light/tensor"
)

var sortSuffix = map[dtype.DType]string{
	dtype.F32: "f32", dtype.F64: "f64", dtype.I32: "i32", dtype.I64: "i64",
}

func init() {
	for dt, sfx := range sortSuffix {
		registry.Register(registry.Key{Op: "sort", Dev: device.KindCUDA, DType: dt}, sortLauncher(sfx))
	}
}

// sortLauncher sorts each row of Inputs[0] along the last axis into Output
// (values) and Inputs[1] (I64 indices); Attrs["descending"]. See sort.cu.
func sortLauncher(sfx string) registry.Kernel {
	return func(ctx *registry.Context) error {
		in, idx, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
		desc := int32(0)
		if ctx.Attrs["descending"].(bool) {
			desc = 1
		}
		cols := int32(in.Dim(-1))
		if cols == 0 {
			return nil
		}
		rows := int32(in.NumElements() / int(cols))
		n2 := int32(2)
		for n2 < cols {
			n2 <<= 1
		}
		if err := cudart.SetDevice(out.Device().Ordinal()); err != nil {
			return err
		}
		inPtr, outPtr, idxPtr := in.Storage().CUDAPtr(), out.Storage().CUDAPtr(), idx.Storage().CUDAPtr()

		if n2 <= 1024 {
			fn, err := GetKernel("sort_shared_" + sfx)
			if err != nil {
				return err
			}
			args := []unsafe.Pointer{
				unsafe.Pointer(&inPtr), unsafe.Pointer(&outPtr), unsafe.Pointer(&idxPtr),
				unsafe.Pointer(&cols), unsafe.Pointer(&n2), unsafe.Pointer(&desc),
			}
			if err := cudart.LaunchKernel(fn, uint32(rows), 1, 1, uint32(n2), 1, 1, 0, 0, args); err != nil {
				return err
			}
			return nil
		}

		// Global-memory path: padded scratch buffers.
		vals, err := tensor.Zeros(in.DType(), []int{int(rows) * int(n2)}, out.Device())
		if err != nil {
			return err
		}
		pidx, err := tensor.Zeros(dtype.I64, []int{int(rows) * int(n2)}, out.Device())
		if err != nil {
			return err
		}
		vPtr, pPtr := vals.Storage().CUDAPtr(), pidx.Storage().CUDAPtr()
		total := uint64(rows) * uint64(n2)

		initFn, err := GetKernel("sort_init_" + sfx)
		if err != nil {
			return err
		}
		if err := Launch1D(initFn, total, []unsafe.Pointer{
			unsafe.Pointer(&inPtr), unsafe.Pointer(&vPtr), unsafe.Pointer(&pPtr),
			unsafe.Pointer(&rows), unsafe.Pointer(&cols), unsafe.Pointer(&n2), unsafe.Pointer(&desc),
		}); err != nil {
			return err
		}
		stepFn, err := GetKernel("sort_step_" + sfx)
		if err != nil {
			return err
		}
		for k := int32(2); k <= n2; k <<= 1 {
			for j := k >> 1; j > 0; j >>= 1 {
				kk, jj := k, j
				if err := Launch1D(stepFn, total, []unsafe.Pointer{
					unsafe.Pointer(&vPtr), unsafe.Pointer(&pPtr), unsafe.Pointer(&rows), unsafe.Pointer(&n2),
					unsafe.Pointer(&jj), unsafe.Pointer(&kk), unsafe.Pointer(&desc),
				}); err != nil {
					return fmt.Errorf("sort step k=%d j=%d: %w", k, j, err)
				}
			}
		}
		gatherFn, err := GetKernel("sort_gather_" + sfx)
		if err != nil {
			return err
		}
		if err := Launch1D(gatherFn, uint64(rows)*uint64(cols), []unsafe.Pointer{
			unsafe.Pointer(&vPtr), unsafe.Pointer(&pPtr), unsafe.Pointer(&outPtr), unsafe.Pointer(&idxPtr),
			unsafe.Pointer(&rows), unsafe.Pointer(&cols), unsafe.Pointer(&n2),
		}); err != nil {
			return err
		}
		return nil
	}
}
