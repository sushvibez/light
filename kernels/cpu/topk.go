package cpu

import (
	"sort"

	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	reg := func(dt dtype.DType, k registry.Kernel) {
		registry.Register(registry.Key{Op: "topk", Dev: device.KindCPU, DType: dt}, k)
	}
	reg(dtype.F32, topkKernel[float32])
	reg(dtype.F64, topkKernel[float64])
	reg(dtype.I32, topkKernel[int32])
	reg(dtype.I64, topkKernel[int64])
}

// topkKernel returns the top-k values (and their indices) along the last axis.
// Outputs: ctx.Output (values, same dtype as input) and an extra I64 tensor
// passed in Inputs[1] (indices). Attrs["k"] = int, "largest" = bool.
//
// Inputs: [input]
// Outputs must be pre-allocated by ops.TopK: (..., k) values and (..., k) I64 indices.
// ops.TopK places the indices tensor at Attrs["idx_out"].
func topkKernel[T dtype.Numeric](ctx *registry.Context) error {
	a := ctx.Inputs[0]
	out := ctx.Output
	idxOut := ctx.Attrs["idx_out"].(*struct {
		Values  []int64
		Columns int
	})
	k := ctx.Attrs["k"].(int)
	largest := ctx.Attrs["largest"].(bool)

	aS := asSlice[T](a)
	outS := asSlice[T](out)

	dims := a.Shape().Dims()
	cols := dims[len(dims)-1]
	rows := a.NumElements() / cols

	type kv struct {
		v T
		i int64
	}
	pool := make([]kv, cols)
	for r := range rows {
		for j := range cols {
			pool[j] = kv{v: aS[r*cols+j], i: int64(j)}
		}
		sort.SliceStable(pool, func(i, j int) bool {
			if largest {
				return pool[i].v > pool[j].v
			}
			return pool[i].v < pool[j].v
		})
		for j := 0; j < k; j++ {
			outS[r*k+j] = pool[j].v
			idxOut.Values[r*idxOut.Columns+j] = pool[j].i
		}
	}
	return nil
}
