package cpu

import (
	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	// Keyed by the activation/output dtype (F32); the weight is always I8.
	registry.Register(registry.Key{Op: "matmul_int8", Dev: device.KindCPU, DType: dtype.F32}, matmulInt8Kernel)
}

// matmulInt8Kernel is the weight-only INT8 reference:
//
//	out[m,n] = scales[n] * sum_k x[m,k] * float(wq[k,n])
//
//	Inputs[0]: x      F32 (M, K) contiguous
//	Inputs[1]: wq     I8  (K, N) contiguous
//	Inputs[2]: scales F32 (N,)
//	Output:           F32 (M, N)
func matmulInt8Kernel(ctx *registry.Context) error {
	x, wq, scales, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	M, K := x.Dim(0), x.Dim(1)
	N := wq.Dim(1)
	xS := asSlice[float32](x)
	wS := asSlice[int8](wq)
	sS := asSlice[float32](scales)
	oS := asSlice[float32](out)
	row := make([]float32, N)
	for m := range M {
		for n := range row {
			row[n] = 0
		}
		for k := range K {
			a := xS[m*K+k]
			wRow := wS[k*N : k*N+N]
			for n, w := range wRow {
				row[n] += a * float32(w)
			}
		}
		for n := range N {
			oS[m*N+n] = row[n] * sS[n]
		}
	}
	return nil
}
