package cpu

import (
	"math"

	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "logsoftmax", Dev: device.KindCPU, DType: dtype.F32}, logSoftmaxKernel[float32])
	registry.Register(registry.Key{Op: "logsoftmax", Dev: device.KindCPU, DType: dtype.F64}, logSoftmaxKernel[float64])
}

// logSoftmaxKernel computes log_softmax along the last axis. Numerically
// stable: subtract the row max, exp, log-sum-exp, subtract.
//
//	ls(x)_i = (x_i - max) - log( sum_j exp(x_j - max) )
func logSoftmaxKernel[T dtype.Numeric](ctx *registry.Context) error {
	a, out := ctx.Inputs[0], ctx.Output
	aS := asSlice[T](a)
	oS := asSlice[T](out)

	cols := a.Dim(-1)
	rows := a.NumElements() / cols
	for r := range rows {
		rowA := aS[r*cols : (r+1)*cols]
		rowO := oS[r*cols : (r+1)*cols]
		maxV := rowA[0]
		for _, v := range rowA[1:] {
			if v > maxV {
				maxV = v
			}
		}
		var sumExp T
		for _, v := range rowA {
			sumExp += T(math.Exp(float64(v - maxV)))
		}
		lse := maxV + T(math.Log(float64(sumExp)))
		for i, v := range rowA {
			rowO[i] = v - lse
		}
	}
	return nil
}
