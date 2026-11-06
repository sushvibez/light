package cpu

import (
	"math"

	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "rope_scaled", Dev: device.KindCPU, DType: dtype.F32}, ropeScaledCPU[float32])
	registry.Register(registry.Key{Op: "rope_scaled", Dev: device.KindCPU, DType: dtype.F64}, ropeScaledCPU[float64])
}

// ropeScaledCPU is RoPE with an explicit inverse-frequency table:
// theta(pos, pair i) = pos * invFreq[i]. Inputs = [x, positions I64,
// invFreq F32 (headDim/2)]. Same row/position broadcast contract as rope.
func ropeScaledCPU[T float32 | float64](ctx *registry.Context) error {
	x, positions, invFreq, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	xS := asSlice[T](x)
	posS := asSlice[int64](positions)
	freq := asSlice[float32](invFreq)
	outS := asSlice[T](out)
	headDim := x.Dim(-1)
	seqLen := x.Dim(-2)
	batch := x.NumElements() / (seqLen * headDim)
	half := headDim / 2
	for b := range batch {
		for s := range seqLen {
			pos := float64(posS[s])
			rowBase := (b*seqLen + s) * headDim
			// Rotate-half pairing (dim i with dim i+half), same as rope.
			for i := 0; i < half; i++ {
				theta := pos * float64(freq[i])
				c := T(math.Cos(theta))
				sn := T(math.Sin(theta))
				x0 := xS[rowBase+i]
				x1 := xS[rowBase+i+half]
				outS[rowBase+i] = x0*c - x1*sn
				outS[rowBase+i+half] = x0*sn + x1*c
			}
		}
	}
	return nil
}
