package cpu

import (
	"math"

	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "flash_attention", Dev: device.KindCPU, DType: dtype.F32}, flashAttnKernel[float32])
	registry.Register(registry.Key{Op: "flash_attention", Dev: device.KindCPU, DType: dtype.F64}, flashAttnKernel[float64])
}

// flashAttnKernel is the CPU reference for FlashAttention: per query row,
// one pass over the keys with an online (running max / running sum)
// softmax so the (S, S) score matrix is never materialised.
//
//	Inputs[0..2]: q, k, v contiguous (BH, S, D)
//	Output:       (BH, S, D)
//	Attrs:        "scale" float64, "causal" bool
func flashAttnKernel[T dtype.Numeric](ctx *registry.Context) error {
	q, k, v, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	scale := ctx.Attrs["scale"].(float64)
	causal := ctx.Attrs["causal"].(bool)
	BH, S, D := q.Dim(0), q.Dim(1), q.Dim(2)
	qS := asSlice[T](q)
	kS := asSlice[T](k)
	vS := asSlice[T](v)
	oS := asSlice[T](out)
	acc := make([]float64, D)
	for b := range BH {
		for i := range S {
			qBase := (b*S + i) * D
			m, l := math.Inf(-1), 0.0
			for x := range acc {
				acc[x] = 0
			}
			jMax := S
			if causal {
				jMax = i + 1
			}
			for j := range jMax {
				kBase := (b*S + j) * D
				var dot float64
				for x := range D {
					dot += float64(qS[qBase+x]) * float64(kS[kBase+x])
				}
				score := dot * scale
				mNew := math.Max(m, score)
				corr := math.Exp(m - mNew)
				p := math.Exp(score - mNew)
				l = l*corr + p
				vBase := (b*S + j) * D
				for x := range D {
					acc[x] = acc[x]*corr + p*float64(vS[vBase+x])
				}
				m = mNew
			}
			oBase := (b*S + i) * D
			inv := 0.0
			if l > 0 {
				inv = 1 / l
			}
			for x := range D {
				oS[oBase+x] = T(acc[x] * inv)
			}
		}
	}
	return nil
}
