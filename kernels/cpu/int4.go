package cpu

import (
	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	for _, dt := range []dtype.DType{dtype.F32, dtype.BF16} {
		registry.Register(registry.Key{Op: "int4_dequant", Dev: device.KindCPU, DType: dt}, int4DequantCPU)
	}
}

// int4DequantCPU expands canonical 4-bit weights (quant.Int4): codes U32
// (N, K/8), scales / zeros F32 (N, K/G) → (N, K) F32 or BF16, the product
// formed in F32 and rounded once.
func int4DequantCPU(ctx *registry.Context) error {
	codes := asSlice[uint32](ctx.Inputs[0])
	s := asSlice[float32](ctx.Inputs[1])
	z := asSlice[float32](ctx.Inputs[2])
	out := ctx.Output
	N, K, G := ctx.Inputs[0].Dim(0), ctx.Inputs[0].Dim(1)*8, ctx.Attrs["group"].(int)
	ng := K / G
	var f32 []float32
	var b16 []uint16
	off, _ := ctx.Attrs["offset"].(int) // write into a slot of a larger buffer
	if out.DType() == dtype.F32 {
		f32 = asSlice[float32](out)[off:]
	} else {
		b16 = asSlice[uint16](out)[off:]
	}
	for n := range N {
		for k := range K {
			g := n*ng + k/G
			v := s[g] * (float32((codes[n*(K/8)+k/8]>>(4*(k%8)))&0xF) - z[g])
			if f32 != nil {
				f32[n*K+k] = v
			} else {
				b16[n*K+k] = uint16(dtype.BF16FromF32(v))
			}
		}
	}
	return nil
}
