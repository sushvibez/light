package cpu

import (
	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

func init() {
	registry.Register(registry.Key{Op: "softmax", Dev: device.KindCPU, DType: dtype.BF16}, viaF32(softmaxKernel[float32]))
}

// viaF32 runs an F32 kernel for BF16 tensors: every input that is BF16 is
// widened to a contiguous F32 copy (honouring its strides and offset), the
// kernel writes an F32 output, and that is rounded into the BF16 output
// once. It suits ops whose F32 kernel already computes in F32 or wider, so
// the only BF16 rounding is the final one, as in transformers' BF16 path
// for softmax (computed in float32, cast back).
func viaF32(k registry.Kernel) registry.Kernel {
	return func(ctx *registry.Context) error {
		ins := make([]*tensor.Tensor, len(ctx.Inputs))
		for i, t := range ctx.Inputs {
			if t != nil && t.DType() == dtype.BF16 {
				w, err := bf16ToF32(t)
				if err != nil {
					return err
				}
				t = w
			}
			ins[i] = t
		}
		out, err := tensor.Zeros(dtype.F32, ctx.Output.Shape().Dims(), device.CPU())
		if err != nil {
			return err
		}
		if err := k(&registry.Context{Inputs: ins, Output: out, Attrs: ctx.Attrs}); err != nil {
			return err
		}
		src := asSlice[float32](out)
		dst := asSlice[uint16](ctx.Output)
		for i, v := range src {
			dst[i] = uint16(dtype.BF16FromF32(v))
		}
		return nil
	}
}

// bf16ToF32 returns a contiguous F32 copy of a (possibly strided) BF16
// CPU tensor.
func bf16ToF32(t *tensor.Tensor) (*tensor.Tensor, error) {
	dims := t.Shape().Dims()
	out, err := tensor.Zeros(dtype.F32, dims, device.CPU())
	if err != nil {
		return nil, err
	}
	src := asSlice[uint16](t)
	dst := asSlice[float32](out)
	strides := t.Strides().Slice()
	rank := len(dims)
	idx := make([]int, rank)
	for i := range dst {
		off := t.Offset()
		for ax := range rank {
			off += idx[ax] * strides[ax]
		}
		dst[i] = dtype.BFloat16(src[off]).ToF32()
		for ax := rank - 1; ax >= 0; ax-- {
			idx[ax]++
			if idx[ax] < dims[ax] {
				break
			}
			idx[ax] = 0
		}
	}
	return out, nil
}
