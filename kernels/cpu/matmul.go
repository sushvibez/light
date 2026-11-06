package cpu

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

func init() {
	registry.Register(registry.Key{Op: "matmul", Dev: device.KindCPU, DType: dtype.F32}, matmulKernel[float32])
	registry.Register(registry.Key{Op: "matmul", Dev: device.KindCPU, DType: dtype.F64}, matmulKernel[float64])
	registry.Register(registry.Key{Op: "matmul", Dev: device.KindCPU, DType: dtype.I32}, matmulKernel[int32])
	registry.Register(registry.Key{Op: "matmul", Dev: device.KindCPU, DType: dtype.I64}, matmulKernel[int64])
	registry.Register(registry.Key{Op: "matmul", Dev: device.KindCPU, DType: dtype.F16}, matmulHalfKernel)
	registry.Register(registry.Key{Op: "matmul", Dev: device.KindCPU, DType: dtype.BF16}, matmulBF16Kernel)
}

// matmulHalfKernel implements F16 matmul by up-casting each read to F32,
// accumulating in F32, and down-casting on write. Matches Candle's CPU
// behaviour and our F16 elementwise convention.
func matmulHalfKernel(ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	aDim := a.Shape().Dims()
	bDim := b.Shape().Dims()
	M, K := aDim[0], aDim[1]
	N := bDim[1]
	aS := asSlice[uint16](a)
	bS := asSlice[uint16](b)
	outS := asSlice[uint16](out)
	aStr := a.Strides().Slice()
	bStr := b.Strides().Slice()
	aBase := a.Offset()
	bBase := b.Offset()
	for i := range M {
		for j := range N {
			var sum float32
			for k := range K {
				av := dtype.Half(aS[aBase+i*aStr[0]+k*aStr[1]]).ToF32()
				bv := dtype.Half(bS[bBase+k*bStr[0]+j*bStr[1]]).ToF32()
				sum += av * bv
			}
			outS[i*N+j] = uint16(dtype.HalfFromF32(sum))
		}
	}
	return nil
}

// matmulBF16Kernel is the BF16 counterpart.
func matmulBF16Kernel(ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	aDim := a.Shape().Dims()
	bDim := b.Shape().Dims()
	M, K := aDim[0], aDim[1]
	N := bDim[1]
	aS := asSlice[uint16](a)
	bS := asSlice[uint16](b)
	outS := asSlice[uint16](out)
	aStr := a.Strides().Slice()
	bStr := b.Strides().Slice()
	aBase := a.Offset()
	bBase := b.Offset()
	for i := range M {
		for j := range N {
			var sum float32
			for k := range K {
				av := dtype.BFloat16(aS[aBase+i*aStr[0]+k*aStr[1]]).ToF32()
				bv := dtype.BFloat16(bS[bBase+k*bStr[0]+j*bStr[1]]).ToF32()
				sum += av * bv
			}
			outS[i*N+j] = uint16(dtype.BF16FromF32(sum))
		}
	}
	return nil
}

// matmulKernel implements 2D matrix multiplication: (M,K) × (K,N) → (M,N).
// Phase 1 is the naive triple-loop reference; Phase 2 wires cuBLAS for CUDA,
// and Phase 3/10 revisits CPU perf with tiling if needed.
func matmulKernel[T dtype.Numeric](ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output

	aDim := a.Shape().Dims()
	bDim := b.Shape().Dims()
	M, K := aDim[0], aDim[1]
	N := bDim[1]

	aS := asSlice[T](a)
	bS := asSlice[T](b)
	outS := asSlice[T](out)

	aStr := a.Strides().Slice()
	bStr := b.Strides().Slice()
	aBase := a.Offset()
	bBase := b.Offset()

	for i := range M {
		for j := range N {
			var sum T
			for k := range K {
				sum += aS[aBase+i*aStr[0]+k*aStr[1]] * bS[bBase+k*bStr[0]+j*bStr[1]]
			}
			outS[i*N+j] = sum
		}
	}
	return nil
}

func init() {
	for _, dt := range []dtype.DType{dtype.F32, dtype.BF16, dtype.F16} {
		registry.Register(registry.Key{Op: "matmul_f8", Dev: device.KindCPU, DType: dt}, matmulF8Kernel)
	}
}

// matmulF8Kernel is the reference for ops.MatMulF8: a (M, K) and b (K, N)
// are F8E4M3 (any strides), out = aScale · bScale · (a @ b) at the output
// dtype (F32 / BF16 / F16, selected by the registry key). Products and the
// accumulator are F32, as cuBLASLt's FP8 GEMM does.
func matmulF8Kernel(ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	scale := float32(ctx.Attrs["a_scale"].(float64) * ctx.Attrs["b_scale"].(float64))
	M, K := a.Dim(0), a.Dim(1)
	N := b.Dim(1)
	aS, bS := a.Storage().CPUBytes(), b.Storage().CPUBytes()
	aStr, bStr := a.Strides().Slice(), b.Strides().Slice()
	aBase, bBase := a.Offset(), b.Offset()
	// Decode once per operand instead of per product.
	af := make([]float32, M*K)
	for i := range M {
		for k := range K {
			af[i*K+k] = dtype.Float8(aS[aBase+i*aStr[0]+k*aStr[1]]).ToF32()
		}
	}
	bf := make([]float32, K*N)
	for k := range K {
		for j := range N {
			bf[k*N+j] = dtype.Float8(bS[bBase+k*bStr[0]+j*bStr[1]]).ToF32()
		}
	}
	var store func(i int, v float32)
	switch out.DType() {
	case dtype.F32:
		o := asSlice[float32](out)
		store = func(i int, v float32) { o[i] = v }
	case dtype.BF16:
		o := asSlice[uint16](out)
		store = func(i int, v float32) { o[i] = uint16(dtype.BF16FromF32(v)) }
	case dtype.F16:
		o := asSlice[uint16](out)
		store = func(i int, v float32) { o[i] = uint16(dtype.HalfFromF32(v)) }
	}
	for i := range M {
		for j := range N {
			var sum float32
			for k := range K {
				sum += af[i*K+k] * bf[k*N+j]
			}
			store(i*N+j, sum*scale)
		}
	}
	return nil
}

func init() {
	for _, dt := range []dtype.DType{dtype.F32, dtype.BF16, dtype.F16} {
		registry.Register(registry.Key{Op: "matmul_mixed", Dev: device.KindCPU, DType: dt}, matmulMixedKernel)
	}
}

// readF32 returns an element reader for an F32 / BF16 / F16 operand that
// follows the tensor's strides (so transpose views work without a copy).
func readF32(t *tensor.Tensor) (func(i, j int) float32, error) {
	str := t.Strides().Slice()
	off := t.Offset()
	switch t.DType() {
	case dtype.F32:
		s := asSlice[float32](t)
		return func(i, j int) float32 { return s[off+i*str[0]+j*str[1]] }, nil
	case dtype.BF16:
		s := asSlice[uint16](t)
		return func(i, j int) float32 { return dtype.BFloat16(s[off+i*str[0]+j*str[1]]).ToF32() }, nil
	case dtype.F16:
		s := asSlice[uint16](t)
		return func(i, j int) float32 { return dtype.Half(s[off+i*str[0]+j*str[1]]).ToF32() }, nil
	}
	return nil, fmt.Errorf("cpu matmul_mixed: unsupported operand dtype %v", t.DType())
}

// matmulMixedKernel is the CPU reference for ops.MatMulMixed: operands may
// be F32 / BF16 / F16 in any combination; products and accumulation are
// F32 and the result is rounded once to the output dtype — the cuBLAS
// GemmEx contract the CUDA kernels follow.
func matmulMixedKernel(ctx *registry.Context) error {
	a, b, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	ra, err := readF32(a)
	if err != nil {
		return err
	}
	rb, err := readF32(b)
	if err != nil {
		return err
	}
	M, K, N := a.Dim(0), a.Dim(1), b.Dim(1)
	var store func(i int, v float32)
	switch out.DType() {
	case dtype.F32:
		o := asSlice[float32](out)
		store = func(i int, v float32) { o[i] = v }
	case dtype.BF16:
		o := asSlice[uint16](out)
		store = func(i int, v float32) { o[i] = uint16(dtype.BF16FromF32(v)) }
	case dtype.F16:
		o := asSlice[uint16](out)
		store = func(i int, v float32) { o[i] = uint16(dtype.HalfFromF32(v)) }
	default:
		return fmt.Errorf("cpu matmul_mixed: unsupported output dtype %v", out.DType())
	}
	// Mixed-precision semantics: the activation is rounded to the weight's
	// 16-bit dtype before multiplying (what the CUDA path does by casting).
	roundA := func(v float32) float32 { return v }
	if a.DType() == dtype.F32 && b.DType() == dtype.BF16 {
		roundA = func(v float32) float32 { return dtype.BF16FromF32(v).ToF32() }
	} else if a.DType() == dtype.F32 && b.DType() == dtype.F16 {
		roundA = func(v float32) float32 { return dtype.HalfFromF32(v).ToF32() }
	}
	for i := range M {
		for j := range N {
			var sum float32
			for k := range K {
				sum += roundA(ra(i, k)) * rb(k, j)
			}
			store(i*N+j, sum)
		}
	}
	return nil
}
