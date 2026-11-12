package ops

import (
	"fmt"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// Decode-step linears: one activation row against several nn.Linear
// weights ((N, K) row-major BF16, no bias) in a single launch, and the
// gated MLP pair with its activation and product fused. The math is the
// unfused one (F32 sums rounded to BF16 once per matmul, the activation
// and product on BF16 values); devices without the kernels run exactly
// that out of MatMulMixed, SiLU / GeluErf and Mul.

// LinearMulti returns x·wᵢᵀ for each weight, shaped like x with the last
// dim replaced by Nᵢ. x must hold one row (its leading dims multiply to
// 1); outDT is BF16 or F32 for the results.
func LinearMulti(x *tensor.Tensor, ws []*tensor.Tensor, outDT dtype.DType) ([]*tensor.Tensor, error) {
	if len(ws) == 0 {
		return nil, fmt.Errorf("LinearMulti: no weights")
	}
	K := x.Dim(-1)
	if x.DType() != dtype.BF16 || x.NumElements() != K {
		return nil, fmt.Errorf("LinearMulti: x must be one BF16 row, got %v %v", x.DType(), x.Shape())
	}
	if outDT != dtype.BF16 && outDT != dtype.F32 {
		return nil, fmt.Errorf("LinearMulti: outDT %v", outDT)
	}
	for i, w := range ws {
		if w.Rank() != 2 || w.Dim(1) != K || w.DType() != dtype.BF16 || !w.Device().SameAs(x.Device()) {
			return nil, fmt.Errorf("LinearMulti: weight %d must be (N, %d) BF16 on %v, got %v %v on %v", i, K, x.Device(), w.Shape(), w.DType(), w.Device())
		}
	}
	lead := x.Shape().Dims()[:x.Rank()-1]
	shape := func(n int) []int { return append(append([]int{}, lead...), n) }
	kern, lerr := registry.Lookup(registry.Key{Op: "gemv_multi", Dev: x.Device().Kind(), DType: dtype.BF16})
	if lerr == nil && gemvKernelOK(K, len(ws)) && allContig(ws) {
		xc, err := ensureContig(x)
		if err != nil {
			return nil, err
		}
		outs := make([]*tensor.Tensor, len(ws))
		for i, w := range ws {
			if outs[i], err = tensor.Empty(outDT, shape(w.Dim(0)), x.Device()); err != nil {
				return nil, err
			}
		}
		if err := kern(&registry.Context{Inputs: append([]*tensor.Tensor{xc}, ws...), Attrs: map[string]any{"outs": outs}}); err != nil {
			return nil, err
		}
		return outs, nil
	}
	x2, err := rowOf(x)
	if err != nil {
		return nil, err
	}
	outs := make([]*tensor.Tensor, len(ws))
	for i, w := range ws {
		y, err := linearRow(x2, w)
		if err != nil {
			return nil, err
		}
		if outDT != y.DType() {
			if y, err = Cast(y, outDT); err != nil {
				return nil, err
			}
		}
		if outs[i], err = y.Reshape(shape(w.Dim(0))...); err != nil {
			return nil, err
		}
	}
	return outs, nil
}

// GatedLinear returns act(x·wgᵀ) · (x·wuᵀ) with act = SiLU, or erf-GELU
// when gelu is set, as BF16 shaped like x with the last dim N.
func GatedLinear(x, wg, wu *tensor.Tensor, gelu bool) (*tensor.Tensor, error) {
	K := x.Dim(-1)
	if x.DType() != dtype.BF16 || x.NumElements() != K {
		return nil, fmt.Errorf("GatedLinear: x must be one BF16 row, got %v %v", x.DType(), x.Shape())
	}
	for _, w := range []*tensor.Tensor{wg, wu} {
		if w.Rank() != 2 || w.Dim(1) != K || w.Dim(0) != wg.Dim(0) || w.DType() != dtype.BF16 || !w.Device().SameAs(x.Device()) {
			return nil, fmt.Errorf("GatedLinear: weights must be (N, %d) BF16 on %v, got %v / %v", K, x.Device(), wg.Shape(), wu.Shape())
		}
	}
	N := wg.Dim(0)
	shape := append(append([]int{}, x.Shape().Dims()[:x.Rank()-1]...), N)
	kern, lerr := registry.Lookup(registry.Key{Op: "gemv_gated", Dev: x.Device().Kind(), DType: dtype.BF16})
	if lerr == nil && gemvKernelOK(K, 1) && allContig([]*tensor.Tensor{wg, wu}) {
		xc, err := ensureContig(x)
		if err != nil {
			return nil, err
		}
		out, err := tensor.Empty(dtype.BF16, shape, x.Device())
		if err != nil {
			return nil, err
		}
		if err := kern(&registry.Context{Inputs: []*tensor.Tensor{xc, wg, wu}, Output: out, Attrs: map[string]any{"gelu": gelu}}); err != nil {
			return nil, err
		}
		return out, nil
	}
	x2, err := rowOf(x)
	if err != nil {
		return nil, err
	}
	g, err := linearRow(x2, wg)
	if err != nil {
		return nil, err
	}
	if gelu {
		g, err = GeluErf(g)
	} else {
		g, err = SiLU(g)
	}
	if err != nil {
		return nil, err
	}
	u, err := linearRow(x2, wu)
	if err != nil {
		return nil, err
	}
	y, err := Mul(g, u)
	if err != nil {
		return nil, err
	}
	return y.Reshape(shape...)
}

// gemvKernelOK mirrors the CUDA kernel's limits: K a multiple of 8 up to
// 12288, at most four weights.
func gemvKernelOK(K, nseg int) bool { return K%8 == 0 && K <= 12288 && nseg >= 1 && nseg <= 4 }

func allContig(ts []*tensor.Tensor) bool {
	for _, t := range ts {
		if !t.IsContiguous() || t.Offset() != 0 {
			return false
		}
	}
	return true
}

// rowOf views one-row x as (1, K).
func rowOf(x *tensor.Tensor) (*tensor.Tensor, error) {
	xc, err := ensureContig(x)
	if err != nil {
		return nil, err
	}
	return xc.Reshape(1, x.Dim(-1))
}

// linearRow is nn.Linear's BF16 matmul: x (1, K) against the transpose
// view of w (N, K).
func linearRow(x2, w *tensor.Tensor) (*tensor.Tensor, error) {
	wT, err := w.Transpose(0, 1)
	if err != nil {
		return nil, err
	}
	return MatMulMixed(x2, wT)
}
