package ops

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/quant"
	"light/tensor"
)

// DequantizeInt4 expands a canonical 4-bit weight (quant.Int4, from GPTQ /
// AWQ checkpoints) to an (N, K) tensor at outDT (F32 or BF16):
// W[n, k] = s[n, k/G] · (code(n, k) − z[n, k/G]), formed in F32, rounded once.
func DequantizeInt4(w quant.Int4, outDT dtype.DType) (*tensor.Tensor, error) {
	out, err := tensor.Zeros(outDT, []int{w.N(), w.K()}, w.Codes.Device())
	if err != nil {
		return nil, err
	}
	if err := dequantizeInt4Into(w, out); err != nil {
		return nil, err
	}
	return out, nil
}

func dequantizeInt4Into(w quant.Int4, out *tensor.Tensor) error {
	if w.Group <= 0 || w.K()%w.Group != 0 {
		return fmt.Errorf("ops.DequantizeInt4: group %d does not divide K %d", w.Group, w.K())
	}
	k, err := registry.Lookup(registry.Key{Op: "int4_dequant", Dev: w.Codes.Device().Kind(), DType: out.DType()})
	if err != nil {
		return err
	}
	codes, err := w.Codes.Contiguous()
	if err != nil {
		return err
	}
	s, err := w.Scales.Contiguous()
	if err != nil {
		return err
	}
	z, err := w.Zeros.Contiguous()
	if err != nil {
		return err
	}
	return k(&registry.Context{Inputs: []*tensor.Tensor{codes, s, z}, Output: out, Attrs: map[string]any{"group": w.Group}})
}

// int4GEMVMaxM is INT4_GEMV_MAXM in kernels/cuda/src/int4.cu.
const int4GEMVMaxM = 8

// MatMulInt4 computes x · Wᵀ for a 4-bit weight w: x is (M, K) F32 or BF16,
// the result (M, N) at x's dtype. Up to 8 rows (decode) on CUDA it runs a
// fused GEMV reading the packed codes directly (half a byte per weight);
// otherwise the weight is expanded into the shared scratch buffer
// (see fp8Scratch) and multiplied by the regular GEMM.
func MatMulInt4(x *tensor.Tensor, w quant.Int4) (*tensor.Tensor, error) {
	if x.Rank() != 2 || x.Dim(1) != w.K() {
		return nil, fmt.Errorf("ops.MatMulInt4: x %v, weight (%d, %d)", x.Shape(), w.N(), w.K())
	}
	if x.DType() != dtype.F32 && x.DType() != dtype.BF16 {
		return nil, fmt.Errorf("ops.MatMulInt4: activations %v, want F32 or BF16", x.DType())
	}
	M, N := x.Dim(0), w.N()
	xc, err := x.Contiguous()
	if err != nil {
		return nil, err
	}
	if w.Perm != nil {
		// Act-order GPTQ: the weight's K axis is stored sorted by group;
		// gather x's columns into the same order (x[:, Perm]).
		if xc, err = gatherColumns(xc, w.Perm); err != nil {
			return nil, err
		}
	}
	if M <= int4GEMVMaxM && w.K()%32 == 0 && w.Group%32 == 0 {
		if k, err := registry.Lookup(registry.Key{Op: "int4_gemv", Dev: x.Device().Kind(), DType: x.DType()}); err == nil {
			out, err := tensor.Zeros(x.DType(), []int{M, N}, x.Device())
			if err != nil {
				return nil, err
			}
			if err := k(&registry.Context{Inputs: []*tensor.Tensor{xc, w.Codes, w.Scales, w.Zeros}, Output: out,
				Attrs: map[string]any{"group": w.Group}}); err != nil {
				return nil, err
			}
			return out, nil
		}
	}
	gemm := func(dq *tensor.Tensor) (*tensor.Tensor, error) {
		wT, err := dq.Transpose(0, 1)
		if err != nil {
			return nil, err
		}
		if x.DType() == dtype.BF16 {
			return MatMulMixed(xc, wT)
		}
		return MatMul(xc, wT)
	}
	if x.Device().Kind() != device.KindCPU {
		fp8ScratchMu.Lock()
		defer fp8ScratchMu.Unlock()
		dq, err := weightScratch(x.Device(), x.DType(), N, w.K())
		if err != nil {
			return nil, err
		}
		if err := dequantizeInt4Into(w, dq); err != nil {
			return nil, err
		}
		return gemm(dq)
	}
	dq, err := DequantizeInt4(w, x.DType())
	if err != nil {
		return nil, err
	}
	return gemm(dq)
}

// gatherColumns returns x[:, idx] for a rank-2 x and an I64 index vector.
func gatherColumns(x, idx *tensor.Tensor) (*tensor.Tensor, error) {
	xt, err := x.Transpose(0, 1)
	if err != nil {
		return nil, err
	}
	if xt, err = xt.Contiguous(); err != nil {
		return nil, err
	}
	g, err := IndexSelect(xt, idx)
	if err != nil {
		return nil, err
	}
	gt, err := g.Transpose(0, 1)
	if err != nil {
		return nil, err
	}
	return gt.Contiguous()
}
