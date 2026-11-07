package ops

import (
	"fmt"

	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// DequantizeF8Block expands a block-scaled F8E4M3 matrix — the "fp8"
// fine-grained format of Qwen3.5 / Qwen3 / DeepSeek-V3 FP8 checkpoints — to
// outDT:
//
//	out[i, j] = float(w[i, j]) · scale[i / blockR, j / blockC]
//
// w is (R, C) F8E4M3, scale is (R / blockR, C / blockC) F32 (the
// checkpoint's `weight_scale_inv`, a multiplier despite the name).
// R and C must be multiples of the block (true of every published
// checkpoint; transformers' own dequantizer requires the same). The product
// is formed in F32 and rounded once to outDT, as transformers does.
func DequantizeF8Block(w, scale *tensor.Tensor, blockR, blockC int, outDT dtype.DType) (*tensor.Tensor, error) {
	if w.DType() != dtype.F8E4M3 || w.Rank() != 2 {
		return nil, fmt.Errorf("ops.DequantizeF8Block: want a rank-2 F8E4M3 weight, got %v %v", w.DType(), w.Shape())
	}
	R, C := w.Dim(0), w.Dim(1)
	if blockR <= 0 || blockC <= 0 || R%blockR != 0 || C%blockC != 0 {
		return nil, fmt.Errorf("ops.DequantizeF8Block: weight (%d, %d) not a multiple of block (%d, %d)", R, C, blockR, blockC)
	}
	gr, gc := R/blockR, C/blockC
	if scale.DType() != dtype.F32 || scale.Rank() != 2 || scale.Dim(0) != gr || scale.Dim(1) != gc {
		return nil, fmt.Errorf("ops.DequantizeF8Block: scale %v %v, want F32 (%d, %d)", scale.DType(), scale.Shape(), gr, gc)
	}
	if !w.Device().SameAs(scale.Device()) {
		return nil, fmt.Errorf("ops.DequantizeF8Block: weight on %v, scale on %v", w.Device(), scale.Device())
	}
	// Fused kernel (one pass, CUDA) when there is one for outDT.
	if C%4 == 0 && blockC%4 == 0 {
		if k, err := registry.Lookup(registry.Key{Op: "fp8_block_dequant", Dev: w.Device().Kind(), DType: outDT}); err == nil {
			wc, err := w.Contiguous()
			if err != nil {
				return nil, err
			}
			sc, err := scale.Contiguous()
			if err != nil {
				return nil, err
			}
			out, err := tensor.Zeros(outDT, []int{R, C}, w.Device())
			if err != nil {
				return nil, err
			}
			if err := k(&registry.Context{Inputs: []*tensor.Tensor{wc, sc}, Output: out,
				Attrs: map[string]any{"block_r": blockR, "block_c": blockC}}); err != nil {
				return nil, err
			}
			return out, nil
		}
	}
	w32, err := Cast(w, dtype.F32)
	if err != nil {
		return nil, err
	}
	if w32, err = w32.Reshape(gr, blockR, gc, blockC); err != nil {
		return nil, err
	}
	s, err := scale.Contiguous()
	if err != nil {
		return nil, err
	}
	if s, err = s.Reshape(gr, 1, gc, 1); err != nil {
		return nil, err
	}
	out, err := Mul(w32, s)
	if err != nil {
		return nil, err
	}
	if out, err = out.Reshape(R, C); err != nil {
		return nil, err
	}
	if outDT == dtype.F32 {
		return out, nil
	}
	return Cast(out, outDT)
}

// fp8GEMVMaxM is the largest row count the fused FP8 GEMV kernel takes
// (FP8_GEMV_MAXM in kernels/cuda/src/fp8_block.cu).
const fp8GEMVMaxM = 8

// MatMulF8Block computes x · Wᵀ for a block-scaled F8E4M3 weight w (N, K)
// with scale (N / blockR, K / blockC): x is (M, K) F32 or BF16, the result
// (M, N) at x's dtype. Up to 8 rows (decode) on CUDA it runs a fused GEMV
// that reads the 1-byte codes directly; otherwise the weight is expanded to
// x's dtype (DequantizeF8Block) and multiplied by the regular GEMM.
func MatMulF8Block(x, w, scale *tensor.Tensor, blockR, blockC int) (*tensor.Tensor, error) {
	if x.Rank() != 2 || w.Rank() != 2 || x.Dim(1) != w.Dim(1) {
		return nil, fmt.Errorf("ops.MatMulF8Block: x %v, w %v", x.Shape(), w.Shape())
	}
	M, K, N := x.Dim(0), x.Dim(1), w.Dim(0)
	if M <= fp8GEMVMaxM && K%16 == 0 && blockC%16 == 0 && blockR > 0 && N%blockR == 0 && K%blockC == 0 &&
		w.DType() == dtype.F8E4M3 && scale.DType() == dtype.F32 && scale.Rank() == 2 &&
		scale.Dim(0) == N/blockR && scale.Dim(1) == K/blockC {
		if k, err := registry.Lookup(registry.Key{Op: "fp8_block_gemv", Dev: x.Device().Kind(), DType: x.DType()}); err == nil {
			xc, err := x.Contiguous()
			if err != nil {
				return nil, err
			}
			wc, err := w.Contiguous()
			if err != nil {
				return nil, err
			}
			sc, err := scale.Contiguous()
			if err != nil {
				return nil, err
			}
			out, err := tensor.Zeros(x.DType(), []int{M, N}, x.Device())
			if err != nil {
				return nil, err
			}
			if err := k(&registry.Context{Inputs: []*tensor.Tensor{xc, wc, sc}, Output: out,
				Attrs: map[string]any{"block_r": blockR, "block_c": blockC}}); err != nil {
				return nil, err
			}
			return out, nil
		}
	}
	dq, err := DequantizeF8Block(w, scale, blockR, blockC, x.DType())
	if err != nil {
		return nil, err
	}
	wT, err := dq.Transpose(0, 1)
	if err != nil {
		return nil, err
	}
	xc, err := x.Contiguous()
	if err != nil {
		return nil, err
	}
	if x.DType() == dtype.BF16 || x.DType() == dtype.F16 {
		return MatMulMixed(xc, wT)
	}
	return MatMul(xc, wT)
}
