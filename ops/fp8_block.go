package ops

import (
	"fmt"

	"light/dtype"
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
