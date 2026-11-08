package ops

import (
	"fmt"
	"sync"

	"light/device"
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
	if k := fusedDequantKernel(w, blockC, outDT); k != nil {
		out, err := tensor.Zeros(outDT, []int{R, C}, w.Device())
		if err != nil {
			return nil, err
		}
		if err := dequantizeF8BlockInto(k, w, scale, blockR, blockC, out); err != nil {
			return nil, err
		}
		return out, nil
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

// fusedDequantKernel returns the one-pass dequantize kernel for w's device
// and outDT, or nil when there is none (CPU) or the shape does not suit it.
func fusedDequantKernel(w *tensor.Tensor, blockC int, outDT dtype.DType) registry.Kernel {
	if w.Dim(1)%4 != 0 || blockC%4 != 0 {
		return nil
	}
	k, err := registry.Lookup(registry.Key{Op: "fp8_block_dequant", Dev: w.Device().Kind(), DType: outDT})
	if err != nil {
		return nil
	}
	return k
}

// dequantizeF8BlockInto runs the fused kernel into out, an offset-0
// contiguous (R, C) tensor.
func dequantizeF8BlockInto(k registry.Kernel, w, scale *tensor.Tensor, blockR, blockC int, out *tensor.Tensor) error {
	wc, err := w.Contiguous()
	if err != nil {
		return err
	}
	sc, err := scale.Contiguous()
	if err != nil {
		return err
	}
	return k(&registry.Context{Inputs: []*tensor.Tensor{wc, sc}, Output: out,
		Attrs: map[string]any{"block_r": blockR, "block_c": blockC}})
}

// fp8Scratch holds, per device and dtype, one buffer that MatMulF8Block
// expands weights into for the GEMM path. A fresh (R, C) tensor per call
// would leave a weight-sized buffer per matmul for the garbage collector to
// find: GPU tensors are freed by runtime cleanups, and the collector paces
// itself on the Go heap, not on device memory, so a 27B prefill piled up
// ~50 GB of them and ran out of memory. Reusing one buffer is safe because
// all work runs in order on one stream: the next dequantize is queued after
// the GEMM that reads the current contents. fp8ScratchMu keeps each
// dequantize + GEMM pair adjacent in that order.
var (
	fp8ScratchMu sync.Mutex
	fp8Scratch   = map[fp8ScratchKey]*tensor.Tensor{}
)

type fp8ScratchKey struct {
	dev  string
	dt   dtype.DType
	slot int // several buffers can be live at once (MoE: gate-up and down stacks)
}

// fp8ScratchView returns an (R, C) view at offset 0 of the scratch buffer
// for (w's device, dt), growing it when needed. Callers hold fp8ScratchMu.
func fp8ScratchView(w *tensor.Tensor, dt dtype.DType) (*tensor.Tensor, error) {
	return weightScratch(w.Device(), dt, w.Dim(0), w.Dim(1))
}

// weightScratch returns an (R, C) view at offset 0 of the per-(device,
// dtype) buffer that quantized weights (FP8, 4-bit) are expanded into before
// a GEMM; see fp8Scratch. Callers hold fp8ScratchMu.
func weightScratch(dev device.Device, dt dtype.DType, R, C int) (*tensor.Tensor, error) {
	return weightScratchSlot(dev, dt, 0, R, C)
}

// weightScratchSlot is weightScratch for buffer `slot`.
func weightScratchSlot(dev device.Device, dt dtype.DType, slot, R, C int) (*tensor.Tensor, error) {
	key := fp8ScratchKey{dev.String(), dt, slot}
	buf := fp8Scratch[key]
	if buf == nil || buf.NumElements() < R*C {
		var err error
		if buf, err = tensor.Zeros(dt, []int{R * C}, dev); err != nil {
			return nil, err
		}
		fp8Scratch[key] = buf
	}
	v, err := buf.Narrow(0, 0, R*C)
	if err != nil {
		return nil, err
	}
	return v.Reshape(R, C)
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
	xc, err := x.Contiguous()
	if err != nil {
		return nil, err
	}
	if k := fusedDequantKernel(w, blockC, x.DType()); k != nil && N%blockR == 0 && K%blockC == 0 {
		// Expand into the shared scratch buffer, then GEMM (see fp8Scratch).
		fp8ScratchMu.Lock()
		defer fp8ScratchMu.Unlock()
		dq, err := fp8ScratchView(w, x.DType())
		if err != nil {
			return nil, err
		}
		if err := dequantizeF8BlockInto(k, w, scale, blockR, blockC, dq); err != nil {
			return nil, err
		}
		wT, err := dq.Transpose(0, 1)
		if err != nil {
			return nil, err
		}
		if x.DType() == dtype.BF16 || x.DType() == dtype.F16 {
			return MatMulMixed(xc, wT)
		}
		return MatMul(xc, wT)
	}
	dq, err := DequantizeF8Block(w, scale, blockR, blockC, x.DType())
	if err != nil {
		return nil, err
	}
	wT, err := dq.Transpose(0, 1)
	if err != nil {
		return nil, err
	}
	if x.DType() == dtype.BF16 || x.DType() == dtype.F16 {
		return MatMulMixed(xc, wT)
	}
	return MatMul(xc, wT)
}
