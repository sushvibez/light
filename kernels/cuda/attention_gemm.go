//go:build !nocuda && cgo

package cuda

import (
	"fmt"
	"os"
	"strconv"
	"time"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/internal/registry"
	"light/tensor"
)

func init() {
	registry.Register(registry.Key{Op: "attention_gemm", Dev: device.KindCUDA, DType: dtype.F32}, func(ctx *registry.Context) error {
		scale, _ := ctx.Attrs["scale"].(float64)
		return AttentionGEMMF32(ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output, float32(scale))
	})
}

// attentionGEMMBudget bounds the (BH, rows, S) score block AttentionGEMMF32
// holds at once; queries run in chunks of rows under it.
var attentionGEMMBudget int64 = 512 << 20

// attentionGEMMPerHeadS is the sequence length from which each head gets
// its own GEMM call (LIGHT_ATTN_PERHEAD_S overrides, for measurements).
var attentionGEMMPerHeadS = func() int {
	if v, err := strconv.Atoi(os.Getenv("LIGHT_ATTN_PERHEAD_S")); err == nil && v > 0 {
		return v
	}
	return 512
}()

// AttentionGEMMF32 computes softmax(scale · Q Kᵀ) V for contiguous F32
// (BH, S, D) tensors with two batched cuBLAS GEMMs and an in-place row
// softmax between them, in chunks of query rows: the unmasked attention of
// diffusion transformers, where Sq = Sk is a few thousand and tensor
// cores are not allowed (exact F32). out is (BH, S, D), written whole.
func AttentionGEMMF32(q, k, v, out *tensor.Tensor, scale float32) error {
	for _, t := range []*tensor.Tensor{q, k, v, out} {
		if t.DType() != dtype.F32 || t.Rank() != 3 || !t.IsContiguous() || t.Offset() != 0 {
			return fmt.Errorf("cuda attention_gemm: want contiguous F32 (BH, S, D), got %v %v", t.DType(), t.Shape())
		}
	}
	BH, S, D := q.Dim(0), q.Dim(1), q.Dim(2)
	if !k.Shape().Equal(q.Shape()) || !v.Shape().Equal(q.Shape()) || !out.Shape().Equal(q.Shape()) {
		return fmt.Errorf("cuda attention_gemm: shapes %v %v %v %v", q.Shape(), k.Shape(), v.Shape(), out.Shape())
	}
	ord := out.Device().Ordinal()
	if err := cudart.SetDevice(ord); err != nil {
		return err
	}
	h, err := cudart.GetBlasHandle(ord)
	if err != nil {
		return err
	}
	expFn, err := GetKernel("attn_exp_rows_f32")
	if err != nil {
		return err
	}
	scaleFn, err := GetKernel("attn_scale_rows_f32")
	if err != nil {
		return err
	}
	rows := int(attentionGEMMBudget / (int64(BH) * int64(S) * 4))
	rows = max(1, min(rows, S))
	scores, err := tensor.Empty(dtype.F32, []int{BH, rows, S}, out.Device())
	if err != nil {
		return err
	}
	sums, err := tensor.Empty(dtype.F32, []int{BH * rows}, out.Device())
	if err != nil {
		return err
	}
	qp, kp, vp, op, sp := q.Storage().CUDAPtr(), k.Storage().CUDAPtr(), v.Storage().CUDAPtr(), out.Storage().CUDAPtr(), scores.Storage().CUDAPtr()
	sump := sums.Storage().CUDAPtr()
	prof := os.Getenv("LIGHT_ATTN_PROFILE") != ""
	// One GEMM per head beats the strided-batched call once the matrices
	// are large (cuBLAS tunes the batched path for many small ones): 1.5x
	// on the score GEMM at 4352 tokens.
	perHead := S >= attentionGEMMPerHeadS
	var tG1, tE, tG2, tS time.Duration
	mark := func(d *time.Duration, t0 time.Time) time.Time {
		if !prof {
			return t0
		}
		_ = cudart.DeviceSynchronize()
		now := time.Now()
		*d += now.Sub(t0)
		return now
	}
	defer func() {
		if prof {
			fmt.Printf("attention_gemm S=%d rows=%d: gemm1 %v exp %v gemm2 %v scale %v\n", S, rows, tG1, tE, tG2, tS)
		}
	}()
	for off := 0; off < S; off += rows {
		m := min(rows, S-off)
		t0 := time.Now()
		if prof {
			_ = cudart.DeviceSynchronize()
			t0 = time.Now()
		}
		// scores (BH, m, S) = scale · Q[off:off+m] · Kᵀ
		if perHead {
			for b := range BH {
				if err := cudart.SgemmStridedBatched(h, m, S, D,
					cudart.BatchedOperand{Ptr: qp + uintptr((b*S+off)*D*4), LD: D, Stride: S * D},
					cudart.BatchedOperand{Ptr: kp + uintptr(b*S*D*4), LD: D, Stride: S * D, Trans: true},
					sp+uintptr(b*m*S*4), m*S, scale, 1); err != nil {
					return err
				}
			}
		} else if err := cudart.SgemmStridedBatched(h, m, S, D,
			cudart.BatchedOperand{Ptr: qp + uintptr(off*D*4), LD: D, Stride: S * D},
			cudart.BatchedOperand{Ptr: kp, LD: D, Stride: S * D, Trans: true},
			sp, m*S, scale, BH); err != nil {
			return err
		}
		t0 = mark(&tG1, t0)
		// e = exp(scores - row max) in place, row sums aside; the division
		// by the sum happens on the (BH, m, D) result instead of the
		// (BH, m, S) block.
		n, cols := uint64(BH*m), uint64(S)
		args := []unsafe.Pointer{unsafe.Pointer(&sp), unsafe.Pointer(&sump), unsafe.Pointer(&n), unsafe.Pointer(&cols)}
		if err := cudart.LaunchKernel(expFn, uint32(n), 1, 1, 256, 1, 1, 256*4, 0, args); err != nil {
			return err
		}
		t0 = mark(&tE, t0)
		// out[off:off+m] (BH, m, D) = e · V
		chunk := op + uintptr(off*D*4)
		if perHead {
			for b := range BH {
				if err := cudart.SgemmStridedBatched(h, m, D, S,
					cudart.BatchedOperand{Ptr: sp + uintptr(b*m*S*4), LD: S, Stride: m * S},
					cudart.BatchedOperand{Ptr: vp + uintptr(b*S*D*4), LD: D, Stride: S * D},
					chunk+uintptr(b*S*D*4), S*D, 1, 1); err != nil {
					return err
				}
			}
		} else if err := cudart.SgemmStridedBatched(h, m, D, S,
			cudart.BatchedOperand{Ptr: sp, LD: S, Stride: m * S},
			cudart.BatchedOperand{Ptr: vp, LD: D, Stride: S * D},
			chunk, S*D, 1, BH); err != nil {
			return err
		}
		t0 = mark(&tG2, t0)
		bh, mm, dd, hs := uint64(BH), uint64(m), uint64(D), uint64(S*D)
		sargs := []unsafe.Pointer{unsafe.Pointer(&chunk), unsafe.Pointer(&sump), unsafe.Pointer(&bh), unsafe.Pointer(&mm), unsafe.Pointer(&dd), unsafe.Pointer(&hs)}
		total := BH * m * D
		if err := cudart.LaunchKernel(scaleFn, uint32((total+255)/256), 1, 1, 256, 1, 1, 0, 0, sargs); err != nil {
			return err
		}
		mark(&tS, t0)
	}
	return nil
}

// SetAttentionGEMMBudget sets the score-block bound (tests) and returns a
// function restoring the previous one.
func SetAttentionGEMMBudget(b int64) func() {
	old := attentionGEMMBudget
	attentionGEMMBudget = b
	return func() { attentionGEMMBudget = old }
}
