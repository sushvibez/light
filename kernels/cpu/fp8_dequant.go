package cpu

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/internal/parallel"
	"light/internal/registry"
)

func init() {
	for _, dt := range []dtype.DType{dtype.F32, dtype.BF16} {
		registry.Register(registry.Key{Op: "fp8_block_dequant", Dev: device.KindCPU, DType: dt}, fp8BlockDequantCPU)
	}
}

// f8Table is every F8E4M3 byte widened to F32.
var f8Table = func() (t [256]float32) {
	for i := range t {
		t[i] = dtype.Float8(i).ToF32()
	}
	return
}()

// fp8BlockDequantCPU is ops.DequantizeF8Block in one parallel pass:
// out[r, c] = f32(w[r, c]) · scale[r/bR, c/bC], rounded once to the output
// dtype — bit for bit the cast → multiply → cast sequence.
//
//	Inputs: w (R, C) F8E4M3 contiguous, scale (R/bR, C/bC) F32 contiguous
//	Output: (R, C) F32 or BF16
//	Attrs:  "block_r", "block_c" int
func fp8BlockDequantCPU(ctx *registry.Context) error {
	w, s, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	bR, _ := ctx.Attrs["block_r"].(int)
	bC, _ := ctx.Attrs["block_c"].(int)
	R, C := w.Dim(0), w.Dim(1)
	if bR <= 0 || bC <= 0 || R%bR != 0 || C%bC != 0 {
		return fmt.Errorf("cpu fp8_block_dequant: (%d, %d) not a multiple of block (%d, %d)", R, C, bR, bC)
	}
	gc := C / bC
	ws := w.Storage().CPUBytes()[w.Offset():]
	ss := asSlice[float32](s)[s.Offset():]
	bf16 := out.DType() == dtype.BF16
	var o32 []float32
	var o16 []uint16
	if bf16 {
		o16 = asSlice[uint16](out)
	} else {
		o32 = asSlice[float32](out)
	}
	parallel.For(R, 16, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			srow := ss[(r/bR)*gc : (r/bR)*gc+gc]
			wrow := ws[r*C : r*C+C]
			for g, sc := range srow {
				for c := g * bC; c < g*bC+bC; c++ {
					v := f8Table[wrow[c]] * sc
					if bf16 {
						o16[r*C+c] = uint16(dtype.BF16FromF32(v))
					} else {
						o32[r*C+c] = v
					}
				}
			}
		}
	})
	return nil
}
