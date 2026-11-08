package cpu

import (
	"encoding/binary"
	"fmt"
	"sync"

	"light/device"
	"light/dtype"
	"light/internal/parallel"
	"light/internal/registry"
	"light/internal/simd"
)

// Quantized matmuls on the CPU without expanding the weight: each worker
// decodes one weight row at a time into a K-float buffer and takes SIMD
// dot products with every activation row. The weight is read once per
// call, and the decode cost is shared by all M rows (prefill).

// blockDecoder decodes one block of raw bytes into dst (len = block elems).
type blockDecoder struct {
	elems, bytes int
	decode       func(blk []byte, dst []float32)
}

func f16(b []byte) float32 { return dtype.Half(binary.LittleEndian.Uint16(b)).ToF32() }

var ggufDecoders = map[dtype.DType]blockDecoder{
	dtype.Q8_0: {32, 34, func(b []byte, dst []float32) {
		d := f16(b)
		for i := range 32 {
			dst[i] = d * float32(int8(b[2+i]))
		}
	}},
	dtype.Q4_0: {32, 18, func(b []byte, dst []float32) {
		d := f16(b)
		for j := range 16 {
			q := b[2+j]
			dst[j] = d * float32(int(q&15)-8)
			dst[j+16] = d * float32(int(q>>4)-8)
		}
	}},
	dtype.Q5_0: {32, 22, func(b []byte, dst []float32) {
		d := f16(b)
		qh := binary.LittleEndian.Uint32(b[2:6])
		for j := range 16 {
			q := b[6+j]
			lo := int(q&15) | int((qh>>j)&1)<<4
			hi := int(q>>4) | int((qh>>(j+16))&1)<<4
			dst[j] = d * float32(lo-16)
			dst[j+16] = d * float32(hi-16)
		}
	}},
	dtype.Q4_K: {256, 144, func(b []byte, dst []float32) {
		d, dmin := f16(b[0:2]), f16(b[2:4])
		scales, qs := b[4:16], b[16:144]
		for p := range 4 {
			sc0, m0 := scaleMinK4(2*p, scales)
			sc1, m1 := scaleMinK4(2*p+1, scales)
			d1, mo1 := d*float32(sc0), dmin*float32(m0)
			d2, mo2 := d*float32(sc1), dmin*float32(m1)
			for l := range 32 {
				q := qs[p*32+l]
				dst[p*64+l] = d1*float32(q&15) - mo1
				dst[p*64+32+l] = d2*float32(q>>4) - mo2
			}
		}
	}},
	dtype.Q6_K: {256, 210, func(b []byte, dst []float32) {
		ql, qh, sc := b[0:128], b[128:192], b[192:208]
		d := f16(b[208:210])
		for half := range 2 {
			qlB, qhB, scB, y := half*64, half*32, half*8, half*128
			for l := range 32 {
				is := l >> 4
				h := qh[qhB+l]
				q1 := int(ql[qlB+l]&15) | int(h&3)<<4 - 32
				q2 := int(ql[qlB+l+32]&15) | int((h>>2)&3)<<4 - 32
				q3 := int(ql[qlB+l]>>4) | int((h>>4)&3)<<4 - 32
				q4 := int(ql[qlB+l+32]>>4) | int((h>>6)&3)<<4 - 32
				dst[y+l] = d * float32(int8(sc[scB+is])) * float32(q1)
				dst[y+l+32] = d * float32(int8(sc[scB+2+is])) * float32(q2)
				dst[y+l+64] = d * float32(int8(sc[scB+4+is])) * float32(q3)
				dst[y+l+96] = d * float32(int8(sc[scB+6+is])) * float32(q4)
			}
		}
	}},
}

// scaleMinK4 unpacks Q4_K's 6-bit (scale, min) pair j (light/quant's
// getScaleMinK4).
func scaleMinK4(j int, q []byte) (uint8, uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

func init() {
	for dt := range ggufDecoders {
		registry.Register(registry.Key{Op: "matmul_q", Dev: device.KindCPU, DType: dt}, matmulQCPU)
	}
	registry.Register(registry.Key{Op: "int4_gemv", Dev: device.KindCPU, DType: dtype.F32}, int4MatMulCPU)
}

// rowBufs hands each worker a reusable K-float row buffer.
var rowBufs = sync.Pool{New: func() any { return new([]float32) }}

func rowBuf(k int) *[]float32 {
	p := rowBufs.Get().(*[]float32)
	if cap(*p) < k {
		*p = make([]float32, k)
	}
	*p = (*p)[:k]
	return p
}

// rowsTimesX computes out[i, n] = x_i · row_n for n in [0, N), decoding the
// weight rows with decodeRow; parallel over n. Each worker decodes
// tileRows rows into a buffer, then multiplies all M activation rows by
// them with 4 × 3 register tiles (simd.MatTB), so prefill shares both the
// decode and every weight load across rows.
func rowsTimesX(x, out []float32, M, K, N int, decodeRow func(n int, dst []float32)) {
	grain := max(1, 16384/(K+1))
	parallel.For(N, grain, func(lo, hi int) {
		bp := rowBuf(tileRows * K)
		defer rowBufs.Put(bp)
		rows := *bp
		for n0 := lo; n0 < hi; n0 += tileRows {
			n1 := min(n0+tileRows, hi)
			for n := n0; n < n1; n++ {
				decodeRow(n, rows[(n-n0)*K:(n-n0+1)*K])
			}
			simd.MatTB(x, K, rows, K, out[n0:], N, M, n1-n0, K)
		}
	})
}

// matmulQCPU is ops.MatMulQ on the CPU: x (M, K) F32 contiguous times the
// GGUF-quantized weight (N, K) transposed.
func matmulQCPU(ctx *registry.Context) error {
	x, w, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Output
	dec, ok := ggufDecoders[w.DType()]
	if !ok {
		return fmt.Errorf("cpu matmul_q: no decoder for %v", w.DType())
	}
	M, K, N := x.Dim(0), x.Dim(1), w.Dim(0)
	if !w.IsContiguous() {
		return fmt.Errorf("cpu matmul_q: weight %v must be contiguous", w.Shape())
	}
	raw := w.Storage().CPUBytes()[w.DType().ByteSize(w.Offset()):]
	rowBytes := w.DType().ByteSize(K)
	xs := asSlice[float32](x)[x.Offset():]
	_, fast := ggufAVX2[w.DType()]
	fast = fast && decodeAVX2
	rowsTimesX(xs, asSlice[float32](out), M, K, N, func(n int, dst []float32) {
		r := raw[n*rowBytes:]
		if fast {
			var ds [2048]float32 // block scales (K ≤ 65536)
			if K/32 <= len(ds) {
				decodeRowAVX2(w.DType(), r, K, dst, ds[:])
				return
			}
		}
		for b := 0; b < K/dec.elems; b++ {
			dec.decode(r[b*dec.bytes:], dst[b*dec.elems:])
		}
	})
	return nil
}

// ggufAVX2 lists the formats decodeRowAVX2 handles.
var ggufAVX2 = map[dtype.DType]bool{dtype.Q8_0: true, dtype.Q4_0: true, dtype.Q5_0: true, dtype.Q4_K: true, dtype.Q6_K: true}

// int4MatMulCPU is ops.MatMulInt4's kernel on the CPU for any M: x (M, K)
// F32 times quant.Int4 (codes U32 (N, K/8), scales / zeros F32 (N, K/G))
// transposed.
func int4MatMulCPU(ctx *registry.Context) error {
	x, codes, sc, zr, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Inputs[3], ctx.Output
	M, K, N := x.Dim(0), x.Dim(1), codes.Dim(0)
	G := ctx.Attrs["group"].(int)
	cs := asSlice[uint32](codes)[codes.Offset():]
	ss, zs := asSlice[float32](sc)[sc.Offset():], asSlice[float32](zr)[zr.Offset():]
	xs := asSlice[float32](x)[x.Offset():]
	ng := K / G
	rowsTimesX(xs, asSlice[float32](out), M, K, N, func(n int, dst []float32) {
		if decodeAVX2 && G%8 == 0 {
			int4RowAVX2(&cs[n*(K/8)], &ss[n*ng], &zs[n*ng], &dst[0], ng, G/8)
			return
		}
		words := cs[n*(K/8):]
		for g := range ng {
			s, z := ss[n*ng+g], zs[n*ng+g]
			for k := g * G; k < (g+1)*G; k += 8 {
				w := words[k/8]
				for j := range 8 {
					dst[k+j] = s * (float32((w>>(4*j))&15) - z)
				}
			}
		}
	})
	return nil
}

// decodeRowAVX2 decodes a GGUF weight row of K elements with the AVX2
// decoders (bit-identical to the Go ones); ds needs K/32 floats of scratch.
// It reports false for formats without one.
func decodeRowAVX2(dt dtype.DType, raw []byte, K int, dst, ds []float32) bool {
	switch dt {
	case dtype.Q8_0, dtype.Q4_0, dtype.Q5_0:
		bb := map[dtype.DType]int{dtype.Q8_0: 34, dtype.Q4_0: 18, dtype.Q5_0: 22}[dt]
		nb := K / 32
		for b := range nb {
			ds[b] = f16(raw[b*bb:])
		}
		switch dt {
		case dtype.Q8_0:
			q8_0RowAVX2(&raw[0], &ds[0], &dst[0], nb)
		case dtype.Q4_0:
			q4_0RowAVX2(&raw[0], &ds[0], &dst[0], nb)
		default:
			q5_0RowAVX2(&raw[0], &ds[0], &dst[0], nb)
		}
		return true
	case dtype.Q4_K:
		var sc, mn [8]float32
		for b := 0; b < K/256; b++ {
			blk := raw[b*144:]
			d, dmin := f16(blk[0:2]), f16(blk[2:4])
			for j := range 8 {
				s, m := scaleMinK4(j, blk[4:16])
				sc[j], mn[j] = d*float32(s), dmin*float32(m)
			}
			q4KBlockAVX2(&blk[16], &sc[0], &mn[0], &dst[b*256])
		}
		return true
	case dtype.Q6_K:
		var sc [16]float32
		for b := 0; b < K/256; b++ {
			blk := raw[b*210:]
			d := f16(blk[208:210])
			for j := range 16 {
				sc[j] = d * float32(int8(blk[192+j]))
			}
			q6KBlockAVX2(&blk[0], &blk[128], &sc[0], &dst[b*256])
		}
		return true
	}
	return false
}
