package quant

import (
	"fmt"
	"sort"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// Int4 is Light's canonical layout for group-quantized 4-bit weights
// (GPTQ, AWQ). For a (N, K) weight with groups of G along K:
//
//	Codes  U32 (N, K/8)  nibble j of Codes[n, c] is the code of W[n, 8c + j]
//	Scales F32 (N, K/G)
//	Zeros  F32 (N, K/G)  zero point, in code units
//	W[n, k] = Scales[n, k/G] · (code(n, k) − Zeros[n, k/G])
//
// Rows are contiguous along K, which is what a GEMV over K and a row-wise
// dequantizer want; both checkpoint formats are converted to it once at load.
type Int4 struct {
	Codes, Scales, Zeros *tensor.Tensor
	Group                int
	// Perm, for act-order GPTQ (desc_act), lists the original input index of
	// each canonical column k (I64 (K,)): the weight's K axis is stored sorted
	// by group, and the activation's columns must be gathered the same way
	// (x[:, Perm]) before multiplying. nil when the groups are already in order.
	Perm *tensor.Tensor
}

// N, K return the dequantized weight's shape.
func (w Int4) N() int { return w.Codes.Dim(0) }
func (w Int4) K() int { return w.Codes.Dim(1) * 8 }

// nibble returns the j-th 4-bit field of a little-endian packed word.
func nibble(word uint32, j int) uint8 { return uint8(word>>(4*j)) & 0xF }

func u32s(t *tensor.Tensor, name string) ([]uint32, error) {
	c, err := t.To(device.CPU())
	if err != nil {
		return nil, err
	}
	if c, err = c.Contiguous(); err != nil {
		return nil, err
	}
	switch c.DType() {
	case dtype.I32:
		v := tensor.MustSlice[int32](c)
		out := make([]uint32, len(v))
		for i, x := range v {
			out[i] = uint32(x)
		}
		return out, nil
	case dtype.U32:
		return append([]uint32(nil), tensor.MustSlice[uint32](c)...), nil
	}
	return nil, fmt.Errorf("quant: %s is %v, want I32 / U32", name, c.DType())
}

func f32s(t *tensor.Tensor) ([]float32, error) {
	c, err := t.To(device.CPU())
	if err != nil {
		return nil, err
	}
	if c.DType() != dtype.F32 {
		return nil, fmt.Errorf("quant: scales / zeros must be F32 (cast the checkpoint's F16 first), got %v", c.DType())
	}
	if c, err = c.Contiguous(); err != nil {
		return nil, err
	}
	return tensor.MustSlice[float32](c), nil
}

// build assembles the canonical tensors on the CPU from codes[n][k]
// (via code(n, k)), scales and zeros indexed [n·groups + g].
func build(N, K, G int, code func(n, k int) uint8, sc, zr []float32) (Int4, error) {
	packed := make([]uint32, N*K/8)
	for n := range N {
		for c := 0; c < K/8; c++ {
			var w uint32
			for j := range 8 {
				w |= uint32(code(n, 8*c+j)) << (4 * j)
			}
			packed[n*(K/8)+c] = w
		}
	}
	codes, err := tensor.New(packed, []int{N, K / 8}, device.CPU())
	if err != nil {
		return Int4{}, err
	}
	s, err := tensor.New(sc, []int{N, K / G}, device.CPU())
	if err != nil {
		return Int4{}, err
	}
	z, err := tensor.New(zr, []int{N, K / G}, device.CPU())
	if err != nil {
		return Int4{}, err
	}
	return Int4{Codes: codes, Scales: s, Zeros: z, Group: G}, nil
}

// FromGPTQ converts an AutoGPTQ / GPTQModel 4-bit linear (checkpoint_format
// "gptq"): qweight I32 (K/8, N) packed along K (nibble j of qweight[r, n] is
// row 8r + j), qzeros I32 (K/G, N/8) packed along N holding zero − 1,
// scales (K/G, N) cast to F32 by the caller, and g_idx (K,). Only the ordered grouping
// (g_idx[k] = k / G, i.e. desc_act false) is supported. v2 is
// checkpoint_format "gptq_v2", whose qzeros hold the zero point itself.
func FromGPTQ(qweight, qzeros, scales, gIdx *tensor.Tensor, group int, v2 bool) (Int4, error) {
	if qweight.Rank() != 2 || qzeros.Rank() != 2 || scales.Rank() != 2 {
		return Int4{}, fmt.Errorf("quant.FromGPTQ: rank-2 qweight / qzeros / scales expected")
	}
	K, N := qweight.Dim(0)*8, qweight.Dim(1)
	if group <= 0 {
		group = K // group_size -1: one group over all of K
	}
	if K%group != 0 || scales.Dim(0) != K/group || scales.Dim(1) != N || qzeros.Dim(0) != K/group || qzeros.Dim(1)*8 != N {
		return Int4{}, fmt.Errorf("quant.FromGPTQ: qweight %v, qzeros %v, scales %v, group %d inconsistent",
			qweight.Shape(), qzeros.Shape(), scales.Shape(), group)
	}
	// Act-order (desc_act) checkpoints assign input rows to groups out of
	// order: sort K by group (stably) so each group is contiguous again, and
	// remember the order so the activation can be gathered to match.
	var perm []int
	if gIdx != nil {
		gi, err := u32s(gIdx, "g_idx")
		if err != nil {
			return Int4{}, err
		}
		if len(gi) != K {
			return Int4{}, fmt.Errorf("quant.FromGPTQ: g_idx has %d entries for K %d", len(gi), K)
		}
		ordered := true
		for k, g := range gi {
			if int(g) != k/group {
				ordered = false
				break
			}
		}
		if !ordered {
			perm = make([]int, K)
			for k := range perm {
				perm[k] = k
			}
			sort.SliceStable(perm, func(a, b int) bool { return gi[perm[a]] < gi[perm[b]] })
			for kk, k := range perm {
				if int(gi[k]) != kk/group {
					return Int4{}, fmt.Errorf("quant.FromGPTQ: g_idx groups are not all of size %d", group)
				}
			}
		}
	}
	qw, err := u32s(qweight, "qweight")
	if err != nil {
		return Int4{}, err
	}
	qz, err := u32s(qzeros, "qzeros")
	if err != nil {
		return Int4{}, err
	}
	sc, err := f32s(scales)
	if err != nil {
		return Int4{}, err
	}
	G := K / group
	scT, zT := make([]float32, N*G), make([]float32, N*G)
	for g := range G {
		for n := range N {
			scT[n*G+g] = sc[g*N+n]
			zT[n*G+g] = float32(nibble(qz[g*(N/8)+n/8], n%8))
			if !v2 {
				zT[n*G+g]++ // "gptq" format stores zero − 1
			}
		}
	}
	src := func(k int) int { return k }
	if perm != nil {
		src = func(k int) int { return perm[k] }
	}
	w, err := build(N, K, group, func(n, k int) uint8 { r := src(k); return nibble(qw[(r/8)*N+n], r%8) }, scT, zT)
	if err != nil || perm == nil {
		return w, err
	}
	p := make([]int64, K)
	for i, k := range perm {
		p[i] = int64(k)
	}
	if w.Perm, err = tensor.New(p, []int{K}, device.CPU()); err != nil {
		return Int4{}, err
	}
	return w, nil
}

// awqOrder is AutoAWQ's packing order: nibble i of a word holds column
// 8c + awqOrder[i].
var awqOrder = [8]int{0, 2, 4, 6, 1, 3, 5, 7}

// FromAWQ converts an AutoAWQ GEMM-version 4-bit linear: qweight I32
// (K, N/8) and qzeros I32 (K/G, N/8), both packed along N in awqOrder,
// scales (K/G, N) cast to F32 by the caller; W = s · (q − z).
func FromAWQ(qweight, qzeros, scales *tensor.Tensor, group int) (Int4, error) {
	if qweight.Rank() != 2 || qzeros.Rank() != 2 || scales.Rank() != 2 {
		return Int4{}, fmt.Errorf("quant.FromAWQ: rank-2 qweight / qzeros / scales expected")
	}
	K, N := qweight.Dim(0), qweight.Dim(1)*8
	if group <= 0 || K%group != 0 || scales.Dim(0) != K/group || scales.Dim(1) != N || qzeros.Dim(0) != K/group || qzeros.Dim(1)*8 != N {
		return Int4{}, fmt.Errorf("quant.FromAWQ: qweight %v, qzeros %v, scales %v, group %d inconsistent",
			qweight.Shape(), qzeros.Shape(), scales.Shape(), group)
	}
	qw, err := u32s(qweight, "qweight")
	if err != nil {
		return Int4{}, err
	}
	qz, err := u32s(qzeros, "qzeros")
	if err != nil {
		return Int4{}, err
	}
	sc, err := f32s(scales)
	if err != nil {
		return Int4{}, err
	}
	var inv [8]int // column offset → nibble index
	for i, o := range awqOrder {
		inv[o] = i
	}
	G := K / group
	scT, zT := make([]float32, N*G), make([]float32, N*G)
	for g := range G {
		for n := range N {
			scT[n*G+g] = sc[g*N+n]
			zT[n*G+g] = float32(nibble(qz[g*(N/8)+n/8], inv[n%8]))
		}
	}
	return build(N, K, group, func(n, k int) uint8 { return nibble(qw[k*(N/8)+n/8], inv[n%8]) }, scT, zT)
}

// Dequant expands w to an F32 (N, K) host slice (the reference the kernels
// are tested against).
func (w Int4) Dequant() ([]float32, error) {
	codes, err := u32s(w.Codes, "codes")
	if err != nil {
		return nil, err
	}
	sc, err := f32s(w.Scales)
	if err != nil {
		return nil, err
	}
	zr, err := f32s(w.Zeros)
	if err != nil {
		return nil, err
	}
	N, K, G := w.N(), w.K(), w.K()/w.Group
	out := make([]float32, N*K)
	for n := range N {
		for k := range K {
			g := n*G + k/w.Group
			out[n*K+k] = sc[g] * (float32(nibble(codes[n*(K/8)+k/8], k%8)) - zr[g])
		}
	}
	return out, nil
}
