package quant

import (
	"fmt"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// MLX affine quantization (mlx.core.quantize, mode "affine"), the format of
// mlx-lm / mlx-community checkpoints. For a (N, K) weight with groups of G
// along K and b bits:
//
//	weight U32 (N, K·b/32)  field j (b bits, low bits first) of word c holds
//	                         the code of W[n, c·32/b + j]
//	scales (N, K/G), biases (N, K/G)  usually BF16
//	W[n, k] = scales[n, k/G] · q(n, k) + biases[n, k/G]
//
// Stacked MoE experts (switch_mlp) add a leading expert axis. Light reads
// b ∈ {2, 4, 8} (the fields then never straddle a word).

// mlxShape checks a rank-2 MLX weight against its scales / biases and
// returns N, K.
func mlxShape(w, scales, biases *tensor.Tensor, group, bits int) (int, int, error) {
	if bits != 2 && bits != 4 && bits != 8 {
		return 0, 0, fmt.Errorf("quant: MLX %d-bit weights not supported (2, 4, 8)", bits)
	}
	if w.Rank() != 2 || scales.Rank() != 2 || biases.Rank() != 2 || group <= 0 {
		return 0, 0, fmt.Errorf("quant: MLX weight %v, scales %v, biases %v, group %d: rank-2 expected",
			w.Shape(), scales.Shape(), biases.Shape(), group)
	}
	N, K := w.Dim(0), w.Dim(1)*32/bits
	if K%group != 0 || group%(32/bits) != 0 || scales.Dim(0) != N || scales.Dim(1) != K/group || !scales.Shape().Equal(biases.Shape()) {
		return 0, 0, fmt.Errorf("quant: MLX weight %v (%d-bit), scales %v, biases %v, group %d inconsistent",
			w.Shape(), bits, scales.Shape(), biases.Shape(), group)
	}
	return N, K, nil
}

// hostF32 returns t as an F32 host slice (any float dtype).
func hostF32(t *tensor.Tensor) ([]float32, error) {
	c, err := t.To(device.CPU())
	if err != nil {
		return nil, err
	}
	if c, err = c.Contiguous(); err != nil {
		return nil, err
	}
	switch c.DType() {
	case dtype.F32:
		return tensor.MustSlice[float32](c), nil
	case dtype.BF16:
		v, err := tensor.ToBF16Slice(c)
		if err != nil {
			return nil, err
		}
		out := make([]float32, len(v))
		for i, x := range v {
			out[i] = x.ToF32()
		}
		return out, nil
	case dtype.F16:
		v, err := tensor.ToHalfSlice(c)
		if err != nil {
			return nil, err
		}
		out := make([]float32, len(v))
		for i, x := range v {
			out[i] = x.ToF32()
		}
		return out, nil
	}
	return nil, fmt.Errorf("quant: MLX scales / biases are %v, want F32 / BF16 / F16", c.DType())
}

// FromMLX converts a 4-bit MLX affine weight to Int4 losslessly in the
// codes: the packing is already Int4's, so Codes is w itself (no copy for
// a host tensor) and only the affine pair is rewritten, Zeros = −biases /
// scales (scales 0, never produced by mlx.quantize, maps to zero 0).
func FromMLX(w, scales, biases *tensor.Tensor, group int) (Int4, error) {
	N, K, err := mlxShape(w, scales, biases, group, 4)
	if err != nil {
		return Int4{}, err
	}
	if w.DType() != dtype.U32 {
		return Int4{}, fmt.Errorf("quant.FromMLX: weight is %v, want U32", w.DType())
	}
	sc, err := hostF32(scales)
	if err != nil {
		return Int4{}, err
	}
	bi, err := hostF32(biases)
	if err != nil {
		return Int4{}, err
	}
	s := append([]float32(nil), sc...)
	z := make([]float32, len(bi))
	for i, b := range bi {
		if s[i] != 0 {
			z[i] = -b / s[i]
		}
	}
	codes, err := w.Contiguous()
	if err != nil {
		return Int4{}, err
	}
	st, err := tensor.New(s, []int{N, K / group}, device.CPU())
	if err != nil {
		return Int4{}, err
	}
	zt, err := tensor.New(z, []int{N, K / group}, device.CPU())
	if err != nil {
		return Int4{}, err
	}
	return Int4{Codes: codes, Scales: st, Zeros: zt, Group: group}, nil
}

// DequantMLX expands an MLX affine weight (N, K·bits/32) to an F32 (N, K)
// host tensor, W = scales · q + biases formed in F32.
func DequantMLX(w, scales, biases *tensor.Tensor, group, bits int) (*tensor.Tensor, error) {
	N, K, err := mlxShape(w, scales, biases, group, bits)
	if err != nil {
		return nil, err
	}
	codes, err := u32s(w, "weight")
	if err != nil {
		return nil, err
	}
	sc, err := hostF32(scales)
	if err != nil {
		return nil, err
	}
	bi, err := hostF32(biases)
	if err != nil {
		return nil, err
	}
	out := make([]float32, N*K)
	DequantMLXRows(out, codes, sc, bi, N, K, group, bits)
	return tensor.New(out, []int{N, K}, device.CPU())
}

// DequantMLXRows is DequantMLX on host slices: codes (N·K·bits/32), scales
// and biases (N·K/G) into out (N·K).
func DequantMLXRows(out []float32, codes []uint32, scales, biases []float32, N, K, group, bits int) {
	per, mask, G := 32/bits, uint32(1)<<bits-1, K/group
	for n := range N {
		row := codes[n*K/per : (n+1)*K/per]
		o := out[n*K : (n+1)*K]
		for k := 0; k < K; k += per {
			word := row[k/per]
			g := n*G + k/group
			s, b := scales[g], biases[g]
			for j := range per {
				o[k+j] = s*float32(word>>(bits*j)&mask) + b
			}
		}
	}
}
