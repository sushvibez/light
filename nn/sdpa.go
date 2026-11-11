package nn

import (
	"fmt"
	"math"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// SDPA computes scaled dot-product attention
//
//	softmax(Q Kᵀ / √D + addMask) V
//
// for q (B, NH, Sq, D) and k, v (B, NH, Sk, D). addMask is an ADDITIVE
// bias (0 = attend, -Inf / large negative = masked) broadcastable to
// (B, NH, Sq, Sk): (Sq, Sk), (NH, Sq, Sk), (B, 1, Sq, Sk) or the full shape.
// nil means no mask. The result is (B, NH, Sq, D).
//
// It is the reusable building block for causal, sliding-window and ALiBi
// attention — each is just a different additive mask (see CausalMask,
// SlidingWindowMask, ALiBiBias). Device-agnostic: everything is expressed
// through autograd Reshape/Transpose/MatMul/Mul/Add/Softmax, so it runs on
// CPU and CUDA and participates in backward.
func SDPA(q, k, v, addMask *tensor.Tensor) (*tensor.Tensor, error) {
	return SDPAScaled(q, k, v, addMask, 0)
}

// SDPAScaled is SDPA with an explicit score scale: softmax(scale·q·kᵀ +
// mask)·v. scale 0 means 1/√D; scale 1 skips the multiply (T5 does not
// scale its scores).
func SDPAScaled(q, k, v, addMask *tensor.Tensor, scale float64) (*tensor.Tensor, error) {
	if q.Rank() != 4 || k.Rank() != 4 || v.Rank() != 4 {
		return nil, fmt.Errorf("nn.SDPA: q/k/v must be (B, NH, S, D); got %v, %v, %v", q.Shape(), k.Shape(), v.Shape())
	}
	B, NH, Sq, D := q.Dim(0), q.Dim(1), q.Dim(2), q.Dim(3)
	Sk := k.Dim(2)
	if k.Dim(0) != B || k.Dim(1) != NH || k.Dim(3) != D {
		return nil, fmt.Errorf("nn.SDPA: k shape %v incompatible with q %v", k.Shape(), q.Shape())
	}
	if !v.Shape().Equal(k.Shape()) {
		return nil, fmt.Errorf("nn.SDPA: v shape %v must equal k shape %v", v.Shape(), k.Shape())
	}
	if q.DType() != k.DType() || q.DType() != v.DType() {
		return nil, fmt.Errorf("nn.SDPA: dtype mismatch")
	}
	if !q.Device().SameAs(k.Device()) || !q.Device().SameAs(v.Device()) {
		return nil, fmt.Errorf("nn.SDPA: device mismatch")
	}
	if addMask != nil {
		if !addMask.Device().SameAs(q.Device()) {
			return nil, fmt.Errorf("nn.SDPA: mask on %v, q on %v", addMask.Device(), q.Device())
		}
		if addMask.DType() != q.DType() {
			return nil, fmt.Errorf("nn.SDPA: mask dtype %v != q dtype %v", addMask.DType(), q.DType())
		}
	}

	flat := func(t *tensor.Tensor, S int) (*tensor.Tensor, error) {
		c, err := autograd.Contiguous(t)
		if err != nil {
			return nil, err
		}
		return autograd.Reshape(c, B*NH, S, D)
	}
	qf, err := flat(q, Sq)
	if err != nil {
		return nil, err
	}
	kf, err := flat(k, Sk)
	if err != nil {
		return nil, err
	}
	vf, err := flat(v, Sk)
	if err != nil {
		return nil, err
	}
	kT, err := autograd.Transpose(kf, 1, 2)
	if err != nil {
		return nil, err
	}
	kTc, err := autograd.Contiguous(kT)
	if err != nil {
		return nil, err
	}
	scores, err := autograd.MatMul(qf, kTc) // (B*NH, Sq, Sk)
	if err != nil {
		return nil, err
	}
	scaled := scores
	if scale != 1 {
		var sc *tensor.Tensor
		if scale == 0 {
			sc, err = attentionScale(D, q.DType(), q.Device())
		} else {
			sc, err = scalarLike(scale, q.DType(), q.Device())
		}
		if err != nil {
			return nil, err
		}
		if scaled, err = autograd.Mul(scores, sc); err != nil {
			return nil, err
		}
	}
	// Un-collapse to (B, NH, Sq, Sk) so masks can broadcast per batch/head.
	scaled4, err := autograd.Reshape(scaled, B, NH, Sq, Sk)
	if err != nil {
		return nil, err
	}
	if addMask != nil {
		if scaled4, err = autograd.Add(scaled4, addMask); err != nil {
			return nil, fmt.Errorf("nn.SDPA: mask %v not broadcastable to scores (B=%d, NH=%d, Sq=%d, Sk=%d): %w", addMask.Shape(), B, NH, Sq, Sk, err)
		}
	}
	attn, err := autograd.Softmax(scaled4, -1)
	if err != nil {
		return nil, err
	}
	attn3, err := autograd.Reshape(attn, B*NH, Sq, Sk)
	if err != nil {
		return nil, err
	}
	ctx, err := autograd.MatMul(attn3, vf) // (B*NH, Sq, D)
	if err != nil {
		return nil, err
	}
	return autograd.Reshape(ctx, B, NH, Sq, D)
}

// attentionScale returns a (1,) tensor holding 1/√D in the given dtype.
func attentionScale(D int, dt dtype.DType, dev device.Device) (*tensor.Tensor, error) {
	s, err := tensor.New([]float32{float32(1.0 / math.Sqrt(float64(D)))}, []int{1}, dev)
	if err != nil {
		return nil, err
	}
	if dt == dtype.F32 {
		return s, nil
	}
	return ops.Cast(s, dt)
}

// CausalMask returns an (S, S) F32 additive mask: 0 where key j <= query i,
// -Inf where j > i.
func CausalMask(S int, dev device.Device) (*tensor.Tensor, error) {
	if S <= 0 {
		return nil, fmt.Errorf("nn.CausalMask: S must be positive (got %d)", S)
	}
	negInf := float32(math.Inf(-1))
	data := make([]float32, S*S)
	for i := range S {
		for j := i + 1; j < S; j++ {
			data[i*S+j] = negInf
		}
	}
	return tensor.New(data, []int{S, S}, dev)
}

// SlidingWindowMask returns a (qLen, kLen) F32 additive causal mask that
// also forbids attending further back than `window` keys (Mistral-style
// SWA): query i (at absolute position i + kLen - qLen, so a KV-cache
// prefix is right-aligned) may attend key j iff pos-window < j <= pos.
// 0 where attended, -Inf elsewhere.
func SlidingWindowMask(qLen, kLen, window int, dev device.Device) (*tensor.Tensor, error) {
	if qLen <= 0 || kLen <= 0 || window <= 0 {
		return nil, fmt.Errorf("nn.SlidingWindowMask: qLen, kLen and window must be positive")
	}
	if kLen < qLen {
		return nil, fmt.Errorf("nn.SlidingWindowMask: kLen (%d) must be >= qLen (%d)", kLen, qLen)
	}
	negInf := float32(math.Inf(-1))
	off := kLen - qLen
	data := make([]float32, qLen*kLen)
	for i := range qLen {
		pos := i + off
		for j := range kLen {
			if j > pos || j <= pos-window {
				data[i*kLen+j] = negInf
			}
		}
	}
	return tensor.New(data, []int{qLen, kLen}, dev)
}

// ALiBiSlopes returns the per-head ALiBi slopes from Press et al. (2022):
// for a power-of-two head count n the geometric sequence
// 2^(-8/n), 2^(-16/n), …; otherwise the sequence for the largest power of
// two below n, extended with every other slope of the 2×-larger sequence
// (the "get_slopes" recipe used by BLOOM / Falcon).
func ALiBiSlopes(numHeads int) []float64 {
	if numHeads <= 0 {
		return nil
	}
	pow2 := func(n int) []float64 {
		base := math.Pow(2, -8.0/float64(n))
		s := make([]float64, n)
		for i := range s {
			s[i] = math.Pow(base, float64(i+1))
		}
		return s
	}
	if numHeads&(numHeads-1) == 0 {
		return pow2(numHeads)
	}
	closest := 1
	for closest*2 <= numHeads {
		closest *= 2
	}
	slopes := pow2(closest)
	extra := pow2(2 * closest)
	for i := 0; len(slopes) < numHeads; i += 2 {
		slopes = append(slopes, extra[i])
	}
	return slopes
}

// ALiBiBias returns the (numHeads, qLen, kLen) F32 additive ALiBi bias:
// bias[h, i, j] = -slope[h] * (pos_i - j) for j <= pos_i (pos_i = i + kLen
// - qLen), -Inf for j > pos_i (causal). It replaces positional embeddings;
// feed it to SDPA as the mask (it broadcasts over the batch axis).
func ALiBiBias(numHeads, qLen, kLen int, dev device.Device) (*tensor.Tensor, error) {
	if numHeads <= 0 || qLen <= 0 || kLen <= 0 {
		return nil, fmt.Errorf("nn.ALiBiBias: numHeads, qLen and kLen must be positive")
	}
	if kLen < qLen {
		return nil, fmt.Errorf("nn.ALiBiBias: kLen (%d) must be >= qLen (%d)", kLen, qLen)
	}
	slopes := ALiBiSlopes(numHeads)
	negInf := float32(math.Inf(-1))
	off := kLen - qLen
	data := make([]float32, numHeads*qLen*kLen)
	for h := range numHeads {
		for i := range qLen {
			pos := i + off
			row := data[(h*qLen+i)*kLen : (h*qLen+i+1)*kLen]
			for j := range kLen {
				if j > pos {
					row[j] = negInf
				} else {
					row[j] = float32(-slopes[h] * float64(pos-j))
				}
			}
		}
	}
	return tensor.New(data, []int{numHeads, qLen, kLen}, dev)
}

// scalarLike returns a (1,) tensor holding v in the given dtype.
func scalarLike(v float64, dt dtype.DType, dev device.Device) (*tensor.Tensor, error) {
	s, err := tensor.New([]float32{float32(v)}, []int{1}, dev)
	if err != nil {
		return nil, err
	}
	if dt == dtype.F32 {
		return s, nil
	}
	return ops.Cast(s, dt)
}
