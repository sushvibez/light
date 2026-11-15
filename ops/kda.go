package ops

import (
	"fmt"
	"math"

	"light/device"
	"light/dtype"
	"light/internal/parallel"
	"light/tensor"
)

// KimiDeltaRule is Kimi delta attention's recurrence (fla's
// fused_recurrent_kda / transformers' recurrent_kimi_delta_attention after
// its q / k normalisation and q scaling): per batch and head a (Dk, Dv)
// state S evolves over the T steps as
//
//	S ← S ⊙ exp(g_t) (per key channel), δ = β_t·(v_t − k_tᵀS),
//	S ← S + k_t ⊗ δ, out_t = q_tᵀS.
//
// q, k (B, T, H, Dk), v (B, T, H, Dv), g (B, T, H, Dk) log decays, beta
// (B, T, H); state (B, H, Dk, Dv) or nil for zeros. Returns out (B, T, H, Dv)
// and the final state, F32. Runs on the host (other devices round-trip).
func KimiDeltaRule(q, k, v, g, beta, state *tensor.Tensor) (out, final *tensor.Tensor, err error) {
	if q.Rank() != 4 || k.Rank() != 4 || v.Rank() != 4 || g.Rank() != 4 || beta.Rank() != 3 {
		return nil, nil, fmt.Errorf("KimiDeltaRule: want q, k, v, g rank 4 and beta rank 3")
	}
	B, T, H, Dk := q.Dim(0), q.Dim(1), q.Dim(2), q.Dim(3)
	Dv := v.Dim(3)
	if !k.Shape().Equal(q.Shape()) || !g.Shape().Equal(q.Shape()) || v.Dim(0) != B || v.Dim(1) != T || v.Dim(2) != H ||
		beta.Dim(0) != B || beta.Dim(1) != T || beta.Dim(2) != H {
		return nil, nil, fmt.Errorf("KimiDeltaRule: shapes q %v k %v v %v g %v beta %v", q.Shape(), k.Shape(), v.Shape(), g.Shape(), beta.Shape())
	}
	if state != nil && (state.Rank() != 4 || state.Dim(0) != B || state.Dim(1) != H || state.Dim(2) != Dk || state.Dim(3) != Dv) {
		return nil, nil, fmt.Errorf("KimiDeltaRule: state %v, want (%d, %d, %d, %d)", state.Shape(), B, H, Dk, Dv)
	}
	dev := q.Device()
	host := func(t *tensor.Tensor) ([]float32, error) {
		c, err := t.To(device.CPU())
		if err != nil {
			return nil, err
		}
		if c.DType() != dtype.F32 {
			if c, err = Cast(c, dtype.F32); err != nil {
				return nil, err
			}
		}
		if c, err = ensureContig(c); err != nil {
			return nil, err
		}
		return tensor.MustSlice[float32](c), nil
	}
	qs, err := host(q)
	if err != nil {
		return nil, nil, err
	}
	ks, err := host(k)
	if err != nil {
		return nil, nil, err
	}
	vs, err := host(v)
	if err != nil {
		return nil, nil, err
	}
	gs, err := host(g)
	if err != nil {
		return nil, nil, err
	}
	bs, err := host(beta)
	if err != nil {
		return nil, nil, err
	}
	S := make([]float32, B*H*Dk*Dv)
	if state != nil {
		ss, err := host(state)
		if err != nil {
			return nil, nil, err
		}
		copy(S, ss)
	}
	o := make([]float32, B*T*H*Dv)
	parallel.For(B*H, 1, func(lo, hi int) {
		mem := make([]float32, Dv)
		for bh := lo; bh < hi; bh++ {
			b, h := bh/H, bh%H
			s := S[bh*Dk*Dv : (bh+1)*Dk*Dv]
			for t := range T {
				row := (b*T+t)*H + h
				qt, kt, gt := qs[row*Dk:(row+1)*Dk], ks[row*Dk:(row+1)*Dk], gs[row*Dk:(row+1)*Dk]
				vt := vs[row*Dv : (row+1)*Dv]
				bt := bs[row]
				for i := range Dk { // decay
					d := float32(math.Exp(float64(gt[i])))
					si := s[i*Dv : (i+1)*Dv]
					for j := range si {
						si[j] *= d
					}
				}
				for j := range mem { // kᵀS
					mem[j] = 0
				}
				for i, kv := range kt {
					si := s[i*Dv : (i+1)*Dv]
					for j, x := range si {
						mem[j] += kv * x
					}
				}
				for j := range mem { // δ
					mem[j] = (vt[j] - mem[j]) * bt
				}
				for i, kv := range kt { // S += k ⊗ δ
					si := s[i*Dv : (i+1)*Dv]
					for j := range si {
						si[j] += kv * mem[j]
					}
				}
				ot := o[row*Dv : (row+1)*Dv]
				for i, qv := range qt { // qᵀS
					si := s[i*Dv : (i+1)*Dv]
					for j, x := range si {
						ot[j] += qv * x
					}
				}
			}
		}
	})
	if out, err = tensor.New(o, []int{B, T, H, Dv}, dev); err != nil {
		return nil, nil, err
	}
	final, err = tensor.New(S, []int{B, H, Dk, Dv}, dev)
	return out, final, err
}
