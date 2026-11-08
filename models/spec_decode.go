package models

import (
	"fmt"

	"light/device"
	"light/tensor"
)

// SpecStats counts speculative decoding outcomes.
type SpecStats struct{ Drafted, Accepted int }

// SpecGreedy decodes up to maxNew tokens greedily after prompt, using the
// model's MTP head as a one-token draft: after choosing token x it drafts d,
// then runs the main model on [x, d] in one step. If the main model's
// choice after x is d, both are kept (and the step also yields the token
// after d); otherwise only x is kept and the main cache drops d's position.
// Every emitted token is the main model's argmax given the emitted prefix,
// so the output equals plain greedy decoding exactly; the draft head only
// changes how many forwards it takes. emit is called per token and may
// stop early by returning false (an EOS, a cancelled request). The MTP
// cache only ever receives accepted (hidden, next-token) pairs, so it needs
// no rollback.
func SpecGreedy(m *Llama, prompt []int64, maxNew int, emit func(int64) bool) (SpecStats, error) {
	if m.MTP == nil {
		return SpecStats{}, fmt.Errorf("SpecGreedy: model has no MTP head")
	}
	mtp := NewKVCache(1)
	defer mtp.Reset()
	return specGreedy(m, prompt, maxNew, emit, func(hidden *tensor.Tensor, next []int64) (int64, error) {
		dl, err := m.MTPDraft(hidden, next, mtp)
		if err != nil {
			return 0, err
		}
		dr, err := rows(dl)
		if err != nil {
			return 0, err
		}
		return argmaxRow(dr[len(dr)-1]), nil
	})
}

// specGreedy is SpecGreedy with the drafter as a parameter: draft receives
// each new run of (hidden state, known next token) pairs, in stream order
// and exactly once, and returns its guess for the token after the last one.
func specGreedy(m *Llama, prompt []int64, maxNew int, emit func(int64) bool,
	draft func(hidden *tensor.Tensor, next []int64) (int64, error)) (SpecStats, error) {
	var st SpecStats
	if len(prompt) == 0 || maxNew <= 0 {
		return st, nil
	}
	dev := m.TokenEmb.Weight.Device()
	main := NewKVCache(m.NumLayers())
	defer main.Reset()

	run := func(tokens []int64) ([][]float32, *tensor.Tensor, error) {
		x, err := tensor.New(tokens, []int{1, len(tokens)}, dev)
		if err != nil {
			return nil, nil, err
		}
		logits, err := m.ForwardWithCache(x, main)
		if err != nil {
			return nil, nil, err
		}
		rs, err := rows(logits)
		return rs, m.LastHidden, err
	}
	// Prefill: logits for the first new token and the hidden states that
	// feed the draft head's cache (pair (h_i, prompt[i+1]) for each i).
	r, hid, err := run(prompt)
	if err != nil {
		return st, err
	}
	x := argmaxRow(r[len(r)-1])
	pendHidden := hid                                          // hidden states whose next tokens are known
	pendTokens := append(prompt[1:len(prompt):len(prompt)], x) // their next tokens
	emitted := 0
	for emitted < maxNew {
		if !emit(x) {
			return st, nil
		}
		emitted++
		if emitted == maxNew {
			return st, nil
		}
		// Draft: feed the pending pairs, read the last row.
		d, err := draft(pendHidden, pendTokens)
		if err != nil {
			return st, err
		}
		st.Drafted++
		// Verify [x, d].
		r, hid, err = run([]int64{x, d})
		if err != nil {
			return st, err
		}
		after := argmaxRow(r[0])
		if after == d {
			st.Accepted++
			if !emit(d) {
				return st, nil
			}
			emitted++
			if emitted == maxNew {
				return st, nil
			}
			next := argmaxRow(r[1])
			pendHidden, pendTokens = hid, []int64{d, next} // (h_x, d), (h_d, next)
			x = next
			continue
		}
		// Rejected: keep x only.
		if err := main.Truncate(main.Len - 1); err != nil {
			return st, err
		}
		h0, err := hid.Narrow(1, 0, 1)
		if err != nil {
			return st, err
		}
		pendHidden, pendTokens = h0, []int64{after} // (h_x, after)
		x = after
	}
	return st, nil
}

// rows returns the (T, V) rows of F32 logits (1, T, V) on the host.
func rows(logits *tensor.Tensor) ([][]float32, error) {
	cpu, err := logits.To(device.CPU())
	if err != nil {
		return nil, err
	}
	if cpu, err = cpu.Contiguous(); err != nil {
		return nil, err
	}
	flat, err := tensor.ToSlice[float32](cpu)
	if err != nil {
		return nil, err
	}
	T, V := logits.Dim(1), logits.Dim(2)
	out := make([][]float32, T)
	for i := range out {
		out[i] = flat[i*V : (i+1)*V]
	}
	return out, nil
}

func argmaxRow(row []float32) int64 { return int64(argmax(row)) }
