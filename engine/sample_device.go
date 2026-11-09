package engine

import (
	"math"
	"math/rand/v2"

	"light/device"
	"light/internal/cudart"
	"light/models"
	"light/ops"
	"light/tensor"
)

// stepLast runs tokens through the model and returns the last position's
// logits as a (1, V) tensor still on the model's device, so sampling can
// stay there (greedy: one argmax; top-k: one fused kernel) instead of
// copying a 128k–250k-entry F32 row to the host every token.
func (m *Model) stepLast(tokens []int64, cache *models.KVCache) (*tensor.Tensor, error) {
	x, err := tensor.New(tokens, []int{1, len(tokens)}, m.Dev)
	if err != nil {
		return nil, err
	}
	logits, err := m.LM.ForwardWithCache(x, cache)
	if err != nil {
		return nil, err
	}
	V := m.LM.VocabSize()
	flat, err := logits.Reshape(logits.NumElements()/V, V)
	if err != nil {
		return nil, err
	}
	row, err := flat.Narrow(0, flat.Dim(0)-1, 1)
	if err != nil {
		return nil, err
	}
	return row.Contiguous() // a no-op for a single decode token (offset 0)
}

// sampleDevice draws the next token from logits (1, V) on the device when
// the sampler's result does not depend on the whole row: greedy (argmax)
// or top-k without top-p (the k best logits, softmax over them on the
// host). Otherwise it reads the row back and uses models.SampleLogits,
// which is also the CPU path. Equivalent to SampleLogits in all cases:
// softmax over the top-k set then sampling within it is the same
// distribution as normalising over the full vocabulary and keeping the
// top-k.
func (m *Model) sampleDevice(logits *tensor.Tensor, opts models.SampleOpts) (int, error) {
	V := logits.Dim(-1)
	onCUDA := device.IsCUDA(logits.Device())
	switch {
	case onCUDA && opts.Temperature == 0:
		am, err := ops.Argmax(logits, -1, false)
		if err != nil {
			return 0, err
		}
		cpu, err := am.To(device.CPU())
		if err != nil {
			return 0, err
		}
		return int(tensor.MustSlice[int64](cpu)[0]), nil
	case onCUDA && opts.TopK > 0 && opts.TopK < V && opts.TopK <= 64 && (opts.TopP <= 0 || opts.TopP >= 1):
		vals, idx, err := ops.TopK(logits, opts.TopK, true)
		if err != nil {
			return 0, err
		}
		vc, err := vals.To(device.CPU())
		if err != nil {
			return 0, err
		}
		ic, err := idx.To(device.CPU())
		if err != nil {
			return 0, err
		}
		v, ids := tensor.MustSlice[float32](vc), tensor.MustSlice[int64](ic)
		rng := opts.Rng
		if rng == nil {
			rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
		}
		invT := 1 / float64(opts.Temperature)
		probs := make([]float64, len(v))
		var sum float64
		for i, x := range v { // v[0] is the max
			probs[i] = math.Exp(float64(x-v[0]) * invT)
			sum += probs[i]
		}
		if sum == 0 || math.IsNaN(sum) {
			return int(ids[0]), nil
		}
		r := rng.Float64() * sum
		for i, p := range probs {
			r -= p
			if r <= 0 {
				return int(ids[i]), nil
			}
		}
		return int(ids[len(ids)-1]), nil
	}
	cpu, err := logits.To(device.CPU())
	if err != nil {
		return 0, err
	}
	return models.SampleLogits(tensor.MustSlice[float32](cpu), opts), nil
}

// sync waits for the device's queued work (a no-op on the CPU).
func (m *Model) sync() error {
	if device.IsCUDA(m.Dev) {
		return cudart.DeviceSynchronize()
	}
	return nil
}
