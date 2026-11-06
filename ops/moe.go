package ops

import (
	"fmt"

	"light/dtype"
	"light/internal/registry"
	"light/quant"
	"light/tensor"
)

// MoERouting is the output of MoERoute: the per-pair routing weights and
// the expert-sorted pair order that MoEGemm consumes. With T tokens and
// topk experts per token there are M = T·topk (token, slot) pairs; pair
// p = token·topk + slot.
type MoERouting struct {
	TopKIDs     *tensor.Tensor // (T, topk) I64 expert per slot
	Weights     *tensor.Tensor // (M,) F32 routing weight per pair
	SortedPairs *tensor.Tensor // (M,) I64 pair ids in expert order
	SortedExpts *tensor.Tensor // (M,) I64 expert of each sorted pair (non-decreasing)
}

// MoERoute turns router probabilities (T, E) into a top-k routing: the k
// largest experts per token, their probabilities (renormalised to sum to 1
// per token when normalize is set — `norm_topk_prob`), and the pairs sorted
// by expert for the grouped GEMM. Everything runs on probs' device; probs
// must be F32.
func MoERoute(probs *tensor.Tensor, topk int, normalize bool) (*MoERouting, error) {
	if probs.Rank() != 2 || probs.DType() != dtype.F32 {
		return nil, fmt.Errorf("moe_route: probs must be (T, E) F32, got %v %v", probs.Shape(), probs.DType())
	}
	T, E := probs.Dim(0), probs.Dim(1)
	if topk < 1 || topk > E {
		return nil, fmt.Errorf("moe_route: topk %d out of range [1, %d]", topk, E)
	}
	vals, ids, err := TopK(probs, topk, true)
	if err != nil {
		return nil, err
	}
	if normalize {
		den, err := Sum(vals, ReduceOptions{Axes: []int{1}, KeepDims: true})
		if err != nil {
			return nil, err
		}
		if vals, err = Div(vals, den); err != nil {
			return nil, err
		}
	}
	valsC, err := ensureContig(vals)
	if err != nil {
		return nil, err
	}
	weights, err := valsC.Reshape(T * topk)
	if err != nil {
		return nil, err
	}
	idsC, err := ensureContig(ids)
	if err != nil {
		return nil, err
	}
	flat, err := idsC.Reshape(T * topk)
	if err != nil {
		return nil, err
	}
	sortedExpts, sortedPairs, err := Sort(flat, false)
	if err != nil {
		return nil, err
	}
	return &MoERouting{TopKIDs: idsC, Weights: weights, SortedPairs: sortedPairs, SortedExpts: sortedExpts}, nil
}

// MoEGemm is the grouped expert matmul (Candle `moe_gemm`):
//
//	out[p, :] = w(p) · input[row(p), :] @ weights[expert(p), :, :]ᵀ
//
// for every routed pair p, with weights (E, N, K) stacked per expert,
// routing from MoERoute (SortedPairs / SortedExpts), w(p) = topkWeights[p]
// (F32, (M,)) or 1 when nil, and row(p) = p / topk when input has one row
// per token (T, K) or p when it has one row per pair (M, K). Output is (M,
// N) at input's dtype; the per-token sum over the topk slots is left to the
// caller (reshape to (T, topk, N) and add the slots). CUDA runs one tiled
// kernel over all experts; the CPU kernel is the reference. F32/BF16/F16.
func MoEGemm(input, weights *tensor.Tensor, routing *MoERouting, topkWeights *tensor.Tensor, topk int) (*tensor.Tensor, error) {
	if input.Rank() != 2 || weights.Rank() != 3 {
		return nil, fmt.Errorf("moe_gemm: input (rows, K) and weights (E, N, K) required, got %v / %v", input.Shape(), weights.Shape())
	}
	if input.DType() != weights.DType() {
		return nil, fmt.Errorf("moe_gemm: input %v / weights %v dtype mismatch", input.DType(), weights.DType())
	}
	return moeGemmDispatch("moe_gemm", input.DType(), input, weights, routing, topkWeights, topk)
}

// moeGemmDispatch validates the routing contract and runs the registered
// kernel `op` (keyed by keyDT) on a fresh (M, N) output at input's dtype.
func moeGemmDispatch(op string, keyDT dtype.DType, input, weights *tensor.Tensor, routing *MoERouting, topkWeights *tensor.Tensor, topk int) (*tensor.Tensor, error) {
	if input.Rank() != 2 || weights.Rank() != 3 {
		return nil, fmt.Errorf("%s: input (rows, K) and weights (E, N, K) required, got %v / %v", op, input.Shape(), weights.Shape())
	}
	if !input.Device().SameAs(weights.Device()) || !input.Device().SameAs(routing.SortedPairs.Device()) {
		return nil, fmt.Errorf("%s: device mismatch", op)
	}
	E, N, K := weights.Dim(0), weights.Dim(1), weights.Dim(2)
	if input.Dim(1) != K {
		return nil, fmt.Errorf("%s: input K %d != weights K %d", op, input.Dim(1), K)
	}
	M := routing.SortedPairs.Dim(0)
	if routing.SortedExpts.Dim(0) != M || routing.SortedPairs.DType() != dtype.I64 || routing.SortedExpts.DType() != dtype.I64 {
		return nil, fmt.Errorf("%s: routing must hold M I64 pair ids and M I64 expert ids", op)
	}
	var rowPerPair bool
	switch input.Dim(0) {
	case M:
		rowPerPair = true
	case M / topk:
		rowPerPair = false
	default:
		return nil, fmt.Errorf("%s: input rows %d must be M=%d (per pair) or M/topk=%d (per token)", op, input.Dim(0), M, M/topk)
	}
	if topkWeights != nil {
		if topkWeights.DType() != dtype.F32 || topkWeights.NumElements() != M {
			return nil, fmt.Errorf("%s: topkWeights must be F32 with %d elements", op, M)
		}
		if !topkWeights.Device().SameAs(input.Device()) {
			return nil, fmt.Errorf("%s: topkWeights device mismatch", op)
		}
	}
	inC, err := ensureContig(input)
	if err != nil {
		return nil, err
	}
	wC, err := ensureContig(weights)
	if err != nil {
		return nil, err
	}
	pairsC, err := ensureContig(routing.SortedPairs)
	if err != nil {
		return nil, err
	}
	exptsC, err := ensureContig(routing.SortedExpts)
	if err != nil {
		return nil, err
	}
	counts, err := tensor.Zeros(dtype.I32, []int{E}, input.Device())
	if err != nil {
		return nil, err
	}
	offsets, err := tensor.Zeros(dtype.I32, []int{E + 1}, input.Device())
	if err != nil {
		return nil, err
	}
	inputs := []*tensor.Tensor{inC, wC, pairsC, exptsC, counts, offsets}
	if topkWeights != nil {
		tw, err := ensureContig(topkWeights)
		if err != nil {
			return nil, err
		}
		inputs = append(inputs, tw)
	}
	out, err := tensor.Zeros(input.DType(), []int{M, N}, input.Device())
	if err != nil {
		return nil, err
	}
	k, err := registry.Lookup(registry.Key{Op: op, Dev: input.Device().Kind(), DType: keyDT})
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{
		Inputs: inputs,
		Output: out,
		Attrs:  map[string]any{"topk": topk, "row_per_pair": rowPerPair},
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// MoECombine sums the topk slot rows of a (M, N) pair-indexed tensor into
// (T, N) using narrow + add, so it works at every dtype the Add kernels
// support (BF16 reductions are not available on every device).
func MoECombine(pairOut *tensor.Tensor, topk int) (*tensor.Tensor, error) {
	M, N := pairOut.Dim(0), pairOut.Dim(1)
	if M%topk != 0 {
		return nil, fmt.Errorf("moe_combine: %d rows not divisible by topk %d", M, topk)
	}
	src, err := ensureContig(pairOut)
	if err != nil {
		return nil, err
	}
	v, err := src.Reshape(M/topk, topk, N)
	if err != nil {
		return nil, err
	}
	acc, err := v.Narrow(1, 0, 1)
	if err != nil {
		return nil, err
	}
	for s := 1; s < topk; s++ {
		slot, err := v.Narrow(1, s, 1)
		if err != nil {
			return nil, err
		}
		if acc, err = Add(acc, slot); err != nil {
			return nil, err
		}
	}
	accC, err := ensureContig(acc)
	if err != nil {
		return nil, err
	}
	return accC.Reshape(M/topk, N)
}

// MoEGemmQ is MoEGemm with GGUF block-quantized expert weights: x is F32
// (rows, K), w is (E, N, K) in a quantized dtype, output F32 (M, N).
// Q4_0 / Q5_0 / Q8_0 / Q4_K / Q5_K / Q6_K have a fused CUDA kernel
// (Candle `moe_gemm_gguf`); other formats and the CPU dequantize the whole
// expert stack to F32 and run MoEGemm, which is also the parity reference.
func MoEGemmQ(x, w *tensor.Tensor, routing *MoERouting, topkWeights *tensor.Tensor, topk int) (*tensor.Tensor, error) {
	if x.DType() != dtype.F32 {
		return nil, fmt.Errorf("moe_gemm_q: x must be F32, got %v", x.DType())
	}
	if !w.DType().IsQuantized() || w.Rank() != 3 {
		return nil, fmt.Errorf("moe_gemm_q: w must be a quantized (E, N, K) tensor, got %v %v", w.DType(), w.Shape())
	}
	if w.Dim(2)%w.DType().BlockElems() != 0 {
		return nil, fmt.Errorf("moe_gemm_q: K=%d not a multiple of %v block size %d", w.Dim(2), w.DType(), w.DType().BlockElems())
	}
	if _, err := registry.Lookup(registry.Key{Op: "moe_gemm_q", Dev: x.Device().Kind(), DType: w.DType()}); err != nil {
		wf, err := quant.Dequantize(w)
		if err != nil {
			return nil, err
		}
		return MoEGemm(x, wf, routing, topkWeights, topk)
	}
	return moeGemmDispatch("moe_gemm_q", w.DType(), x, w, routing, topkWeights, topk)
}
