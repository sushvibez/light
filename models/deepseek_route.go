package models

import (
	"fmt"
	"math"
	"sort"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// deepseekRouter returns the expert selection of DeepSeek's MoE routers
// that softmax top-k does not cover (transformers' DeepseekV2TopkRouter
// "group_limited_greedy" and DeepseekV3TopkRouter):
//
//   - V2 group-limited: scores = softmax(logits); keep the topkGroup
//     groups with the largest maximum score, zero the rest; top-k of the
//     masked scores, weighted by them (no renormalisation).
//   - V3 ("sigmoid", bias given): scores = sigmoid(logits); choice =
//     scores + e_score_correction_bias; keep the topkGroup groups with the
//     largest sum of their two best choices, −inf the rest; top-k of the
//     choices, weighted by the unbiased scores, renormalised (+1e-20) when
//     normTopK.
//
// routed_scaling_factor is applied by the model after the experts. The
// selection runs on the host: one (T, E) F32 copy per MoE layer, fine for
// correctness; a fused kernel can replace it.
func deepseekRouter(topk, nGroup, topkGroup int, sigmoid bool, bias []float32, normTopK bool) func(*tensor.Tensor) (*ops.MoERouting, error) {
	return func(logits *tensor.Tensor) (*ops.MoERouting, error) {
		dev := logits.Device()
		host, err := logits.To(device.CPU())
		if err != nil {
			return nil, err
		}
		if host.DType() != dtype.F32 {
			return nil, fmt.Errorf("deepseek router: logits %v, want F32", host.DType())
		}
		if host, err = host.Contiguous(); err != nil {
			return nil, err
		}
		lg := tensor.MustSlice[float32](host)
		T, E := host.Dim(0), host.Dim(1)
		if nGroup < 1 {
			nGroup = 1
		}
		if E%nGroup != 0 {
			return nil, fmt.Errorf("deepseek router: %d experts not divisible into %d groups", E, nGroup)
		}
		per := E / nGroup
		ids := make([]int64, 0, T*topk)
		ws := make([]float32, 0, T*topk)
		scores := make([]float64, E)
		choice := make([]float64, E)
		for t := range T {
			row := lg[t*E : (t+1)*E]
			if sigmoid {
				for e, v := range row {
					scores[e] = 1 / (1 + math.Exp(-float64(v)))
					choice[e] = scores[e]
					if bias != nil {
						choice[e] += float64(bias[e])
					}
				}
			} else {
				mx := math.Inf(-1)
				for _, v := range row {
					mx = math.Max(mx, float64(v))
				}
				var sum float64
				for e, v := range row {
					scores[e] = math.Exp(float64(v) - mx)
					sum += scores[e]
				}
				for e := range scores {
					scores[e] /= sum
					choice[e] = scores[e]
				}
			}
			if nGroup > 1 && topkGroup < nGroup {
				gs := make([]float64, nGroup)
				for g := range nGroup {
					grp := append([]float64(nil), choice[g*per:(g+1)*per]...)
					sort.Sort(sort.Reverse(sort.Float64Slice(grp)))
					if sigmoid { // V3: sum of the two best
						gs[g] = grp[0]
						if per > 1 {
							gs[g] += grp[1]
						}
					} else { // V2: the best
						gs[g] = grp[0]
					}
				}
				order := make([]int, nGroup)
				for i := range order {
					order[i] = i
				}
				sort.SliceStable(order, func(a, b int) bool { return gs[order[a]] > gs[order[b]] })
				keep := make([]bool, nGroup)
				for _, g := range order[:topkGroup] {
					keep[g] = true
				}
				for e := range choice {
					if !keep[e/per] {
						if sigmoid {
							choice[e] = math.Inf(-1)
						} else {
							choice[e], scores[e] = 0, 0 // masked_fill(0.0) on the scores themselves
						}
					}
				}
			}
			order := make([]int, E)
			for i := range order {
				order[i] = i
			}
			sort.SliceStable(order, func(a, b int) bool { return choice[order[a]] > choice[order[b]] })
			var den float64
			for _, e := range order[:topk] {
				den += scores[e]
			}
			for _, e := range order[:topk] {
				w := scores[e]
				if sigmoid && normTopK {
					w /= den + 1e-20
				}
				ids = append(ids, int64(e))
				ws = append(ws, float32(w))
			}
		}
		return ops.MoERouteFromIDs(ids, ws, T, topk, dev)
	}
}
