package models

import (
	"light/device"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// moeHit is one token's routing to an expert: row tok of the layer's
// input, weighted w.
type moeHit struct {
	tok int
	w   float32
}

// moeOrder lists the experts that got tokens, ascending.
func moeOrder(byExpert [][]moeHit) []int {
	var out []int
	for e, hs := range byExpert {
		if len(hs) > 0 {
			out = append(out, e)
		}
	}
	return out
}

// moeRun adds each routed expert's output on its rows of x (T, D) into acc
// (nil: starts from the first expert's), scattered back with a one-hot
// (T, n) product (every device has MatMul; index_add is not on all).
// fwd(v, xs, wt) runs expert v on rows xs with routing weights wt (n, 1).
// Experts come from cache when set (loaded on demand, the next ones read
// while one runs), otherwise from resident.
func moeRun[V any](o *iop, dev device.Device, x *tensor.Tensor, byExpert [][]moeHit, acc *tensor.Tensor,
	cache *nn.ExpertCache[V], resident []V, fwd func(v V, xs, wt *tensor.Tensor) *tensor.Tensor) *tensor.Tensor {
	if o.err != nil {
		return nil
	}
	T := x.Dim(0)
	one := func(e int, v V) error {
		hs := byExpert[e]
		tok := make([]int64, len(hs))
		wv := make([]float32, len(hs))
		for i, h := range hs {
			tok[i], wv[i] = int64(h.tok), h.w
		}
		ti := o.do(func() (*tensor.Tensor, error) { return tensor.New(tok, []int{len(tok)}, dev) })
		wt := o.do(func() (*tensor.Tensor, error) { return tensor.New(wv, []int{len(wv), 1}, dev) })
		xs := o.do(func() (*tensor.Tensor, error) { return ops.IndexSelect(x, ti) })
		y := fwd(v, xs, wt)
		oh := o.do(func() (*tensor.Tensor, error) {
			ohv := make([]float32, T*len(tok))
			for i, t := range tok {
				ohv[int(t)*len(tok)+i] = 1
			}
			return tensor.New(ohv, []int{T, len(tok)}, dev)
		})
		y = o.matmul(oh, y)
		if acc == nil {
			acc = y
		} else {
			acc = o.add(acc, y)
		}
		return o.err
	}
	order := moeOrder(byExpert)
	if cache != nil {
		if err := cache.Each(order, one); err != nil && o.err == nil {
			o.err = err
		}
		return acc
	}
	for _, e := range order {
		if one(e, resident[e]) != nil {
			break
		}
	}
	return acc
}

// moePrefetchMaxTokens bounds the batches that predict the next MoE
// layer's experts: decode steps and short chunks, not a whole prompt.
const moePrefetchMaxTokens = 8

// moePrefetch starts loading the experts route picks for x into cache
// (the next layer's router applied to this layer's input: its likely
// picks). Errors only cost the prefetch.
func moePrefetch[V any](cache *nn.ExpertCache[V], x *tensor.Tensor, route func(o *iop, x *tensor.Tensor) [][]moeHit) {
	if cache == nil || x.Dim(0) > moePrefetchMaxTokens {
		return
	}
	o := &iop{}
	byExpert := route(o, x)
	if o.err != nil {
		return
	}
	// The most used experts first: Prefetch starts at most Slots.
	order := moeOrder(byExpert)
	for i := 1; i < len(order); i++ {
		for j := i; j > 0 && len(byExpert[order[j]]) > len(byExpert[order[j-1]]); j-- {
			order[j], order[j-1] = order[j-1], order[j]
		}
	}
	cache.Prefetch(order)
}

// sub returns a loader over only the named weights' tensors (each name's
// .weight / .bias / scales / packed forms), so building them never writes
// into l's map: per-expert loads can run concurrently.
func (l *idolLoader) sub(names ...string) *idolLoader {
	s := &idolLoader{hf: map[string]*tensor.Tensor{}, dev: l.dev, fp8Block: l.fp8Block}
	for _, n := range names {
		for _, suf := range []string{".weight", ".bias", ".weight_scale_inv", ".scale", ".weight_packed", ".weight_scale"} {
			if t, ok := l.hf[n+suf]; ok {
				s.hf[n+suf] = t
			}
		}
	}
	return s
}
