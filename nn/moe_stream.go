package nn

import (
	"fmt"
	"sync"

	"light/device"
	"light/ops"
	"light/tensor"
)

// ExpertStream keeps only Slots of a SparseMoE's experts in device memory:
// the rest stay wherever Load reads them from (typically the memory-mapped
// checkpoint) until the router picks them. Each forward routes as usual,
// makes the picked experts resident (evicting the least recently used),
// remaps the routing onto slots and runs the same grouped GEMMs on the slot
// buffers. A batch that needs more experts than there are slots runs in
// token chunks that each fit.
//
// Prefetch(ids) starts loading experts in the background (a later layer's
// predicted picks); a forward that needs one waits for it.
type ExpertStream struct {
	// Load returns expert e's stacked gate+up (2I, H) and down (H, I)
	// weights at the slot dtype (any device).
	Load  func(e int) (gateUp, down *tensor.Tensor, err error)
	Slots int
	// Loads counts experts read in; Hits experts already resident.
	// Prefetched counts background loads Prefetch started; PrefetchUsed
	// those a forward then needed (the prediction's hit count).
	Loads, Hits, Prefetched, PrefetchUsed int64

	mu      sync.Mutex
	dev     device.Device
	gu, dn  *tensor.Tensor // (Slots, 2I, H), (Slots, H, I)
	slotOf  map[int]int
	owner   []int
	used    []uint64
	clock   uint64
	pending map[int]*expertLoad
	fromPF  map[int]bool // resident or pending through Prefetch, not yet used
}

type expertLoad struct {
	done   chan struct{}
	gu, dn *tensor.Tensor
	err    error
}

// prefetchMaxTokens bounds the batches that predict the next layer: decode
// steps and short chunks, not a whole prompt.
const prefetchMaxTokens = 8

// prefetchFor predicts the experts m routes x (T, H) to and prefetches
// them. Errors only cost the prefetch.
func (m *SparseMoE) prefetchFor(x *tensor.Tensor) {
	route, err := m.route(x)
	if err != nil {
		return
	}
	idsT, err := route.TopKIDs.To(device.CPU())
	if err == nil {
		idsT, err = idsT.Contiguous()
	}
	if err != nil {
		return
	}
	seen := map[int]bool{}
	var ids []int
	for _, e := range tensor.MustSlice[int64](idsT) {
		if !seen[int(e)] {
			seen[int(e)] = true
			ids = append(ids, int(e))
		}
	}
	if len(ids) > m.Stream.Slots {
		ids = ids[:m.Stream.Slots]
	}
	m.Stream.Prefetch(ids)
}

// NewExpertStream returns a stream with slots resident experts on dev.
func NewExpertStream(slots int, dev device.Device, load func(e int) (*tensor.Tensor, *tensor.Tensor, error)) (*ExpertStream, error) {
	if slots < 1 || load == nil {
		return nil, fmt.Errorf("nn.ExpertStream: need ≥ 1 slot and a loader")
	}
	s := &ExpertStream{Load: load, Slots: slots, dev: dev, slotOf: map[int]int{}, owner: make([]int, slots),
		used: make([]uint64, slots), pending: map[int]*expertLoad{}, fromPF: map[int]bool{}}
	for i := range s.owner {
		s.owner[i] = -1
	}
	return s, nil
}

// Prefetch starts reading the given experts in the background unless they
// are resident or already on their way.
func (s *ExpertStream) Prefetch(ids []int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, e := range ids {
		if _, ok := s.slotOf[e]; ok {
			continue
		}
		if _, ok := s.pending[e]; ok {
			continue
		}
		s.start(e)
		s.Prefetched++
		s.fromPF[e] = true
	}
}

// start launches expert e's load (s.mu held).
func (s *ExpertStream) start(e int) *expertLoad {
	l := &expertLoad{done: make(chan struct{})}
	s.pending[e] = l
	go func() {
		defer close(l.done)
		gu, dn, err := s.Load(e)
		if err == nil && !gu.Device().SameAs(s.dev) {
			if gu, err = gu.To(s.dev); err == nil {
				dn, err = dn.To(s.dev)
			}
		}
		l.gu, l.dn, l.err = gu, dn, err
	}()
	return l
}

// ensure makes every expert in need resident and returns their slots.
// len(need) must not exceed Slots.
func (s *ExpertStream) ensure(need []int) (map[int]int, error) {
	if len(need) > s.Slots {
		return nil, fmt.Errorf("nn.ExpertStream: %d experts for %d slots", len(need), s.Slots)
	}
	s.mu.Lock()
	keep := map[int]bool{}
	for _, e := range need {
		keep[e] = true
	}
	var waits []int
	for _, e := range need {
		if s.fromPF[e] {
			s.PrefetchUsed++
			delete(s.fromPF, e)
		}
		s.clock++
		if sl, ok := s.slotOf[e]; ok {
			s.used[sl] = s.clock
			s.Hits++
			continue
		}
		if _, ok := s.pending[e]; !ok {
			s.start(e)
		}
		waits = append(waits, e)
	}
	s.mu.Unlock()
	for _, e := range waits {
		s.mu.Lock()
		l := s.pending[e]
		s.mu.Unlock()
		<-l.done
		s.mu.Lock()
		delete(s.pending, e)
		if l.err != nil {
			s.mu.Unlock()
			return nil, fmt.Errorf("nn.ExpertStream: expert %d: %w", e, l.err)
		}
		if err := s.install(e, l.gu, l.dn, keep); err != nil {
			s.mu.Unlock()
			return nil, err
		}
		s.Loads++
		s.mu.Unlock()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[int]int, len(need))
	for _, e := range need {
		out[e] = s.slotOf[e]
	}
	return out, nil
}

// install copies an expert into the least recently used slot not in keep
// (s.mu held).
func (s *ExpertStream) install(e int, gu, dn *tensor.Tensor, keep map[int]bool) error {
	if s.gu == nil {
		var err error
		if s.gu, err = tensor.Zeros(gu.DType(), append([]int{s.Slots}, gu.Shape().Dims()...), s.dev); err != nil {
			return err
		}
		if s.dn, err = tensor.Zeros(dn.DType(), append([]int{s.Slots}, dn.Shape().Dims()...), s.dev); err != nil {
			return err
		}
	}
	best := -1
	for sl, o := range s.owner {
		if o >= 0 && keep[o] {
			continue
		}
		if best < 0 || s.used[sl] < s.used[best] || (s.owner[sl] < 0 && s.owner[best] >= 0) {
			best = sl
		}
	}
	if best < 0 {
		return fmt.Errorf("nn.ExpertStream: no free slot")
	}
	put := func(buf, w *tensor.Tensor) error {
		v, err := buf.Narrow(0, best, 1)
		if err != nil {
			return err
		}
		if w.DType() != buf.DType() {
			return fmt.Errorf("nn.ExpertStream: expert %d is %v, slots are %v", e, w.DType(), buf.DType())
		}
		w1, err := w.Contiguous()
		if err == nil {
			w1, err = w1.Reshape(append([]int{1}, w.Shape().Dims()...)...)
		}
		if err == nil {
			err = v.Assign(w1)
		}
		return err
	}
	if err := put(s.gu, gu); err != nil {
		return err
	}
	if err := put(s.dn, dn); err != nil {
		return err
	}
	if o := s.owner[best]; o >= 0 {
		delete(s.slotOf, o)
		delete(s.fromPF, o)
	}
	s.owner[best], s.slotOf[e] = e, best
	s.clock++
	s.used[best] = s.clock
	return nil
}

// Resident lists the experts currently in slots.
func (s *ExpertStream) Resident() []int {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []int
	for _, o := range s.owner {
		if o >= 0 {
			out = append(out, o)
		}
	}
	return out
}

// forwardStreamed runs the experts of route on flat (T, H) through the
// stream: chunks of tokens whose experts fit the slots.
func (m *SparseMoE) forwardStreamed(flat *tensor.Tensor, route *ops.MoERouting) (*tensor.Tensor, error) {
	s := m.Stream
	T, K := flat.Dim(0), m.TopK
	if K > s.Slots {
		return nil, fmt.Errorf("nn.SparseMoE: top-%d routing needs ≥ %d expert slots, have %d", K, K, s.Slots)
	}
	idsT, err := route.TopKIDs.To(device.CPU())
	if err == nil {
		idsT, err = idsT.Contiguous()
	}
	if err != nil {
		return nil, err
	}
	wT, err := route.Weights.To(device.CPU())
	if err == nil {
		wT, err = wT.Contiguous()
	}
	if err != nil {
		return nil, err
	}
	ids, ws := tensor.MustSlice[int64](idsT), tensor.MustSlice[float32](wT)
	var outs []*tensor.Tensor
	for start := 0; start < T; {
		set := map[int]bool{}
		end := start
		for end < T {
			add := 0
			for k := range K {
				if e := int(ids[end*K+k]); !set[e] {
					add++
				}
			}
			if len(set)+add > s.Slots && end > start {
				break
			}
			for k := range K {
				set[int(ids[end*K+k])] = true
			}
			end++
		}
		need := make([]int, 0, len(set))
		for e := range set {
			need = append(need, e)
		}
		slots, err := s.ensure(need)
		if err != nil {
			return nil, err
		}
		sid := make([]int64, (end-start)*K)
		for i := range sid {
			sid[i] = int64(slots[int(ids[start*K+i])])
		}
		sub, err := ops.MoERouteFromIDs(sid, append([]float32(nil), ws[start*K:end*K]...), end-start, K, flat.Device())
		if err != nil {
			return nil, err
		}
		x := flat
		if start > 0 || end < T {
			if x, err = flat.Narrow(0, start, end-start); err == nil {
				x, err = x.Contiguous()
			}
			if err != nil {
				return nil, err
			}
		}
		s.mu.Lock()
		gu, dn := s.gu, s.dn
		s.mu.Unlock()
		y, err := m.runExperts(x, gu, dn, sub)
		if err != nil {
			return nil, err
		}
		outs = append(outs, y)
		start = end
	}
	if len(outs) == 1 {
		return outs[0], nil
	}
	return ops.Concat(outs, 0)
}
