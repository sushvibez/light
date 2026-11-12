package models

import (
	"errors"
	"fmt"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/storage"
	"light/tensor"
)

// DecodeGraph replays one decode step of a CausalLM as a CUDA graph. The
// step is captured once (every launch, copy and memset the model issues
// for one token, with the position, the KV append slot and the attention
// length read on the device), then each token is one upload of its id
// and one graph launch instead of a few hundred driver calls.
//
// What the capture relies on: the cache's capacity buffers do not move
// (Reserve sizes them up front, and Step re-captures if a longer run
// outgrows them), the step allocates its transients through the storage
// pool (an allocation observer retains them for the graph's life), and
// the model's single-token path uploads nothing per step (a synchronous
// copy during capture fails the capture, and the caller falls back to
// eager decoding).
type DecodeGraph struct {
	m     CausalLM
	cache *KVCache
	dev   device.Device

	stream  uintptr
	exec    uintptr
	tok     *tensor.Tensor // (1, 1) I64 on the device: the step's input
	tokHost uintptr        // pinned mirror of tok, the source of each upload
	logits  *tensor.Tensor // (1, V) on the device, rewritten by every launch
	keep    []*storage.Storage
	nodes   int
	reserve int
}

// ErrNoGraph says the decode step could not be captured (the model's
// single-token path is not graph-safe); decode eagerly instead.
var ErrNoGraph = errors.New("decode graph: step not capturable")

// GraphDecodable is implemented by models whose single-token step reads
// the position on the device at every layer, so the launches it issues
// are the same at every position. A model that does not implement it, or
// reports an error, decodes eagerly.
type GraphDecodable interface {
	DecodeGraphOK() error
}

// NewDecodeGraph captures the decode step of m over cache, which must hold
// a prefilled prompt, on CUDA device dev. reserve is the sequence length
// the run may reach (prompt plus new tokens); the cache's key / value
// buffers are sized to it so no step reallocates.
func NewDecodeGraph(m CausalLM, cache *KVCache, dev device.Device, reserve int) (*DecodeGraph, error) {
	if !device.IsCUDA(dev) {
		return nil, fmt.Errorf("decode graph: %v is not a CUDA device", dev)
	}
	if cache == nil || cache.Len == 0 {
		return nil, errors.New("decode graph: prefill the cache first")
	}
	if cache.ring {
		return nil, errors.New("decode graph: ring-buffer cache")
	}
	gd, ok := m.(GraphDecodable)
	if !ok {
		return nil, fmt.Errorf("%w: %T has no device-position decode path", ErrNoGraph, m)
	}
	if err := gd.DecodeGraphOK(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNoGraph, err)
	}
	if err := cudart.DeviceSynchronize(); err != nil {
		return nil, err
	}
	if cudart.Stream() == 0 {
		s, err := cudart.StreamCreate()
		if err != nil {
			return nil, err
		}
		cudart.SetStream(s)
	}
	g := &DecodeGraph{m: m, cache: cache, dev: dev, stream: cudart.Stream(), reserve: reserve}
	var err error
	if g.tok, err = tensor.Zeros(dtype.I64, []int{1, 1}, dev); err != nil {
		return nil, err
	}
	if g.tokHost, err = cudart.MallocHost(8); err != nil {
		return nil, err
	}
	if err := g.capture(); err != nil {
		g.Close()
		return nil, err
	}
	return g, nil
}

// capture records one step. Nothing runs: the host-side cache state the
// pass advanced is put back afterwards.
func (g *DecodeGraph) capture() error {
	c := g.cache
	if err := c.Reserve(max(g.reserve, c.Len+1)); err != nil {
		return err
	}
	if _, err := c.position(g.dev); err != nil { // upload now, outside the capture
		return err
	}
	lenBefore := c.Len
	var keep []*storage.Storage
	prev := storage.SetCUDAAllocObserver(func(s *storage.Storage) {
		s.Acquire()
		keep = append(keep, s)
	})
	if err := cudart.StreamBeginCapture(g.stream); err != nil {
		storage.SetCUDAAllocObserver(prev)
		return err
	}
	c.capturing = true
	logits, ferr := g.forward()
	c.capturing = false
	graph, eerr := cudart.StreamEndCapture(g.stream)
	storage.SetCUDAAllocObserver(prev)
	// The pass advanced Len (and queued, not ran, the device increment).
	c.Len = lenBefore
	for i, b := range c.kBuf {
		if b != nil && c.K[i] != nil && c.K[i].Storage() == b.Storage() {
			if err := c.setView(i, lenBefore); err != nil {
				return err
			}
		}
	}
	if ferr != nil || eerr != nil {
		for _, s := range keep {
			s.Release()
		}
		if ferr != nil {
			return fmt.Errorf("%w: %v", ErrNoGraph, ferr)
		}
		return fmt.Errorf("%w: %v", ErrNoGraph, eerr)
	}
	nodes, _ := cudart.GraphNodeCount(graph)
	exec, err := cudart.GraphInstantiate(graph)
	_ = cudart.GraphDestroy(graph)
	if err != nil {
		for _, s := range keep {
			s.Release()
		}
		return fmt.Errorf("%w: %v", ErrNoGraph, err)
	}
	g.dropExec()
	g.exec, g.logits, g.keep, g.nodes = exec, logits, keep, nodes
	return nil
}

// forward is the step as the engine runs it: the token through the model,
// the last position's logits as a (1, V) view.
func (g *DecodeGraph) forward() (*tensor.Tensor, error) {
	logits, err := g.m.ForwardWithCache(g.tok, g.cache)
	if err != nil {
		return nil, err
	}
	V := g.m.VocabSize()
	flat, err := logits.Reshape(logits.NumElements()/V, V)
	if err != nil {
		return nil, err
	}
	row, err := flat.Narrow(0, flat.Dim(0)-1, 1)
	if err != nil {
		return nil, err
	}
	return row.Contiguous()
}

// Step feeds tok to the captured step and returns the logits (1, V) on
// the device; they stay valid until the next Step.
func (g *DecodeGraph) Step(tok int64) (*tensor.Tensor, error) {
	c := g.cache
	if c.Len+1 > c.Capacity() { // outgrew the buffers: bigger ones, new capture
		g.reserve = c.Len + 1 + 1024
		if err := cudart.DeviceSynchronize(); err != nil {
			return nil, err
		}
		if err := g.capture(); err != nil {
			return nil, err
		}
	}
	*(*int64)(unsafe.Pointer(g.tokHost)) = tok
	if err := cudart.MemcpyHtoDAsyncOn(g.tok.Storage().CUDAPtr(), g.tokHost, 8, g.stream); err != nil {
		return nil, err
	}
	if err := cudart.GraphLaunch(g.exec, g.stream); err != nil {
		return nil, err
	}
	c.Len++ // the graph bumped the device position itself
	for i, b := range c.kBuf {
		if b != nil && c.K[i] != nil {
			if err := c.setView(i, c.Len); err != nil {
				return nil, err
			}
		}
	}
	return g.logits, nil
}

// Nodes is the number of nodes in the captured graph.
func (g *DecodeGraph) Nodes() int { return g.nodes }

func (g *DecodeGraph) dropExec() {
	if g.exec != 0 {
		_ = cudart.GraphExecDestroy(g.exec)
		g.exec = 0
	}
	for _, s := range g.keep {
		s.Release()
	}
	g.keep, g.logits = nil, nil
}

// Close releases the executable graph and the step's retained buffers.
// The device work queued by the last Step must be complete (sync first
// unless the logits were already read back).
func (g *DecodeGraph) Close() {
	g.dropExec()
	if g.tokHost != 0 {
		_ = cudart.FreeHost(g.tokHost)
		g.tokHost = 0
	}
}
