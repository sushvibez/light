package nn

import (
	"fmt"
	"math"
	"unsafe"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/ops"
	"light/quant"
	"light/tensor"
)

// Embedding is a lookup table: indices (..., K) → (..., K, EmbDim).
type Embedding struct {
	Weight *tensor.Tensor // (NumEmbeddings, EmbDim)
	// WidenF32 casts the gathered rows to F32: a 16-bit table kept for
	// memory under an F32 model (exact widening).
	WidenF32 bool
	numEmb   int
	embDim   int
}

// NewEmbedding initialises a table with small uniform values so the network
// has some variance at inference time on fresh (untrained) instances.
// Loaders overwrite Weight in place anyway.
func NewEmbedding(numEmbeddings, embDim int, dev device.Device) (*Embedding, error) {
	// Uniform[-0.1, 0.1] — reasonable default for fresh networks.
	w, err := tensor.Rand(dtype.F32, []int{numEmbeddings, embDim}, device.CPU())
	if err != nil {
		return nil, err
	}
	scaleUniformIn(w, 0.1)
	if !dev.SameAs(device.CPU()) {
		w, err = w.To(dev)
		if err != nil {
			return nil, err
		}
	}
	return &Embedding{Weight: w, numEmb: numEmbeddings, embDim: embDim}, nil
}

// NewEmbeddingFromWeight wraps an existing (NumEmbeddings, EmbDim) table,
// e.g. a GGUF block-quantized one, without allocating.
func NewEmbeddingFromWeight(w *tensor.Tensor) (*Embedding, error) {
	if w == nil || w.Rank() != 2 {
		return nil, fmt.Errorf("nn.NewEmbeddingFromWeight: want a rank-2 table, got %v", w)
	}
	return &Embedding{Weight: w, numEmb: w.Dim(0), embDim: w.Dim(1)}, nil
}

// NewEmbeddingUninit allocates a zero-initialised table at `wdt` directly on
// `dev`. Use this when a loader is about to overwrite the weight — for a
// 3B-param model this skips ~300M random draws and a CPU→GPU transfer.
func NewEmbeddingUninit(numEmbeddings, embDim int, wdt dtype.DType, dev device.Device) (*Embedding, error) {
	w, err := tensor.Zeros(wdt, []int{numEmbeddings, embDim}, dev)
	if err != nil {
		return nil, err
	}
	return &Embedding{Weight: w, numEmb: numEmbeddings, embDim: embDim}, nil
}

// Forward looks up embeddings by the given I64 indices tensor of arbitrary
// shape. Output shape = indices.Shape + (EmbDim,).
func (e *Embedding) Forward(indices *tensor.Tensor) (*tensor.Tensor, error) {
	// Flatten indices, do IndexSelect, then reshape.
	flatLen := indices.NumElements()
	flat, err := indices.Contiguous()
	if err != nil {
		return nil, err
	}
	flat, err = flat.Reshape(flatLen)
	if err != nil {
		return nil, err
	}
	// A table kept in host memory under a device model (too large for one
	// device buffer): gather on the host, ship only the rows.
	outDev := indices.Device()
	if !outDev.SameAs(e.Weight.Device()) {
		if flat, err = flat.To(e.Weight.Device()); err != nil {
			return nil, err
		}
	}
	var looked *tensor.Tensor
	if e.Weight.DType().IsQuantized() {
		// GGUF table kept in its block format: gather the rows' blocks, then
		// dequantize only those (inference only, no autograd).
		rows, err := ops.IndexSelect(e.Weight, flat)
		if err != nil {
			return nil, err
		}
		if looked, err = quant.Dequantize(rows); err != nil {
			return nil, err
		}
	} else if looked, err = autograd.IndexSelect(e.Weight, flat); err != nil {
		return nil, err
	}
	if e.WidenF32 && looked.DType() != dtype.F32 {
		if looked, err = ops.Cast(looked, dtype.F32); err != nil {
			return nil, err
		}
	}
	if !outDev.SameAs(looked.Device()) {
		if looked, err = looked.To(outDev); err != nil {
			return nil, err
		}
	}
	// Reshape result back: indices shape + (embDim,)
	newShape := append([]int{}, indices.Shape().Dims()...)
	newShape = append(newShape, e.embDim)
	return autograd.Reshape(looked, newShape...)
}

func (e *Embedding) Parameters() []*tensor.Tensor { return []*tensor.Tensor{e.Weight} }

// NumEmbeddings returns the vocabulary size.
func (e *Embedding) NumEmbeddings() int { return e.numEmb }

// EmbDim returns the embedding dimension.
func (e *Embedding) EmbDim() int { return e.embDim }

// scaleUniformIn rescales a CPU F32 tensor from Uniform[0,1) to Uniform[-k,+k).
func scaleUniformIn(t *tensor.Tensor, k float32) {
	_ = math.Pi // keep math in case we need it later
	buf := t.Storage().CPUBytes()
	vals := unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), len(buf)/4)
	for i := range vals {
		vals[i] = vals[i]*2*k - k
	}
}
