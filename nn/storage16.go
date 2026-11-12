package nn

import (
	"sync"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// storage16 is the load-time option set by WithWeightStorage16.
var storage16 struct {
	load sync.Mutex // one 16-bit-storage load at a time
	mu   sync.Mutex
	on   bool
	dt   dtype.DType
	// mixed: Linears built in the scope round F32 activations to the
	// weight's 16-bit type on devices with a tensor-core GEMM (Mixed16).
	mixed bool
}

// WithWeightStorage16 runs load (a model loader) with 16-bit weight
// storage: while it runs, NewLinearUninit and NewEmbeddingUninit asked for
// an F32 weight allocate it at dt (BF16 or F16) instead, when rows are a
// whole number of 32-bit words. Everything else (norms, biases, the model's
// arithmetic) stays F32; nn.Linear runs ops.MatMulW16 on such weights and
// the embeddings widen gathered rows to F32. A BF16 checkpoint then loads
// with no F32 copy of its weights at all, and computes exactly what the F32
// model does. On WebGPU an embedding table over the device's buffer cap is
// allocated in host memory (its lookups gather there).
//
// Loads under this option are serialized; other loads are unaffected.
func WithWeightStorage16(dt dtype.DType, load func() error) error {
	storage16.load.Lock()
	defer storage16.load.Unlock()
	storage16.mu.Lock()
	storage16.on, storage16.dt = true, dt
	storage16.mu.Unlock()
	defer func() {
		storage16.mu.Lock()
		storage16.on = false
		storage16.mu.Unlock()
	}()
	return load()
}

// WithWeightStorage16Mixed is WithWeightStorage16 whose Linears also
// compute at 16 bits where that is faster (Linear.Mixed16): on CUDA the
// F32 activations are rounded to the weight's type for a tensor-core GEMM
// with F32 accumulation and output, as BF16 / F16 inference in PyTorch
// does. Elsewhere they compute exactly, as under WithWeightStorage16. For
// models run in 16 bits upstream (diffusion transformers), not for exact
// F32 parity.
func WithWeightStorage16Mixed(dt dtype.DType, load func() error) error {
	return WithWeightStorage16(dt, func() error {
		storage16.mu.Lock()
		storage16.mixed = true
		storage16.mu.Unlock()
		defer func() {
			storage16.mu.Lock()
			storage16.mixed = false
			storage16.mu.Unlock()
		}()
		return load()
	})
}

// WeightStorageDType is the dtype a loader that builds its own weight
// tensors should store a matmul weight (or embedding table) of row width
// `in` at, given the model's dtype wdt: the 16-bit storage dtype inside
// WithWeightStorage16, else wdt. An embedding built this way needs
// WidenF32 set when the result differs from wdt.
func WeightStorageDType(wdt dtype.DType, in int) dtype.DType {
	dt, _ := weightDType16(wdt, in)
	return dt
}

// weightDType16 is the dtype an Uninit weight of row width `in` asked for
// at wdt is allocated at, and whether it is the 16-bit storage dtype.
func weightDType16(wdt dtype.DType, in int) (dtype.DType, bool) {
	dt, ok, _ := weightDType16Mixed(wdt, in)
	return dt, ok
}

// weightDType16Mixed is weightDType16 plus whether the scope asks for
// 16-bit compute (WithWeightStorage16Mixed).
func weightDType16Mixed(wdt dtype.DType, in int) (dtype.DType, bool, bool) {
	storage16.mu.Lock()
	defer storage16.mu.Unlock()
	if storage16.on && wdt == dtype.F32 && in%2 == 0 {
		return storage16.dt, true, storage16.mixed
	}
	return wdt, false, false
}

// embeddingDevice16 keeps a 16-bit table too large for one WebGPU buffer in
// host memory.
func embeddingDevice16(dt dtype.DType, n int, dev device.Device) device.Device {
	if device.IsWebGPU(dev) {
		if max := tensor.WebGPUMaxTensorBytes(); max > 0 && int64(dt.ByteSize(n)) > max {
			return device.CPU()
		}
	}
	return dev
}
