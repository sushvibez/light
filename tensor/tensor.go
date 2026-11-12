package tensor

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/shape"
	"light/storage"
)

// Tensor is a view into a backing Storage, with a Shape, Strides, and an
// element offset. Tensors are Go-GC-managed; attach storage cleanup via
// storage.Track at construction.
type Tensor struct {
	storage *storage.Storage
	shape   shape.Shape
	strides shape.Strides
	offset  int

	// Autograd fields (populated by Phase 4). Reserved here so Tensor's
	// layout is stable across phases.
	requiresGrad bool
	grad         *Tensor
	gradFn       any // *autograd.FunctionNode, untyped to avoid import cycle
	outputNr     int
	version      uint64 //nolint:unused // phase 4 (in-place safety)
}

// Shape returns this tensor's shape.
func (t *Tensor) Shape() shape.Shape { return t.shape.Clone() }

// Strides returns a copy of this tensor's strides.
func (t *Tensor) Strides() shape.Strides { return shape.NewStrides(t.strides.Slice()...) }

// DType returns the element type.
func (t *Tensor) DType() dtype.DType { return t.storage.DType() }

// Device returns the device this tensor lives on.
func (t *Tensor) Device() device.Device { return t.storage.Device() }

// NumElements returns the total logical element count.
func (t *Tensor) NumElements() int { return t.shape.NumElements() }

// Rank returns the number of dimensions.
func (t *Tensor) Rank() int { return t.shape.Rank() }

// Dim returns the size of axis i (supports negative indexing).
func (t *Tensor) Dim(i int) int { return t.shape.Dim(i) }

// IsContiguous reports whether the tensor's memory layout is row-major
// contiguous with no broadcasting.
func (t *Tensor) IsContiguous() bool {
	if shape.IsContiguous(t.shape, t.strides) {
		return true
	}
	// A dense block with permuted size-1 axes (a single decode token's
	// (B, 1, H, D) → (B, H, 1, D) head transpose) needs no copy either, as
	// long as it is exactly its storage: kernels take a contiguous tensor's
	// data as the whole buffer from offset 0.
	return t.offset == 0 && t.storage.Len() == t.shape.NumElements() && shape.IsDense(t.shape, t.strides)
}

// Offset returns the element offset into the backing storage.
func (t *Tensor) Offset() int { return t.offset }

// Storage returns the underlying storage. Intended for kernel implementers.
func (t *Tensor) Storage() *storage.Storage { return t.storage }

// RequiresGrad reports whether this tensor should track gradients in autograd.
func (t *Tensor) RequiresGrad() bool { return t.requiresGrad }

// SetRequiresGrad toggles gradient tracking for this tensor.
func (t *Tensor) SetRequiresGrad(v bool) { t.requiresGrad = v }

// Grad returns the accumulated gradient (populated by backward; nil until
// backward has run).
func (t *Tensor) Grad() *Tensor { return t.grad }

// SetGrad assigns a new gradient tensor. Used by the autograd engine.
func (t *Tensor) SetGrad(g *Tensor) { t.grad = g }

// GradFn returns the FunctionNode that produced this tensor (nil for leaves).
func (t *Tensor) GradFn() any { return t.gradFn }

// OutputNr returns which output index of the producing FunctionNode this
// tensor corresponds to. Meaningful only when GradFn() is non-nil.
func (t *Tensor) OutputNr() int { return t.outputNr }

// SetGradFn is used by autograd to install the producing FunctionNode.
func (t *Tensor) SetGradFn(fn any, outputNr int) {
	t.gradFn = fn
	t.outputNr = outputNr
}

// new constructs a fully-initialized Tensor and attaches storage cleanup.
// Callers are ops/factory functions.
func newTensor(s *storage.Storage, sh shape.Shape, st shape.Strides, offset int) *Tensor {
	t := &Tensor{
		storage: s,
		shape:   sh,
		strides: st,
		offset:  offset,
	}
	storage.Track(s, t)
	return t
}

// Zeros allocates a zero-initialized tensor of the given dtype and shape on
// the given device (CPU or CUDA).
func Zeros(dt dtype.DType, dims []int, dev device.Device) (*Tensor, error) {
	sh := shape.New(dims...)
	s, err := allocStorage(dt, sh.NumElements(), dev)
	if err != nil {
		return nil, err
	}
	return newTensor(s, sh, shape.DefaultStrides(sh), 0), nil
}

// Empty allocates a tensor whose contents are unspecified: the output
// buffer of an op that writes every element. On CUDA this skips the
// cudaMemset that Zeros issues; on other devices it is Zeros.
func Empty(dt dtype.DType, dims []int, dev device.Device) (*Tensor, error) {
	sh := shape.New(dims...)
	s, err := allocStorageUninit(dt, sh.NumElements(), dev)
	if err != nil {
		return nil, err
	}
	return newTensor(s, sh, shape.DefaultStrides(sh), 0), nil
}

// Ones allocates a tensor filled with 1 (in the given dtype) on the given
// device. For quantized dtypes, returns an error — "1" isn't meaningful at
// the block level until Phase 9 quantization paths exist.
func Ones(dt dtype.DType, dims []int, dev device.Device) (*Tensor, error) {
	if dt.IsQuantized() {
		return nil, fmt.Errorf("tensor.Ones: quantized dtype %v not supported", dt)
	}
	// Stage on CPU (fillScalar is CPU-only); transfer to target device if needed.
	cpuT, err := Zeros(dt, dims, device.CPU())
	if err != nil {
		return nil, err
	}
	if err := fillScalar(cpuT, 1.0); err != nil {
		return nil, err
	}
	if dev.Kind() == device.KindCPU {
		return cpuT, nil
	}
	return cpuT.To(dev)
}

// Full allocates a tensor filled with val.
func Full[T dtype.Numeric](val T, dims []int, dev device.Device) (*Tensor, error) {
	dt := dtype.Of[T]()
	// Always fill on CPU first; transfer to GPU if needed.
	cpuT, err := Zeros(dt, dims, device.CPU())
	if err != nil {
		return nil, err
	}
	buf := cpuT.storage.CPUBytes()
	elemSize := dt.Size()
	vbytes := unsafe.Slice((*byte)(unsafe.Pointer(&val)), elemSize)
	for i := 0; i < len(buf); i += elemSize {
		copy(buf[i:i+elemSize], vbytes)
	}
	if dev.Kind() == device.KindCPU {
		return cpuT, nil
	}
	return cpuT.To(dev)
}

// fillScalar fills a CPU tensor with the given float64 value, converting per
// dtype. Used only during Phase 1; replace with a proper kernel in the op
// registry later.
func fillScalar(t *Tensor, v float64) error {
	if t.Device().Kind() != device.KindCPU {
		return fmt.Errorf("fillScalar: only CPU")
	}
	buf := t.storage.CPUBytes()
	dt := t.DType()
	switch dt {
	case dtype.F32:
		vals := unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), len(buf)/4)
		fv := float32(v)
		for i := range vals {
			vals[i] = fv
		}
	case dtype.F64:
		vals := unsafe.Slice((*float64)(unsafe.Pointer(&buf[0])), len(buf)/8)
		for i := range vals {
			vals[i] = v
		}
	case dtype.I32:
		vals := unsafe.Slice((*int32)(unsafe.Pointer(&buf[0])), len(buf)/4)
		iv := int32(v)
		for i := range vals {
			vals[i] = iv
		}
	case dtype.I64:
		vals := unsafe.Slice((*int64)(unsafe.Pointer(&buf[0])), len(buf)/8)
		iv := int64(v)
		for i := range vals {
			vals[i] = iv
		}
	case dtype.U8:
		uv := byte(v)
		for i := range buf {
			buf[i] = uv
		}
	case dtype.I8:
		iv := int8(v)
		for i := range buf {
			buf[i] = byte(iv)
		}
	case dtype.F16:
		h := dtype.HalfFromF32(float32(v))
		vals := unsafe.Slice((*uint16)(unsafe.Pointer(&buf[0])), len(buf)/2)
		for i := range vals {
			vals[i] = uint16(h)
		}
	case dtype.BF16:
		b := dtype.BF16FromF32(float32(v))
		vals := unsafe.Slice((*uint16)(unsafe.Pointer(&buf[0])), len(buf)/2)
		for i := range vals {
			vals[i] = uint16(b)
		}
	default:
		return fmt.Errorf("fillScalar: unsupported dtype %v", dt)
	}
	return nil
}

// New constructs a tensor from a Go slice. The element count must match the
// product of shape dims. On CUDA devices, data is uploaded via an
// intermediate CPU tensor.
func New[T dtype.Numeric](data []T, dims []int, dev device.Device) (*Tensor, error) {
	dt := dtype.Of[T]()
	sh := shape.New(dims...)
	if len(data) != sh.NumElements() {
		return nil, fmt.Errorf("tensor.New: data has %d elements but shape %v needs %d", len(data), sh, sh.NumElements())
	}
	// Always stage on CPU first — data is a Go slice.
	s := storage.NewCPU(dt, sh.NumElements())
	if len(data) > 0 {
		dst := s.CPUBytes()
		src := unsafeBytes(data)
		copy(dst, src)
	}
	cpuT := newTensor(s, sh, shape.DefaultStrides(sh), 0)
	if dev.Kind() == device.KindCPU {
		return cpuT, nil
	}
	return cpuT.To(dev)
}

// ToSlice reads a contiguous CPU tensor's data into a Go slice. Returns an
// error if the tensor is non-contiguous, on a non-CPU device, or has a dtype
// that doesn't match T.
func ToSlice[T dtype.Numeric](t *Tensor) ([]T, error) {
	if t.Device().Kind() != device.KindCPU {
		return nil, fmt.Errorf("ToSlice: tensor is on %v, not CPU", t.Device())
	}
	if !t.IsContiguous() || t.offset != 0 {
		return nil, errors.New("ToSlice: tensor is not contiguous — call Contiguous() first")
	}
	if dtype.Of[T]() != t.DType() {
		return nil, fmt.Errorf("ToSlice: dtype mismatch (tensor is %v, asked for %v)", t.DType(), dtype.Of[T]())
	}
	n := t.NumElements()
	if n == 0 {
		return []T{}, nil
	}
	buf := t.storage.CPUBytes()
	src := unsafe.Slice((*T)(unsafe.Pointer(&buf[0])), n)
	out := make([]T, n)
	copy(out, src)
	return out, nil
}

// MustSlice is a panicking convenience for examples.
func MustSlice[T dtype.Numeric](t *Tensor) []T {
	s, err := ToSlice[T](t)
	if err != nil {
		panic(err)
	}
	return s
}

// ToHalfSlice returns a contiguous F16 tensor's data as []dtype.Half. Returns
// an error if the tensor isn't F16, isn't contiguous, or is off-CPU.
func ToHalfSlice(t *Tensor) ([]dtype.Half, error) {
	if t.Device().Kind() != device.KindCPU {
		return nil, fmt.Errorf("ToHalfSlice: tensor is on %v, not CPU", t.Device())
	}
	if t.DType() != dtype.F16 {
		return nil, fmt.Errorf("ToHalfSlice: dtype is %v, not F16", t.DType())
	}
	if !t.IsContiguous() || t.offset != 0 {
		return nil, errors.New("ToHalfSlice: tensor is not contiguous")
	}
	buf := t.storage.CPUBytes()
	n := t.NumElements()
	if n == 0 {
		return []dtype.Half{}, nil
	}
	src := unsafe.Slice((*dtype.Half)(unsafe.Pointer(&buf[0])), n)
	out := make([]dtype.Half, n)
	copy(out, src)
	return out, nil
}

// ToBF16Slice returns a contiguous BF16 tensor's data as []dtype.BFloat16.
func ToBF16Slice(t *Tensor) ([]dtype.BFloat16, error) {
	if t.Device().Kind() != device.KindCPU {
		return nil, fmt.Errorf("ToBF16Slice: tensor is on %v, not CPU", t.Device())
	}
	if t.DType() != dtype.BF16 {
		return nil, fmt.Errorf("ToBF16Slice: dtype is %v, not BF16", t.DType())
	}
	if !t.IsContiguous() || t.offset != 0 {
		return nil, errors.New("ToBF16Slice: tensor is not contiguous")
	}
	buf := t.storage.CPUBytes()
	n := t.NumElements()
	if n == 0 {
		return []dtype.BFloat16{}, nil
	}
	src := unsafe.Slice((*dtype.BFloat16)(unsafe.Pointer(&buf[0])), n)
	out := make([]dtype.BFloat16, n)
	copy(out, src)
	return out, nil
}

// unsafeBytes reinterprets a typed slice as a byte slice. Little-endian hosts
// only — all Go targets that Light supports (amd64, arm64) are LE.
func unsafeBytes[T any](data []T) []byte {
	if len(data) == 0 {
		return nil
	}
	elemSize := int(unsafe.Sizeof(data[0]))
	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), elemSize*len(data))
}

// RandN returns a tensor filled with standard-normal (mean 0, std 1) random
// values via the Box-Muller transform. F32 and F64 only.
func RandN(dt dtype.DType, dims []int, dev device.Device) (*Tensor, error) {
	if dt != dtype.F32 && dt != dtype.F64 {
		return nil, fmt.Errorf("tensor.RandN: unsupported dtype %v (try F32/F64)", dt)
	}
	cpuT, err := Zeros(dt, dims, device.CPU())
	if err != nil {
		return nil, err
	}
	n := cpuT.NumElements()
	if n == 0 {
		if dev.Kind() == device.KindCPU {
			return cpuT, nil
		}
		return cpuT.To(dev)
	}
	buf := cpuT.storage.CPUBytes()
	switch dt {
	case dtype.F32:
		vals := unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), n)
		for i := 0; i < n; i += 2 {
			u1 := rand.Float64()
			if u1 < 1e-12 {
				u1 = 1e-12
			}
			u2 := rand.Float64()
			r := math.Sqrt(-2 * math.Log(u1))
			θ := 2 * math.Pi * u2
			vals[i] = float32(r * math.Cos(θ))
			if i+1 < n {
				vals[i+1] = float32(r * math.Sin(θ))
			}
		}
	case dtype.F64:
		vals := unsafe.Slice((*float64)(unsafe.Pointer(&buf[0])), n)
		for i := 0; i < n; i += 2 {
			u1 := rand.Float64()
			if u1 < 1e-12 {
				u1 = 1e-12
			}
			u2 := rand.Float64()
			r := math.Sqrt(-2 * math.Log(u1))
			θ := 2 * math.Pi * u2
			vals[i] = r * math.Cos(θ)
			if i+1 < n {
				vals[i+1] = r * math.Sin(θ)
			}
		}
	}
	if dev.Kind() == device.KindCPU {
		return cpuT, nil
	}
	return cpuT.To(dev)
}

// Rand returns a tensor filled with Uniform[0,1) random values. F32 and F64
// are supported on any device: values are generated on CPU and transferred
// to the target device if needed.
func Rand(dt dtype.DType, dims []int, dev device.Device) (*Tensor, error) {
	cpuT, err := Zeros(dt, dims, device.CPU())
	if err != nil {
		return nil, err
	}
	buf := cpuT.storage.CPUBytes()
	switch dt {
	case dtype.F32:
		vals := unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), len(buf)/4)
		for i := range vals {
			vals[i] = rand.Float32()
		}
	case dtype.F64:
		vals := unsafe.Slice((*float64)(unsafe.Pointer(&buf[0])), len(buf)/8)
		for i := range vals {
			vals[i] = rand.Float64()
		}
	default:
		return nil, fmt.Errorf("tensor.Rand: unsupported dtype %v", dt)
	}
	if dev.Kind() == device.KindCPU {
		return cpuT, nil
	}
	return cpuT.To(dev)
}

// Arange returns a tensor containing the arithmetic sequence [start, stop)
// with the given step. Only F32 and F64 supported in Phase 1.
func Arange(start, stop, step float64, dt dtype.DType, dev device.Device) (*Tensor, error) {
	if step == 0 {
		return nil, errors.New("Arange: step must be non-zero")
	}
	// Count elements.
	n := max(int((stop-start)/step), 0)
	t, err := Zeros(dt, []int{n}, dev)
	if err != nil {
		return nil, err
	}
	buf := t.storage.CPUBytes()
	switch dt {
	case dtype.F32:
		vals := unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), len(buf)/4)
		for i := range vals {
			vals[i] = float32(start + float64(i)*step)
		}
	case dtype.F64:
		vals := unsafe.Slice((*float64)(unsafe.Pointer(&buf[0])), len(buf)/8)
		for i := range vals {
			vals[i] = start + float64(i)*step
		}
	default:
		return nil, fmt.Errorf("Arange: unsupported dtype %v", dt)
	}
	return t, nil
}
