package tensor

import (
	"fmt"

	"light/device"
)

// CopyFrom overwrites dst's storage contents with src's. Both tensors must
// have matching shape, dtype, and device. Works on CPU (byte copy) and CUDA
// (via cudaMemcpyDtoD through the registered hook).
//
// This is the primitive in-place update used by optimizers: after computing
// `new_p = p - lr * grad` as a fresh tensor, call `p.CopyFrom(new_p)` to
// mutate p's underlying storage in place so any view or gradFn still
// referencing `p` observes the updated values.
func (t *Tensor) CopyFrom(src *Tensor) error {
	if !t.Shape().Equal(src.Shape()) {
		return fmt.Errorf("tensor.CopyFrom: shape mismatch (%v vs %v)", t.Shape(), src.Shape())
	}
	if t.DType() != src.DType() {
		return fmt.Errorf("tensor.CopyFrom: dtype mismatch (%v vs %v)", t.DType(), src.DType())
	}
	srcC, err := src.Contiguous()
	if err != nil {
		return err
	}
	dst := t
	// Require dst to be contiguous too for the simple byte-level copy.
	if !dst.IsContiguous() || dst.offset != 0 {
		return fmt.Errorf("tensor.CopyFrom: dst must be contiguous with zero offset")
	}
	if !t.Device().SameAs(src.Device()) {
		// Cross-device: stream bytes directly into the existing destination
		// storage so we don't allocate a transient that has to wait for GC.
		// Loading 3B-param checkpoints onto a 16 GB GPU can pile up enough
		// per-layer transients to OOM otherwise.
		if src.Device().Kind() != device.KindCPU {
			return fmt.Errorf("tensor.CopyFrom: cross-device copy only supports a CPU source (got %v → %v)", src.Device(), t.Device())
		}
		fn := copyHostFns[t.Device().Kind()]
		if fn == nil {
			return fmt.Errorf("tensor.CopyFrom: no host→%v copy hook registered (build with the matching tag and import ops)", t.Device().Kind())
		}
		return fn(dst, srcC)
	}
	switch kind := t.Device().Kind(); kind {
	case device.KindCPU:
		dstBuf := dst.storage.CPUBytes()
		srcBuf := srcC.storage.CPUBytes()
		copy(dstBuf, srcBuf)
		return nil
	default:
		fn := copyFns[kind]
		if fn == nil {
			return fmt.Errorf("tensor.CopyFrom: no %v device-copy hook registered (build with the matching tag and import ops)", kind)
		}
		return fn(dst, srcC)
	}
}

// CopyFunc is a device-copy implementation registered by an accelerator
// kernel package.
type CopyFunc func(dst, src *Tensor) error

// CUDACopyFunc is the historical name of CopyFunc.
type CUDACopyFunc = CopyFunc

var (
	copyFns     = map[device.Kind]CopyFunc{} // dev-to-dev on the same device kind
	copyHostFns = map[device.Kind]CopyFunc{} // CPU-src into existing dst storage (in place)
)

// RegisterCopy wires the device-to-device copy for a device kind.
func RegisterCopy(kind device.Kind, fn CopyFunc) { copyFns[kind] = fn }

// RegisterCopyFromHost wires the CPU→device in-place copy for a device
// kind. Required for CopyFrom across devices; lets the loader stream bytes
// into model parameters without allocating a transient GPU tensor per
// weight.
func RegisterCopyFromHost(kind device.Kind, fn CopyFunc) { copyHostFns[kind] = fn }

// RegisterCUDACopy wires the CUDA device-to-device copy.
func RegisterCUDACopy(fn CUDACopyFunc) { RegisterCopy(device.KindCUDA, fn) }

// RegisterCUDACopyFromHost wires the CPU→CUDA in-place copy.
func RegisterCUDACopyFromHost(fn CUDACopyFunc) { RegisterCopyFromHost(device.KindCUDA, fn) }
