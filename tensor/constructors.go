package tensor

import (
	"fmt"
	"math"
	"unsafe"

	"light/device"
	"light/dtype"
)

// tritriHook is installed by `ops` (or another higher-level package) so that
// tensor.Tril/Triu can dispatch through the op registry without creating an
// import cycle.
type tritriFn func(t *Tensor, k int, op string) (*Tensor, error)

var tritriHookFn tritriFn

// RegisterTritriHook installs the Tril/Triu dispatcher. Called by the ops
// package at init.
func RegisterTritriHook(fn tritriFn) { tritriHookFn = fn }

func tritri(t *Tensor, k int, op string) (*Tensor, error) {
	if tritriHookFn == nil {
		return nil, fmt.Errorf("tensor: %s hook not registered — import the ops package", op)
	}
	return tritriHookFn(t, k, op)
}

// Linspace returns n equally-spaced values in [start, stop]. If n <= 1, the
// output contains just `start`. F32/F64 only.
func Linspace(start, stop float64, n int, dt dtype.DType, dev device.Device) (*Tensor, error) {
	if n < 0 {
		return nil, fmt.Errorf("Linspace: n must be non-negative")
	}
	if dt != dtype.F32 && dt != dtype.F64 {
		return nil, fmt.Errorf("Linspace: dtype %v unsupported (F32/F64)", dt)
	}
	cpuT, err := Zeros(dt, []int{n}, device.CPU())
	if err != nil {
		return nil, err
	}
	if n == 0 {
		if dev.Kind() == device.KindCPU {
			return cpuT, nil
		}
		return cpuT.To(dev)
	}
	buf := cpuT.storage.CPUBytes()
	step := 0.0
	if n > 1 {
		step = (stop - start) / float64(n-1)
	}
	switch dt {
	case dtype.F32:
		vals := unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), n)
		for i := range n {
			vals[i] = float32(start + float64(i)*step)
		}
		if n > 1 {
			vals[n-1] = float32(stop) // avoid rounding drift
		}
	case dtype.F64:
		vals := unsafe.Slice((*float64)(unsafe.Pointer(&buf[0])), n)
		for i := range n {
			vals[i] = start + float64(i)*step
		}
		if n > 1 {
			vals[n-1] = stop
		}
	}
	if dev.Kind() == device.KindCPU {
		return cpuT, nil
	}
	return cpuT.To(dev)
}

// Eye returns an (n, n) identity matrix. F32/F64 only.
func Eye(n int, dt dtype.DType, dev device.Device) (*Tensor, error) {
	if n < 0 {
		return nil, fmt.Errorf("Eye: n must be non-negative")
	}
	if dt != dtype.F32 && dt != dtype.F64 {
		return nil, fmt.Errorf("Eye: dtype %v unsupported (F32/F64)", dt)
	}
	cpuT, err := Zeros(dt, []int{n, n}, device.CPU())
	if err != nil {
		return nil, err
	}
	buf := cpuT.storage.CPUBytes()
	switch dt {
	case dtype.F32:
		vals := unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), n*n)
		for i := range n {
			vals[i*n+i] = 1
		}
	case dtype.F64:
		vals := unsafe.Slice((*float64)(unsafe.Pointer(&buf[0])), n*n)
		for i := range n {
			vals[i*n+i] = 1
		}
	}
	if dev.Kind() == device.KindCPU {
		return cpuT, nil
	}
	return cpuT.To(dev)
}

// Tril and Triu are implemented as package-level helpers instead of Tensor
// methods now, because they need device-aware dispatch. Kept here so existing
// tests that call `tensor.Tril` still compile; the actual work happens in the
// registry-dispatched kernel under kernels/cpu and kernels/cuda.

// Tril zeroes elements above the k-th diagonal of the receiver's trailing
// two dims.
func Tril(t *Tensor, k int) (*Tensor, error) {
	return tritri(t, k, "tril")
}

// Triu zeroes elements below the k-th diagonal.
func Triu(t *Tensor, k int) (*Tensor, error) {
	return tritri(t, k, "triu")
}

// Meshgrid returns a pair of 2-D tensors with broadcasted coordinate values,
// "xy" indexing style. Given 1-D inputs x (len M) and y (len N):
//
//	gridX: (N, M) — x repeated along rows
//	gridY: (N, M) — y repeated along cols
//
// Both must share dtype and device; only F32/F64 supported.
func Meshgrid(x, y *Tensor) (*Tensor, *Tensor, error) {
	if x.Rank() != 1 || y.Rank() != 1 {
		return nil, nil, fmt.Errorf("Meshgrid: inputs must be 1-D")
	}
	if x.DType() != y.DType() {
		return nil, nil, fmt.Errorf("Meshgrid: dtype mismatch")
	}
	if !x.Device().SameAs(y.Device()) {
		return nil, nil, fmt.Errorf("Meshgrid: device mismatch")
	}
	if x.DType() != dtype.F32 && x.DType() != dtype.F64 {
		return nil, nil, fmt.Errorf("Meshgrid: dtype %v unsupported", x.DType())
	}
	if x.Device().Kind() != device.KindCPU {
		return nil, nil, fmt.Errorf("Meshgrid: CUDA not supported yet")
	}
	M := x.Dim(0)
	N := y.Dim(0)
	xc, _ := x.Contiguous()
	yc, _ := y.Contiguous()
	gridX, _ := Zeros(x.DType(), []int{N, M}, x.Device())
	gridY, _ := Zeros(x.DType(), []int{N, M}, x.Device())
	switch x.DType() {
	case dtype.F32:
		xs := unsafe.Slice((*float32)(unsafe.Pointer(&xc.storage.CPUBytes()[0])), M)
		ys := unsafe.Slice((*float32)(unsafe.Pointer(&yc.storage.CPUBytes()[0])), N)
		gx := unsafe.Slice((*float32)(unsafe.Pointer(&gridX.storage.CPUBytes()[0])), N*M)
		gy := unsafe.Slice((*float32)(unsafe.Pointer(&gridY.storage.CPUBytes()[0])), N*M)
		for i := range N {
			for j := range M {
				gx[i*M+j] = xs[j]
				gy[i*M+j] = ys[i]
			}
		}
	case dtype.F64:
		xs := unsafe.Slice((*float64)(unsafe.Pointer(&xc.storage.CPUBytes()[0])), M)
		ys := unsafe.Slice((*float64)(unsafe.Pointer(&yc.storage.CPUBytes()[0])), N)
		gx := unsafe.Slice((*float64)(unsafe.Pointer(&gridX.storage.CPUBytes()[0])), N*M)
		gy := unsafe.Slice((*float64)(unsafe.Pointer(&gridY.storage.CPUBytes()[0])), N*M)
		for i := range N {
			for j := range M {
				gx[i*M+j] = xs[j]
				gy[i*M+j] = ys[i]
			}
		}
	}
	_ = math.Pi
	return gridX, gridY, nil
}
