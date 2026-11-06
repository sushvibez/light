package registry

import (
	"fmt"
	"sync"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// Key identifies a kernel by operation name, device kind, and element type.
type Key struct {
	Op    string
	Dev   device.Kind
	DType dtype.DType
}

// String renders a human-readable form for error messages.
func (k Key) String() string {
	return fmt.Sprintf("%s/%v/%v", k.Op, k.Dev, k.DType)
}

// Context is the bundle a kernel receives: inputs, the pre-allocated output,
// and an opaque attribute map for op-specific scalar arguments.
type Context struct {
	Inputs []*tensor.Tensor
	Output *tensor.Tensor
	Attrs  map[string]any
}

// Kernel is the signature every registered op implementation satisfies.
type Kernel func(ctx *Context) error

var (
	mu       sync.RWMutex
	kernels  = make(map[Key]Kernel)
	regEvent = make(chan Key, 256) //nolint:unused // reserved for later introspection
)

// Register inserts a kernel into the dispatch table. Duplicate registration
// for the same Key panics — each op/device/dtype triple must have exactly
// one implementation.
func Register(k Key, fn Kernel) {
	mu.Lock()
	defer mu.Unlock()
	if _, exists := kernels[k]; exists {
		panic(fmt.Sprintf("registry: duplicate kernel for %s", k))
	}
	kernels[k] = fn
}

// Lookup finds a kernel. Returns ErrKernelNotFound if no match.
func Lookup(k Key) (Kernel, error) {
	mu.RLock()
	defer mu.RUnlock()
	fn, ok := kernels[k]
	if !ok {
		return nil, fmt.Errorf("registry: %w: %s", ErrKernelNotFound, k)
	}
	return fn, nil
}

// ErrKernelNotFound is returned by Lookup when no kernel is registered.
var ErrKernelNotFound = errorString("kernel not found")

// errorString is a lightweight error sentinel type.
type errorString string

func (e errorString) Error() string { return string(e) }

// Registered returns a snapshot of currently-registered keys, useful for
// diagnostics and tests.
func Registered() []Key {
	mu.RLock()
	defer mu.RUnlock()
	out := make([]Key, 0, len(kernels))
	for k := range kernels {
		out = append(out, k)
	}
	return out
}
