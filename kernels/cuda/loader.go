//go:build !nocuda && cgo

package cuda

import (
	_ "embed"
	"fmt"
	"sync"
	"unsafe"

	"light/internal/cudart"
)

//go:embed ptx/elementwise.ptx
var elementwisePTX []byte

//go:embed ptx/dequant.ptx
var dequantPTX []byte

//go:embed ptx/cnn_backward.ptx
var cnnBackwardPTX []byte

//go:embed ptx/batchnorm.ptx
var batchnormPTX []byte

//go:embed ptx/bf16_ops.ptx
var bf16OpsPTX []byte

//go:embed ptx/groupnorm.ptx
var groupnormPTX []byte

//go:embed ptx/rope_scaled.ptx
var ropeScaledPTX []byte

//go:embed ptx/shape_ops.ptx
var shapeOpsPTX []byte

//go:embed ptx/conv3d.ptx
var conv3dPTX []byte

//go:embed ptx/flash_attn.ptx
var flashAttnPTX []byte

//go:embed ptx/matmul_int8.ptx
var matmulInt8PTX []byte

//go:embed ptx/matmul_q.ptx
var matmulQPTX []byte

//go:embed ptx/sort.ptx
var sortPTX []byte

//go:embed ptx/conv_transpose.ptx
var convTransposePTX []byte

//go:embed ptx/index_ops.ptx
var indexOpsPTX []byte

//go:embed ptx/moe.ptx
var moePTX []byte

//go:embed ptx/moe_gguf.ptx
var moeGGUFPTX []byte

//go:embed ptx/gated_delta.ptx
var gatedDeltaPTX []byte

//go:embed ptx/fp8_block.ptx
var fp8BlockPTX []byte

//go:embed ptx/int4.ptx
var int4PTX []byte

//go:embed ptx/kv_ring.ptx
var kvRingPTX []byte

type lazyModule struct {
	name string
	data []byte
	mu   sync.Mutex
	mod  *cudart.Module
}

// load JITs the module on first use. A failed load is not cached: the
// driver can reject cuModuleLoadDataEx with CUDA_ERROR_INVALID_CONTEXT on a
// thread whose context is stale, so we re-bind the primary context and
// retry once, and leave the slot empty for the next caller if that fails
// too (otherwise every later kernel lookup would report "not found").
func (lm *lazyModule) load() (*cudart.Module, error) {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	if lm.mod != nil {
		return lm.mod, nil
	}
	mod, err := cudart.ModuleLoadData(lm.data)
	if err != nil {
		if bindErr := cudart.BindPrimaryContext(); bindErr != nil {
			return nil, fmt.Errorf("module %s: %w (rebind: %v)", lm.name, err, bindErr)
		}
		if mod, err = cudart.ModuleLoadData(lm.data); err != nil {
			return nil, fmt.Errorf("module %s: %w", lm.name, err)
		}
	}
	lm.mod = mod
	return mod, nil
}

var modules = []*lazyModule{
	{name: "elementwise", data: elementwisePTX},
	{name: "dequant", data: dequantPTX},
	{name: "cnn_backward", data: cnnBackwardPTX},
	{name: "batchnorm", data: batchnormPTX},
	{name: "bf16_ops", data: bf16OpsPTX},
	{name: "groupnorm", data: groupnormPTX},
	{name: "rope_scaled", data: ropeScaledPTX},
	{name: "shape_ops", data: shapeOpsPTX},
	{name: "conv3d", data: conv3dPTX},
	{name: "flash_attn", data: flashAttnPTX},
	{name: "matmul_int8", data: matmulInt8PTX},
	{name: "matmul_q", data: matmulQPTX},
	{name: "sort", data: sortPTX},
	{name: "conv_transpose", data: convTransposePTX},
	{name: "index_ops", data: indexOpsPTX},
	{name: "moe", data: moePTX},
	{name: "moe_gguf", data: moeGGUFPTX},
	{name: "gated_delta", data: gatedDeltaPTX},
	{name: "fp8_block", data: fp8BlockPTX},
	{name: "int4", data: int4PTX},
	{name: "kv_ring", data: kvRingPTX},
}

var (
	funcCacheMu sync.RWMutex
	funcCache   = make(map[string]*cudart.Function)
)

// GetKernel returns a cached Function for the given kernel name, searching
// each embedded PTX module in order.
func GetKernel(name string) (*cudart.Function, error) {
	funcCacheMu.RLock()
	fn, ok := funcCache[name]
	funcCacheMu.RUnlock()
	if ok {
		return fn, nil
	}
	var lastErr, loadErr error
	for _, lm := range modules {
		mod, err := lm.load()
		if err != nil {
			loadErr = err
			continue
		}
		fn, err := mod.GetFunction(name)
		if err != nil {
			lastErr = err
			continue
		}
		funcCacheMu.Lock()
		if existing, exists := funcCache[name]; exists {
			funcCacheMu.Unlock()
			return existing, nil
		}
		funcCache[name] = fn
		funcCacheMu.Unlock()
		return fn, nil
	}
	if loadErr != nil {
		// A module that failed to load is the more likely explanation
		// than a genuinely missing symbol.
		return nil, fmt.Errorf("kernel %q: %w", name, loadErr)
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("kernel %q not found", name)
	}
	return nil, lastErr
}

// Launch1D runs fn with a 1-D grid sized to cover n elements, 256 threads/block.
// args is a slice of pointers to each argument value.
func Launch1D(fn *cudart.Function, n uint64, args []unsafe.Pointer) error {
	const block = 256
	grid := uint32((n + block - 1) / block)
	if grid == 0 {
		grid = 1
	}
	return cudart.LaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, args)
}
