package tensor

import "errors"

// errNonContigCUDAToCPU is returned by toCPU when the source CUDA tensor has
// non-default strides. Support for strided CUDA→CPU copy lands in Phase 3.
var errNonContigCUDAToCPU = errors.New("tensor.To(CPU): non-contiguous CUDA tensors not yet supported")

// errNoCUDA is returned whenever CUDA functionality is requested at runtime
// but the binary was built without the "cuda" build tag.
var errNoCUDA = errors.New("tensor: CUDA support not compiled (build with -tags cuda)")

// errContigNoCUDA is returned when Contiguous is asked to process a CUDA
// tensor but no CUDA handler has been registered (cuda build tag required,
// and the kernels/cuda package must be imported — ops imports it).
var errContigNoCUDA = errors.New("tensor.Contiguous: no CUDA handler registered (import ops or kernels/cuda)")

// errNoWebGPU is returned whenever WebGPU functionality is requested at
// runtime but the binary was built without the "webgpu" build tag.
var errNoWebGPU = errors.New("tensor: WebGPU support not compiled (built with -tags nowebgpu)")
