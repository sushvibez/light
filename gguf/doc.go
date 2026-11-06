// Package gguf reads the llama.cpp GGUF model format: metadata keys and
// tensor data including quantized (Q4_0 … Q8K) blocks. Quantized tensors load
// into Tensors of the corresponding dtype and are dequantized lazily or on
// matmul via the quant package's kernels.
package gguf
