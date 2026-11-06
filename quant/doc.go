// Package quant contains quantization schemes and the CPU + CUDA kernels that
// dequantize or directly matmul against quantized weights: Q4_0, Q4_1, Q5_0,
// Q5_1, Q8_0, Q8_1, and the K-quants Q2K…Q8K used by llama.cpp / GGUF.
package quant
