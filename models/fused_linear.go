package models

import (
	"os"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/tensor"
)

// fusedRow reports whether a decode step can run these linears through
// the fused GEMVs (ops.LinearMulti / ops.GatedLinear): one BF16
// activation row on CUDA, plain BF16 weights without bias or scales, no
// autograd. Prefill, CPU and quantised weights keep nn.Linear.
func fusedRow(x *tensor.Tensor, ls ...*nn.Linear) bool {
	if fusedOff {
		return false
	}
	if x == nil || !device.IsCUDA(x.Device()) || x.DType() != dtype.BF16 || x.RequiresGrad() || x.NumElements() != x.Dim(-1) {
		return false
	}
	bytes := 0
	for _, l := range ls {
		if l == nil || l.W == nil || l.B != nil || l.W.Rank() != 2 || l.W.DType() != dtype.BF16 || l.WScale != 0 || l.WBlockScale != nil || l.W.RequiresGrad() {
			return false
		}
		bytes += l.W.NumElements() * 2
	}
	// Where it measured: at K = 1024 (Qwen3.5-0.8B: 5.12 → 5.0 ms per
	// step) the fused kernels win on launch count and bandwidth; at
	// K = 2048 (Llama-3.2-1B) they lose 0.17 ms per step to cuBLAS's GEMV
	// even for the 12.6 MB q / k / v trio, with the same microbenchmark
	// bandwidth, so the kernel's K = 2048 streaming is a follow-up and
	// cuBLAS keeps those shapes. Weights above 32 MB per call stream
	// from DRAM where cuBLAS also keeps a small edge.
	return x.Dim(-1) <= fusedMaxK && bytes <= fusedMaxBytes
}

const (
	fusedMaxK     = 1024
	fusedMaxBytes = 32 << 20
)

// fusedOff disables the fused decode GEMVs (LIGHT_FUSED_GEMV=0) for A/B
// timing against nn.Linear.
var fusedOff = os.Getenv("LIGHT_FUSED_GEMV") == "0"
