//go:build !nocuda && cgo

package ops_test

import (
	"testing"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/ops"
	"light/tensor"
)

// Per-op issue cost on CUDA for decode-sized tensors: how much of a launch
// is Light's own overhead (allocation, registry, pinning) versus the driver
// call measured by cudart's TestCallCosts.
func BenchmarkOpIssueCUDA(b *testing.B) {
	gpu, err := device.CUDA(0)
	if err != nil {
		b.Skip(err)
	}
	x, _ := tensor.Zeros(dtype.BF16, []int{1, 1, 1024}, gpu)
	y, _ := tensor.Zeros(dtype.BF16, []int{1, 1, 1024}, gpu)
	w, _ := tensor.Zeros(dtype.BF16, []int{1024}, gpu)
	wm, _ := tensor.Zeros(dtype.BF16, []int{1024, 1024}, gpu)
	x2, _ := x.Reshape(1, 1024)
	run := func(name string, f func() error) {
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := f(); err != nil {
					b.Fatal(err)
				}
			}
			cudart.DeviceSynchronize()
		})
	}
	run("add_bf16", func() error { _, err := ops.Add(x, y); return err })
	run("affine_bf16", func() error { _, err := ops.Affine(x, 0.5, 0); return err })
	run("rmsnorm_bf16", func() error { _, err := ops.RMSNorm(x, w, 1e-6); return err })
	run("matmul_1x1024x1024_bf16", func() error { _, err := ops.MatMul(x2, wm); return err })
	run("empty_bf16", func() error { _, err := tensor.Empty(dtype.BF16, []int{1, 1, 1024}, gpu); return err })
}
