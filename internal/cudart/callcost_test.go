//go:build !nocuda && cgo

package cudart_test

import (
	"os"
	"testing"
	"time"
	"unsafe"

	"light/internal/cudart"
)

// TestCallCosts prints the host-side cost of one driver / runtime call of
// each kind Light issues on the decode path (LIGHT_HEAVY_TESTS=1 to run).
// The numbers multiply with the per-step counts from
// models.TestDecodeStepProfile into a per-token overhead budget.
func TestCallCosts(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1")
	}
	if n, err := cudart.DeviceCount(); err != nil || n == 0 {
		t.Skip("no CUDA device")
	}
	if err := cudart.SetDevice(0); err != nil {
		t.Fatal(err)
	}
	const N = 2000
	const bytes = 8192
	a, _ := cudart.Malloc(bytes)
	b, _ := cudart.Malloc(bytes)
	host := make([]byte, 256)
	pinned, _ := cudart.MallocHost(256)
	ptx, err := os.ReadFile("../../kernels/cuda/ptx/elementwise.ptx")
	if err != nil {
		t.Fatal(err)
	}
	mod, err := cudart.ModuleLoadData(ptx)
	if err != nil {
		t.Fatal(err)
	}
	fn, err := mod.GetFunction("add_f32")
	if err != nil {
		t.Fatal(err)
	}
	n := int32(bytes / 4)
	args := []unsafe.Pointer{unsafe.Pointer(&a), unsafe.Pointer(&b), unsafe.Pointer(&a), unsafe.Pointer(&n)}
	h, err := cudart.GetBlasHandle(0)
	if err != nil {
		t.Fatal(err)
	}
	w, _ := cudart.Malloc(2048 * 2048 * 2)
	x, _ := cudart.Malloc(2048 * 2)
	y, _ := cudart.Malloc(2048 * 4)

	run := func(name string, f func()) {
		f() // warm
		_ = cudart.DeviceSynchronize()
		t0 := time.Now()
		for i := 0; i < N; i++ {
			f()
		}
		issue := time.Since(t0)
		_ = cudart.DeviceSynchronize()
		total := time.Since(t0)
		t.Logf("%-34s issue %6.2f µs/call, incl. GPU drain %6.2f µs/call", name, float64(issue.Microseconds())/N, float64(total.Microseconds())/N)
	}
	run("cudaMalloc+cudaFree 8 KB", func() { p, _ := cudart.Malloc(bytes); _ = cudart.Free(p) })
	run("cudaMemset 8 KB", func() { _ = cudart.Memset(a, 0, bytes) })
	run("cudaMemcpy DtoD 8 KB", func() { _ = cudart.MemcpyDtoD(b, a, bytes) })
	run("cudaMemcpy2D DtoD 8 KB (1 row)", func() { _ = cudart.Memcpy2DDtoD(b, bytes, a, bytes, bytes, 1) })
	run("cudaMemcpy HtoD 256 B (pageable)", func() { _ = cudart.MemcpyHtoD(a, host) })
	run("cudaMemcpyAsync HtoD 256 B (pinned)", func() { _ = cudart.MemcpyHtoDAsync(a, pinned, 256) })
	run("cudaMemcpy DtoH 256 B", func() { _ = cudart.MemcpyDtoH(host, a) })
	run("cuLaunchKernel add_f32 2048", func() { _ = cudart.LaunchKernel(fn, 8, 1, 1, 256, 1, 1, 0, 0, args) })
	run("cublasGemmEx BF16 1x2048x2048", func() { _ = cudart.GemmExBF16RowMajorTransB(h, 1, 2048, 2048, x, w, y) })
	run("cudaDeviceSynchronize (idle)", func() { _ = cudart.DeviceSynchronize() })
}
