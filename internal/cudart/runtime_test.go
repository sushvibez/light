//go:build !nocuda && cgo

package cudart_test

import (
	"testing"
	"unsafe"

	"light/internal/cudart"
)

func TestDeviceCount(t *testing.T) {
	n, err := cudart.DeviceCount()
	if err != nil {
		t.Fatal(err)
	}
	if n < 1 {
		t.Skipf("no CUDA device visible (count=%d)", n)
	}
	t.Logf("CUDA devices: %d", n)
}

func TestDeviceProperties(t *testing.T) {
	n, _ := cudart.DeviceCount()
	if n < 1 {
		t.Skip("no GPU")
	}
	p, err := cudart.GetDeviceProperties(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("GPU 0: %s (sm_%d%d, %d SMs, %d MB)", p.Name, p.ComputeMajor, p.ComputeMinor, p.MultiProcCount, p.TotalGlobalMem/(1<<20))
	if p.ComputeMajor < 3 {
		t.Errorf("unexpectedly old compute capability %d.%d", p.ComputeMajor, p.ComputeMinor)
	}
}

func TestMemInfo(t *testing.T) {
	n, _ := cudart.DeviceCount()
	if n < 1 {
		t.Skip("no GPU")
	}
	if err := cudart.SetDevice(0); err != nil {
		t.Fatal(err)
	}
	free, total, err := cudart.MemInfo()
	if err != nil {
		t.Fatal(err)
	}
	if total == 0 {
		t.Error("total == 0")
	}
	if free > total {
		t.Errorf("free %d > total %d", free, total)
	}
	t.Logf("Free: %d MB / Total: %d MB", free/(1<<20), total/(1<<20))
}

func TestMallocFree(t *testing.T) {
	if n, _ := cudart.DeviceCount(); n < 1 {
		t.Skip("no GPU")
	}
	if err := cudart.SetDevice(0); err != nil {
		t.Fatal(err)
	}
	ptr, err := cudart.Malloc(1024)
	if err != nil {
		t.Fatal(err)
	}
	if ptr == 0 {
		t.Error("Malloc returned 0")
	}
	if err := cudart.Free(ptr); err != nil {
		t.Fatal(err)
	}
	// Double-free safety: freeing 0 is OK.
	if err := cudart.Free(0); err != nil {
		t.Errorf("Free(0) error: %v", err)
	}
}

func TestMemcpyRoundTrip(t *testing.T) {
	if n, _ := cudart.DeviceCount(); n < 1 {
		t.Skip("no GPU")
	}
	if err := cudart.SetDevice(0); err != nil {
		t.Fatal(err)
	}
	src := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	ptr, err := cudart.Malloc(uintptr(len(src)))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cudart.Free(ptr) }()

	if err := cudart.MemcpyHtoD(ptr, src); err != nil {
		t.Fatal(err)
	}
	dst := make([]byte, len(src))
	if err := cudart.MemcpyDtoH(dst, ptr); err != nil {
		t.Fatal(err)
	}
	for i := range src {
		if dst[i] != src[i] {
			t.Errorf("round-trip mismatch at %d: got %d, want %d", i, dst[i], src[i])
		}
	}
}

func TestMemset(t *testing.T) {
	if n, _ := cudart.DeviceCount(); n < 1 {
		t.Skip("no GPU")
	}
	if err := cudart.SetDevice(0); err != nil {
		t.Fatal(err)
	}
	ptr, err := cudart.Malloc(16)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cudart.Free(ptr) }()

	if err := cudart.Memset(ptr, 0xAB, 16); err != nil {
		t.Fatal(err)
	}
	dst := make([]byte, 16)
	if err := cudart.MemcpyDtoH(dst, ptr); err != nil {
		t.Fatal(err)
	}
	for i, b := range dst {
		if b != 0xAB {
			t.Errorf("memset[%d] = %#x", i, b)
		}
	}
}

func TestEnsureDriver(t *testing.T) {
	if err := cudart.EnsureDriver(); err != nil {
		t.Fatal(err)
	}
	// Idempotent.
	if err := cudart.EnsureDriver(); err != nil {
		t.Fatal(err)
	}
}

// Round-trip with a kernel launch — loads PTX, runs add_f32, verifies.
func TestLaunchAddF32(t *testing.T) {
	if n, _ := cudart.DeviceCount(); n < 1 {
		t.Skip("no GPU")
	}
	if err := cudart.SetDevice(0); err != nil {
		t.Fatal(err)
	}
	if err := cudart.EnsureDriver(); err != nil {
		t.Fatal(err)
	}
	// Load PTX from disk (Phase 2 wires go:embed later).
	ptx, err := readFile("../../kernels/cuda/ptx/elementwise.ptx")
	if err != nil {
		t.Skipf("PTX not built: %v", err)
	}
	mod, err := cudart.ModuleLoadData(ptx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = mod.Unload() }()
	fn, err := mod.GetFunction("add_f32")
	if err != nil {
		t.Fatal(err)
	}

	// Allocate three device buffers: a, b, out (each 8 float32 = 32 bytes).
	const n = 8
	const sz = n * 4
	ap, _ := cudart.Malloc(sz)
	bp, _ := cudart.Malloc(sz)
	op, _ := cudart.Malloc(sz)
	defer cudart.Free(ap)
	defer cudart.Free(bp)
	defer cudart.Free(op)

	// Upload inputs.
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	b := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	cudart.MemcpyHtoD(ap, floatsToBytes(a))
	cudart.MemcpyHtoD(bp, floatsToBytes(b))

	// Launch.
	N := uint64(n)
	args := []unsafe.Pointer{
		unsafe.Pointer(&ap),
		unsafe.Pointer(&bp),
		unsafe.Pointer(&op),
		unsafe.Pointer(&N),
	}
	err = cudart.LaunchKernel(fn, 1, 1, 1, n, 1, 1, 0, 0, args)
	if err != nil {
		t.Fatal(err)
	}
	if err := cudart.DeviceSynchronize(); err != nil {
		t.Fatal(err)
	}

	// Download and verify.
	out := make([]float32, n)
	cudart.MemcpyDtoH(bytesFromFloats(out), op)
	want := []float32{11, 22, 33, 44, 55, 66, 77, 88}
	for i := range want {
		if out[i] != want[i] {
			t.Errorf("out[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}

// helpers
func floatsToBytes(f []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&f[0])), len(f)*4)
}

func bytesFromFloats(f []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&f[0])), len(f)*4)
}

func readFile(path string) ([]byte, error) {
	return osReadFile(path)
}
