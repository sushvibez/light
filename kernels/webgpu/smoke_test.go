//go:build webgpu

package webgpu_test

import (
	"testing"

	"light/device"
	"light/internal/wgpudev"
	"light/ops"
	"light/tensor"
)

func TestDeviceLimits(t *testing.T) {
	d, err := wgpudev.Get()
	if err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	t.Logf("adapter=%q backend=%s", d.AdapterName, d.BackendName)
	t.Logf("granted MaxBufferSize=%d bytes (%d MiB), MaxStorageBufferBindingSize=%d bytes (%d MiB), MaxWorkgroupsPerDim=%d",
		d.MaxBufferSize, d.MaxBufferSize>>20, d.MaxStorageBufferBindingSize, d.MaxStorageBufferBindingSize>>20, d.MaxWorkgroupsPerDim)
	if d.MaxBufferSize < 256<<20 {
		t.Errorf("granted MaxBufferSize %d below the 256 MiB WebGPU default", d.MaxBufferSize)
	}
	if err := d.CheckSize(d.MaxBufferSize + 1); err == nil {
		t.Error("CheckSize should reject an allocation above the granted limit")
	}
}

func TestSmokeAdd(t *testing.T) {
	gpu, err := device.WebGPU(0)
	if err != nil {
		t.Skipf("no WebGPU: %v", err)
	}
	a, _ := tensor.New([]float32{1, 2, 3, 4}, []int{4}, device.CPU())
	b, _ := tensor.New([]float32{10, 20, 30, 40}, []int{4}, device.CPU())
	aG, err := a.To(gpu)
	if err != nil {
		t.Fatal(err)
	}
	bG, err := b.To(gpu)
	if err != nil {
		t.Fatal(err)
	}
	out, err := ops.Add(aG, bG)
	if err != nil {
		t.Fatal(err)
	}
	back, err := out.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](back)
	want := []float32{11, 22, 33, 44}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v want %v", got, want)
		}
	}
}
