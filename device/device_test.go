package device_test

import (
	"errors"
	"testing"

	"light/device"
)

func TestCPU(t *testing.T) {
	d := device.CPU()
	if d.Kind() != device.KindCPU {
		t.Errorf("Kind() = %v, want KindCPU", d.Kind())
	}
	if d.Ordinal() != 0 {
		t.Errorf("Ordinal() = %d, want 0", d.Ordinal())
	}
	if d.String() != "CPU" {
		t.Errorf("String() = %q, want \"CPU\"", d.String())
	}
	if !d.SameAs(device.CPU()) {
		t.Error("CPU should SameAs CPU")
	}
}

func TestKindString(t *testing.T) {
	if device.KindCPU.String() != "CPU" {
		t.Error("KindCPU.String()")
	}
	if device.KindCUDA.String() != "CUDA" {
		t.Error("KindCUDA.String()")
	}
	if device.Kind(99).String() == "" {
		t.Error("invalid kind string empty")
	}
}

func TestIsCPUCUDA(t *testing.T) {
	cpu := device.CPU()
	if !device.IsCPU(cpu) {
		t.Error("IsCPU(cpu)")
	}
	if device.IsCUDA(cpu) {
		t.Error("IsCUDA(cpu) should be false")
	}
	if device.IsCPU(nil) {
		t.Error("IsCPU(nil) should be false")
	}
}

func TestSameAsNil(t *testing.T) {
	if device.CPU().SameAs(nil) {
		t.Error("CPU.SameAs(nil) should be false")
	}
}

func TestCUDABehaviour(t *testing.T) {
	d, err := device.CUDA(0)
	if !device.CUDAAvailable() {
		if !errors.Is(err, device.ErrCUDANotBuilt) {
			t.Errorf("without cuda tag, expected ErrCUDANotBuilt, got %v", err)
		}
		if d != nil {
			t.Error("device should be nil when error returned")
		}
		return
	}
	// With cuda tag: we get a device.
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d.Kind() != device.KindCUDA {
		t.Errorf("Kind() = %v, want KindCUDA", d.Kind())
	}
	if d.Ordinal() != 0 {
		t.Errorf("Ordinal() = %d", d.Ordinal())
	}
	if d.String() != "CUDA(0)" {
		t.Errorf("String() = %q, want CUDA(0)", d.String())
	}
	// Negative ordinal → error.
	if _, err := device.CUDA(-1); err == nil {
		t.Error("expected error for -1")
	}
}

func TestMustCUDAFailsWithoutTag(t *testing.T) {
	defer func() {
		if !device.CUDAAvailable() {
			if r := recover(); r == nil {
				t.Error("expected panic on MustCUDA without cuda tag")
			}
		} else {
			// cuda tag present: should not panic for ordinal 0
			if r := recover(); r != nil {
				t.Errorf("unexpected panic with cuda tag: %v", r)
			}
		}
	}()
	_ = device.MustCUDA(0)
}

func TestWebGPUBehaviour(t *testing.T) {
	d, err := device.WebGPU(0)
	if !device.WebGPUAvailable() {
		if !errors.Is(err, device.ErrWebGPUNotBuilt) {
			t.Fatalf("expected ErrWebGPUNotBuilt without the webgpu tag, got %v", err)
		}
		return
	}
	if err != nil {
		t.Skipf("webgpu compiled in but no adapter: %v", err)
	}
	if d.Kind() != device.KindWebGPU || !device.IsWebGPU(d) || d.Ordinal() != 0 {
		t.Errorf("unexpected WebGPU device %v", d)
	}
	if d.String() != "WebGPU(0)" {
		t.Errorf("String() = %q", d.String())
	}
	d2, _ := device.WebGPU(0)
	if !d.SameAs(d2) {
		t.Error("WebGPU(0) should SameAs WebGPU(0)")
	}
	if d.SameAs(device.CPU()) {
		t.Error("WebGPU should not SameAs CPU")
	}
	if _, err := device.WebGPU(1); err == nil {
		t.Error("WebGPU(1) should be out of range")
	}
	if device.KindWebGPU.String() != "WebGPU" {
		t.Error("KindWebGPU.String()")
	}
}
