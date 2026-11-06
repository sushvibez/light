//go:build !nocuda && cgo

package storage_test

import (
	"testing"

	"light/device"
	"light/dtype"
	"light/storage"
)

func TestNewCUDAFromPointer(t *testing.T) {
	dev, err := device.CUDA(0)
	if err != nil {
		t.Skip("CUDA(0) not available at test time:", err)
	}
	s := storage.NewCUDAFromPointer(dev, dtype.F32, 10, 40, 0xdeadbeef, nil)
	if s.Device().Kind() != device.KindCUDA {
		t.Error("expected CUDA storage")
	}
	if s.DType() != dtype.F32 {
		t.Error("wrong dtype")
	}
	if s.Len() != 10 {
		t.Error("wrong length")
	}
	if s.Bytes() != 40 {
		t.Error("wrong byte count")
	}
	if s.CUDAPtr() != 0xdeadbeef {
		t.Errorf("CUDAPtr = %#x, want 0xdeadbeef", s.CUDAPtr())
	}
}

func TestCUDAReleaseCallbackFires(t *testing.T) {
	dev, err := device.CUDA(0)
	if err != nil {
		t.Skip("CUDA(0) not available at test time:", err)
	}
	released := false
	s := storage.NewCUDAFromPointer(dev, dtype.F32, 1, 4, 0x1000, func() {
		released = true
	})
	s.Acquire()
	s.Release()
	if !released {
		t.Error("release callback should have fired")
	}
}

func TestCPUBytesPanicOnCUDA(t *testing.T) {
	dev, err := device.CUDA(0)
	if err != nil {
		t.Skip("CUDA not available:", err)
	}
	s := storage.NewCUDAFromPointer(dev, dtype.F32, 1, 4, 0x1000, nil)
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()
	_ = s.CPUBytes()
}

func TestCUDAPtrPanicOnCPU(t *testing.T) {
	s := storage.NewCPU(dtype.F32, 1)
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()
	_ = s.CUDAPtr()
}
