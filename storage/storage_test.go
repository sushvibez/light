package storage_test

import (
	"runtime"
	"testing"
	"time"

	"light/device"
	"light/dtype"
	"light/storage"
)

func TestNewCPU(t *testing.T) {
	s := storage.NewCPU(dtype.F32, 10)
	if s.Device().Kind() != device.KindCPU {
		t.Error("wrong device kind")
	}
	if s.DType() != dtype.F32 {
		t.Error("wrong dtype")
	}
	if s.Len() != 10 {
		t.Errorf("Len = %d, want 10", s.Len())
	}
	if s.Bytes() != 40 {
		t.Errorf("Bytes = %d, want 40", s.Bytes())
	}
	if len(s.CPUBytes()) != 40 {
		t.Errorf("CPUBytes len = %d", len(s.CPUBytes()))
	}
}

func TestNewCPUQuantized(t *testing.T) {
	// Q4_0: 32 elems per 18-byte block.
	s := storage.NewCPU(dtype.Q4_0, 64) // 2 blocks → 36 bytes
	if s.Bytes() != 36 {
		t.Errorf("Bytes = %d, want 36", s.Bytes())
	}
	if s.Len() != 64 {
		t.Errorf("Len = %d, want 64", s.Len())
	}
}

func TestWrapCPU(t *testing.T) {
	buf := make([]byte, 16)
	s := storage.WrapCPU(dtype.I32, 4, buf)
	if &s.CPUBytes()[0] != &buf[0] {
		t.Error("WrapCPU should not copy")
	}
}

func TestCPUBytesOnCPUStorage(t *testing.T) {
	s := storage.NewCPU(dtype.F32, 4)
	// Write then read back via CPUBytes.
	b := s.CPUBytes()
	b[0] = 0xab
	if s.CPUBytes()[0] != 0xab {
		t.Error("CPUBytes should alias the underlying buffer")
	}
}

func TestAcquireRelease(t *testing.T) {
	fired := false
	s := storage.NewCPU(dtype.F32, 1)
	// Manually install a release fn for testing (normally CPU has none).
	// Reach into storage via WrapCPU trick: we can't directly set release. Use
	// reference-count semantics only.
	if s.RefCount() != 0 {
		t.Errorf("initial refcount = %d, want 0", s.RefCount())
	}
	s.Acquire()
	s.Acquire()
	if s.RefCount() != 2 {
		t.Errorf("after 2 acquires, refcount = %d", s.RefCount())
	}
	s.Release()
	if s.RefCount() != 1 {
		t.Errorf("after release, refcount = %d", s.RefCount())
	}
	s.Release()
	if s.RefCount() != 0 {
		t.Errorf("after final release, refcount = %d", s.RefCount())
	}
	_ = fired
}

type trackDummy struct{ _ [16]byte }

// allocAndTrack creates a dummy that lives only for the duration of this
// function, then Tracks it. After this function returns, the Go escape
// analyzer knows the dummy is no longer referenced from a live frame, so GC
// can reclaim it.
func allocAndTrack(s *storage.Storage) {
	h := &trackDummy{}
	storage.Track(s, h)
}

func TestTrackCleanup(t *testing.T) {
	s := storage.NewCPU(dtype.F32, 1)
	allocAndTrack(s)

	if s.RefCount() != 1 {
		t.Errorf("after Track, refcount = %d, want 1", s.RefCount())
	}

	// Wait for the cleanup to run on its goroutine. Up to 3s.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		runtime.GC()
		runtime.GC()
		if s.RefCount() == 0 {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Errorf("cleanup did not run: refcount = %d", s.RefCount())
}

func TestTrackMultiple(t *testing.T) {
	s := storage.NewCPU(dtype.F32, 1)
	allocAndTrack(s)
	allocAndTrack(s)
	if s.RefCount() != 2 {
		t.Errorf("after 2 Track, refcount = %d, want 2", s.RefCount())
	}
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		runtime.GC()
		runtime.GC()
		if s.RefCount() == 0 {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Errorf("cleanup did not run: refcount = %d", s.RefCount())
}

func TestReleaseCallback(t *testing.T) {
	// Use the CUDA-internal constructor to install a custom release fn.
	// Since that constructor is unexported, simulate via Acquire / Release
	// on a stub we can inspect.
	s := storage.NewCPU(dtype.F32, 1)
	s.Acquire() // refcount = 1
	s.Release() // refcount = 0 → release callback (nil for CPU) is a no-op.
	// No direct assertion possible without a custom release; instead verify
	// no panic / no double-run.
	if s.RefCount() != 0 {
		t.Errorf("refcount = %d", s.RefCount())
	}
}
