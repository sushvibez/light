package engine

import (
	"os"
	"runtime"
	"runtime/debug"
	"testing"

	"light/internal/hostmem"
)

// requireHostMemGB (as in models/testmem_test.go) skips the test when less than gb of host memory is
// available: model loads stage whole checkpoints on the CPU first, and an
// OOM takes the IDE down with the test. When the platform cannot report it
// the test is skipped too, unless LIGHT_IGNORE_HOST_MEM=1. Tests of models
// ≥ 2B params must also be gated on LIGHT_HEAVY_TESTS=1.
func requireHostMemGB(t *testing.T, gb float64) {
	t.Helper()
	avail := hostmem.AvailableGB()
	switch {
	case avail == 0 && os.Getenv("LIGHT_IGNORE_HOST_MEM") != "1":
		t.Skipf("need %.0f GB host memory; this platform does not report it (LIGHT_IGNORE_HOST_MEM=1 runs anyway)", gb)
	case avail > 0 && avail < gb:
		t.Skipf("need %.0f GB host memory, %.1f GB available", gb, avail)
	}
}

func releaseModelMemory(t *testing.T) {
	t.Helper()
	t.Cleanup(func() {
		runtime.GC()
		debug.FreeOSMemory()
	})
}
