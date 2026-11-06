package models

import (
	"bufio"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"testing"
)

// hostMemAvailableGB reads MemAvailable from /proc/meminfo (0 when unknown).
func hostMemAvailableGB() float64 {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if strings.HasPrefix(sc.Text(), "MemAvailable:") {
			fs := strings.Fields(sc.Text())
			if len(fs) >= 2 {
				kb, _ := strconv.ParseFloat(fs[1], 64)
				return kb / (1 << 20)
			}
		}
	}
	return 0
}

// requireHostMemGB skips the test when less than gb of host memory is
// available: model loads stage whole checkpoints on the CPU first, and an
// OOM takes the IDE down with the test. Tests of models ≥ 2B params must
// also be gated on LIGHT_HEAVY_TESTS=1.
func requireHostMemGB(t *testing.T, gb float64) {
	t.Helper()
	if avail := hostMemAvailableGB(); avail > 0 && avail < gb {
		t.Skipf("need %.0f GB host memory, %.1f GB available", gb, avail)
	}
}

// releaseModelMemory hands the test's model memory back to the OS when it
// ends; without it the test binary keeps every checkpoint it loaded resident
// until the package finishes.
func releaseModelMemory(t *testing.T) {
	t.Helper()
	t.Cleanup(func() {
		runtime.GC()
		debug.FreeOSMemory()
	})
}
