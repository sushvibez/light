package engine

import (
	"bufio"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"testing"
)

// Same helpers as models/testmem_test.go: skip when host memory is short,
// and return model memory to the OS when the test ends.
func requireHostMemGB(t *testing.T, gb float64) {
	t.Helper()
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if strings.HasPrefix(sc.Text(), "MemAvailable:") {
			fs := strings.Fields(sc.Text())
			if len(fs) >= 2 {
				kb, _ := strconv.ParseFloat(fs[1], 64)
				if avail := kb / (1 << 20); avail < gb {
					t.Skipf("need %.0f GB host memory, %.1f GB available", gb, avail)
				}
			}
			return
		}
	}
}

func releaseModelMemory(t *testing.T) {
	t.Helper()
	t.Cleanup(func() {
		runtime.GC()
		debug.FreeOSMemory()
	})
}
