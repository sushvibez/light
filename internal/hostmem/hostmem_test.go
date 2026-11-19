package hostmem

import (
	"runtime"
	"testing"
)

func TestAvailableGB(t *testing.T) {
	gb := AvailableGB()
	t.Logf("%s: %.1f GB available", runtime.GOOS, gb)
	if (runtime.GOOS == "linux" || runtime.GOOS == "windows") && !(gb > 0 && gb < 1<<20) {
		t.Errorf("available memory %.1f GB on %s", gb, runtime.GOOS)
	}
}
