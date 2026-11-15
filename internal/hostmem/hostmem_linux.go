//go:build linux

package hostmem

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

func availableGB() float64 {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if strings.HasPrefix(sc.Text(), "MemAvailable:") {
			if fs := strings.Fields(sc.Text()); len(fs) >= 2 {
				kb, _ := strconv.ParseFloat(fs[1], 64)
				return kb / (1 << 20)
			}
		}
	}
	return 0
}
