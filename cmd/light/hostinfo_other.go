//go:build !windows

package main

import (
	"bufio"
	"os"
	"runtime"
	"strconv"
	"strings"
)

// procField returns the first number on the line of /proc file f that
// starts with key (kB values are converted to bytes).
func procField(f, key string) uint64 {
	fh, err := os.Open(f)
	if err != nil {
		return 0
	}
	defer fh.Close()
	sc := bufio.NewScanner(fh)
	for sc.Scan() {
		if line := sc.Text(); strings.HasPrefix(line, key) {
			fs := strings.Fields(strings.TrimPrefix(line, key))
			if len(fs) == 0 {
				return 0
			}
			n, _ := strconv.ParseUint(fs[0], 10, 64)
			if len(fs) > 1 && fs[1] == "kB" {
				n <<= 10
			}
			return n
		}
	}
	return 0
}

// cpuName is /proc/cpuinfo's model name (Linux), else the architecture.
func cpuName() string {
	if b, err := os.ReadFile("/proc/cpuinfo"); err == nil {
		for _, line := range strings.Split(string(b), "\n") {
			if strings.HasPrefix(line, "model name") {
				if i := strings.Index(line, ":"); i >= 0 {
					return strings.TrimSpace(line[i+1:])
				}
			}
		}
	}
	return runtime.GOOS + "/" + runtime.GOARCH
}

func totalRAM() uint64 { return procField("/proc/meminfo", "MemTotal:") }

func processRSS() uint64 { return procField("/proc/self/status", "VmRSS:") }
