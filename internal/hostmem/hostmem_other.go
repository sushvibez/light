//go:build !linux && !windows

package hostmem

func availableGB() float64 { return 0 }
