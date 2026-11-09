//go:build nowebgpu

package main

func webgpuMemMB() float64 { return 0 }

func webgpuName() string { return "" }
