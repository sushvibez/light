//go:build unix

package mmap

import (
	"os"
	"syscall"
)

func mapFile(f *os.File, size int64) ([]byte, func() error, error) {
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, os.NewSyscallError("mmap", err)
	}
	return data, func() error { return syscall.Munmap(data) }, nil
}
