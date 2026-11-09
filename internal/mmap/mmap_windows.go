//go:build windows

package mmap

import (
	"os"
	"syscall"
	"unsafe"
)

func mapFile(f *os.File, size int64) ([]byte, func() error, error) {
	h, err := syscall.CreateFileMapping(syscall.Handle(f.Fd()), nil, syscall.PAGE_READONLY, uint32(size>>32), uint32(size), nil)
	if err != nil {
		return nil, nil, os.NewSyscallError("CreateFileMapping", err)
	}
	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_READ, 0, 0, uintptr(size))
	if err != nil {
		syscall.CloseHandle(h)
		return nil, nil, os.NewSyscallError("MapViewOfFile", err)
	}
	p := *(*unsafe.Pointer)(unsafe.Pointer(&addr))
	data := unsafe.Slice((*byte)(p), int(size))
	return data, func() error {
		err := syscall.UnmapViewOfFile(addr)
		if cerr := syscall.CloseHandle(h); err == nil {
			err = cerr
		}
		return err
	}, nil
}
