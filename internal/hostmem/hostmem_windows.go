//go:build windows

package hostmem

import (
	"syscall"
	"unsafe"
)

var procMemStatus = syscall.NewLazyDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")

func availableGB() float64 {
	var st struct {
		length, load                    uint32
		totalPhys, availPhys            uint64
		totalPage, availPage            uint64
		totalVirtual, availVirtual, ext uint64
	}
	st.length = uint32(unsafe.Sizeof(st))
	if r, _, _ := procMemStatus.Call(uintptr(unsafe.Pointer(&st))); r == 0 {
		return 0
	}
	return float64(st.availPhys) / (1 << 30)
}
