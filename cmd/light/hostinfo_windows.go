//go:build windows

package main

import (
	"syscall"
	"unsafe"
)

var (
	kernel32         = syscall.NewLazyDLL("kernel32.dll")
	procMemStatus    = kernel32.NewProc("GlobalMemoryStatusEx")
	procProcMemInfo  = kernel32.NewProc("K32GetProcessMemoryInfo")
	cpuRegistryKey   = `HARDWARE\DESCRIPTION\System\CentralProcessor\0`
	cpuRegistryValue = "ProcessorNameString"
)

// cpuName reads the processor's marketing name from the registry.
func cpuName() string {
	var k syscall.Handle
	path, _ := syscall.UTF16PtrFromString(cpuRegistryKey)
	if syscall.RegOpenKeyEx(syscall.HKEY_LOCAL_MACHINE, path, 0, syscall.KEY_READ, &k) != nil {
		return ""
	}
	defer syscall.RegCloseKey(k)
	name, _ := syscall.UTF16PtrFromString(cpuRegistryValue)
	buf := make([]uint16, 256)
	n := uint32(len(buf) * 2)
	var typ uint32
	if syscall.RegQueryValueEx(k, name, nil, &typ, (*byte)(unsafe.Pointer(&buf[0])), &n) != nil {
		return ""
	}
	return syscall.UTF16ToString(buf)
}

// totalRAM is the physical memory in bytes (GlobalMemoryStatusEx).
func totalRAM() uint64 {
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
	return st.totalPhys
}

// processRSS is this process's working set in bytes.
func processRSS() uint64 {
	var c struct {
		cb, pageFaults                   uint32
		peakWorkingSet, workingSet       uintptr
		quotaPeakPaged, quotaPaged       uintptr
		quotaPeakNonPaged, quotaNonPaged uintptr
		pagefileUsage, peakPagefileUsage uintptr
	}
	c.cb = uint32(unsafe.Sizeof(c))
	h, _ := syscall.GetCurrentProcess()
	if r, _, _ := procProcMemInfo.Call(uintptr(h), uintptr(unsafe.Pointer(&c)), uintptr(c.cb)); r == 0 {
		return 0
	}
	return uint64(c.workingSet)
}
