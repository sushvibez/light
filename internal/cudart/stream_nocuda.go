//go:build nocuda || !cgo

package cudart

import "errors"

var errNoCUDA = errors.New("cudart: built without CUDA")

func SetStream(uintptr)                                          {}
func Stream() uintptr                                            { return 0 }
func StreamCreate() (uintptr, error)                             { return 0, errNoCUDA }
func StreamDestroy(uintptr) error                                { return errNoCUDA }
func StreamSynchronize(uintptr) error                            { return errNoCUDA }
func StreamBeginCapture(uintptr) error                           { return errNoCUDA }
func StreamEndCapture(uintptr) (uintptr, error)                  { return 0, errNoCUDA }
func GraphInstantiate(uintptr) (uintptr, error)                  { return 0, errNoCUDA }
func GraphNodeCount(uintptr) (int, error)                        { return 0, errNoCUDA }
func GraphLaunch(uintptr, uintptr) error                         { return errNoCUDA }
func GraphExecDestroy(uintptr) error                             { return errNoCUDA }
func GraphDestroy(uintptr) error                                 { return errNoCUDA }
func MemcpyHtoDAsyncOn(uintptr, uintptr, uintptr, uintptr) error { return errNoCUDA }
func MallocHost(uintptr) (uintptr, error)                        { return 0, errNoCUDA }
func FreeHost(uintptr) error                                     { return errNoCUDA }
func DeviceSynchronize() error                                   { return errNoCUDA }

type GPUTime struct {
	Count  int
	Micros float64
}

func StartGPUTrace()                             {}
func StopGPUTrace() (map[string]GPUTime, error)  { return nil, errNoCUDA }
func GPUTraceTop(map[string]GPUTime, int) string { return "" }
