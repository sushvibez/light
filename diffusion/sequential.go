package diffusion

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"time"

	"light/storage"
	"light/tensor"
)

// releaseMemory hands freed model weights back: collections (device
// storages are freed by GC cleanups, which run on their own goroutine after a
// collection, so it waits briefly for them), the Go heap to the OS, idle
// WebGPU buffers and CUDA's cached blocks to their drivers.
func releaseMemory() {
	runtime.GC()
	runtime.GC()
	time.Sleep(100 * time.Millisecond) // let the cleanup goroutine release device buffers
	runtime.GC()
	debug.FreeOSMemory()
	tensor.TrimWebGPU()
	storage.DrainCUDAPool()
}

// transformerInChannels reads in_channels from <dir>/config.json, so the
// latent shape is known without loading the weights.
func transformerInChannels(dir string) (int, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return 0, err
	}
	var c struct {
		InChannels int `json:"in_channels"`
	}
	err = json.Unmarshal(raw, &c)
	return c.InChannels, err
}
