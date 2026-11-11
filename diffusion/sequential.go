package diffusion

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"

	"light/tensor"
)

// releaseMemory hands freed model weights back: two collections (storage
// cleanups run after the first), the Go heap to the OS, and idle WebGPU
// buffers to the driver.
func releaseMemory() {
	runtime.GC()
	runtime.GC()
	debug.FreeOSMemory()
	tensor.TrimWebGPU()
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
