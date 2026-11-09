package safetensors

import (
	"fmt"
	"os"

	"light/device"
	"light/tensor"
)

// LoadFiles opens every path in `paths` in turn, calls Load on it, and
// merges the resulting tensor maps into one. Used for sharded HF checkpoints
// (`model-00001-of-NNNNN.safetensors`, …).
//
// Duplicate tensor names across shards return an error — HF index.json
// guarantees uniqueness, but defending here catches accidental misuse.
// The first file's metadata is returned; per-shard metadata is dropped.
func LoadFiles(paths []string, dev device.Device) (map[string]*tensor.Tensor, map[string]string, error) {
	if len(paths) == 0 {
		return nil, nil, fmt.Errorf("safetensors.LoadFiles: no paths")
	}
	out := make(map[string]*tensor.Tensor)
	var firstMeta map[string]string
	for i, p := range paths {
		var shard map[string]*tensor.Tensor
		var meta map[string]string
		var err error
		if dev.Kind() == device.KindCPU {
			shard, meta, err = loadMapped(p) // no heap copy of the weights
		}
		if shard == nil {
			f, err2 := os.Open(p)
			if err2 != nil {
				return nil, nil, fmt.Errorf("safetensors.LoadFiles: open %s: %w", p, err2)
			}
			shard, meta, err = Load(f, dev)
			f.Close()
		}
		if err != nil {
			return nil, nil, fmt.Errorf("safetensors.LoadFiles: load %s: %w", p, err)
		}
		if i == 0 {
			firstMeta = meta
		}
		for name, t := range shard {
			if _, dup := out[name]; dup {
				return nil, nil, fmt.Errorf("safetensors.LoadFiles: tensor %q present in multiple shards", name)
			}
			out[name] = t
		}
	}
	return out, firstMeta, nil
}

func deviceCPU() device.Device { return device.CPU() }
