package safetensors

import (
	"bytes"
	"fmt"
	"sync/atomic"

	"light/internal/mmap"
	"light/storage"
	"light/tensor"
)

// loadMapped loads a safetensors file as CPU tensors over a read-only
// mapping of it (the tensors must not be written in place): no heap copy
// of the weights, and the pages stay file-backed (the OS may drop them under memory pressure) until a loader
// copies them where they belong. The mapping is closed once every tensor
// on it has been collected. A tensor whose bytes are misaligned for its
// element size is copied instead.
func loadMapped(path string) (map[string]*tensor.Tensor, map[string]string, error) {
	m, err := mmap.Open(path)
	if err != nil {
		return nil, nil, err
	}
	// One reference per tensor plus the loader's own, dropped on return.
	var refs atomic.Int64
	refs.Store(1)
	drop := func() {
		if refs.Add(-1) == 0 {
			m.Close()
		}
	}
	defer drop()
	data := m.Data
	infos, metadata, dataStart, dataLen, err := readHeader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, nil, err
	}
	out := make(map[string]*tensor.Tensor, len(infos))
	for name, info := range infos {
		// checkEntry bounds the range inside the data section, so the
		// slice below can neither panic nor reach back into the header.
		dt, n, err := checkEntry(name, info, dataLen)
		if err != nil {
			return nil, nil, err
		}
		lo, hi := dataStart+int64(info.DataOffsets[0]), dataStart+int64(info.DataOffsets[1])
		buf := data[lo:hi:hi]
		if hi == lo {
			t, err := tensor.FromBytes(buf, dt, info.Shape, deviceCPU())
			if err != nil {
				return nil, nil, fmt.Errorf("safetensors: build %q: %w", name, err)
			}
			out[name] = t
			continue
		}
		wrap := storage.WrapCPUOwned
		if align := dt.ByteSize(1); align > 1 && lo%int64(align) != 0 {
			// Misaligned for its dtype (mlx writes an unpadded header):
			// copied to an aligned buffer on first use, or slice by slice.
			wrap = storage.WrapCPUUnaligned
		}
		refs.Add(1)
		t, err := tensor.FromStorage(wrap(dt, n, buf, drop), info.Shape)
		if err != nil {
			drop()
			return nil, nil, fmt.Errorf("safetensors: build %q: %w", name, err)
		}
		out[name] = t
	}
	return out, metadata, nil
}
