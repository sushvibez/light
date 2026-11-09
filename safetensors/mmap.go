package safetensors

import (
	"bytes"
	"encoding/binary"
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
	infos, metadata, err := Header(bytes.NewReader(data))
	if err != nil {
		return nil, nil, err
	}
	dataStart := int64(8 + binary.LittleEndian.Uint64(data[:8]))
	out := make(map[string]*tensor.Tensor, len(infos))
	for name, info := range infos {
		dt, err := parseDType(info.DType)
		if err != nil {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		n := product(info.Shape)
		lo, hi := dataStart+int64(info.DataOffsets[0]), dataStart+int64(info.DataOffsets[1])
		if hi < lo || hi > int64(len(data)) || int(hi-lo) != dt.ByteSize(n) {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: bad byte range [%d, %d) for %v %v", name, lo, hi, dt, info.Shape)
		}
		buf := data[lo:hi:hi]
		if align := dt.ByteSize(1); align > 1 && lo%int64(align) != 0 || hi == lo {
			t, err := tensor.FromBytes(buf, dt, info.Shape, deviceCPU())
			if err != nil {
				return nil, nil, fmt.Errorf("safetensors: build %q: %w", name, err)
			}
			out[name] = t
			continue
		}
		refs.Add(1)
		t, err := tensor.FromStorage(storage.WrapCPUOwned(dt, n, buf, drop), info.Shape)
		if err != nil {
			drop()
			return nil, nil, fmt.Errorf("safetensors: build %q: %w", name, err)
		}
		out[name] = t
	}
	return out, metadata, nil
}
