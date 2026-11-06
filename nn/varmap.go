package nn

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"light/device"
	"light/npy"
	"light/pth"
	"light/safetensors"
	"light/tensor"
)

// VarMap is a named collection of tensors (parameters and buffers), the
// counterpart of Candle's `VarMap`: it gives a model a flat `name → tensor`
// view that can be saved to safetensors / npz and reloaded in place from
// safetensors, npz or PyTorch .pth files. Loading copies data into the
// existing tensors (shape-checked, device- and dtype-converted as needed),
// so modules holding those tensors see the new values without rewiring.
type VarMap struct {
	vars  map[string]*tensor.Tensor
	order []string
}

// NewVarMap returns an empty VarMap.
func NewVarMap() *VarMap { return &VarMap{vars: make(map[string]*tensor.Tensor)} }

// Set registers (or replaces) a tensor under name.
func (m *VarMap) Set(name string, t *tensor.Tensor) {
	if _, ok := m.vars[name]; !ok {
		m.order = append(m.order, name)
	}
	m.vars[name] = t
}

// SetAll registers every entry of a map, in sorted name order.
func (m *VarMap) SetAll(ts map[string]*tensor.Tensor) {
	names := make([]string, 0, len(ts))
	for n := range ts {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		m.Set(n, ts[n])
	}
}

// Get returns the tensor registered under name.
func (m *VarMap) Get(name string) (*tensor.Tensor, bool) {
	t, ok := m.vars[name]
	return t, ok
}

// Names returns the registered names in insertion order.
func (m *VarMap) Names() []string { return append([]string(nil), m.order...) }

// Len is the number of entries.
func (m *VarMap) Len() int { return len(m.vars) }

// Tensors returns the entries as a plain map (shared tensors).
func (m *VarMap) Tensors() map[string]*tensor.Tensor {
	out := make(map[string]*tensor.Tensor, len(m.vars))
	for k, v := range m.vars {
		out[k] = v
	}
	return out
}

// Parameters returns all tensors in insertion order (Module-style).
func (m *VarMap) Parameters() []*tensor.Tensor {
	out := make([]*tensor.Tensor, 0, len(m.order))
	for _, n := range m.order {
		out = append(out, m.vars[n])
	}
	return out
}

// Prefixed returns a VarMap view of the entries under `prefix.` with the
// prefix stripped (tensors are shared, not copied).
func (m *VarMap) Prefixed(prefix string) *VarMap {
	p := prefix + "."
	out := NewVarMap()
	for _, n := range m.order {
		if strings.HasPrefix(n, p) {
			out.Set(strings.TrimPrefix(n, p), m.vars[n])
		}
	}
	return out
}

// cpuContiguous returns a CPU, contiguous copy of every entry for writing.
func (m *VarMap) cpuContiguous() (map[string]*tensor.Tensor, error) {
	out := make(map[string]*tensor.Tensor, len(m.vars))
	for n, t := range m.vars {
		c := t
		var err error
		if c.Device().Kind() != device.KindCPU {
			if c, err = c.To(device.CPU()); err != nil {
				return nil, fmt.Errorf("VarMap: %s: %w", n, err)
			}
		}
		if !c.IsContiguous() || c.Offset() != 0 {
			if c, err = c.Contiguous(); err != nil {
				return nil, fmt.Errorf("VarMap: %s: %w", n, err)
			}
		}
		out[n] = c
	}
	return out, nil
}

// Save writes every entry to a safetensors file (`.safetensors`) or an npz
// archive (`.npz`), chosen by extension.
func (m *VarMap) Save(path string) error {
	ts, err := m.cpuContiguous()
	if err != nil {
		return err
	}
	switch {
	case strings.HasSuffix(path, ".npz"):
		return npy.WriteNpz(path, ts)
	default:
		f, err := os.Create(path)
		if err != nil {
			return err
		}
		if err := safetensors.Write(f, ts, map[string]string{"format": "light"}); err != nil {
			f.Close()
			return err
		}
		return f.Close()
	}
}

// Load reads a safetensors, npz or PyTorch .pth/.pt/.bin checkpoint and
// copies each entry into the registered tensor of the same name. Every
// registered name must be present in the file with a matching shape;
// extra tensors in the file are ignored. For .pth files, key selects the
// state dict inside a checkpoint dict ("" = top level).
func (m *VarMap) Load(path, key string) error {
	src, err := ReadCheckpoint(path, key, device.CPU())
	if err != nil {
		return err
	}
	for _, n := range m.order {
		s, ok := src[n]
		if !ok {
			return fmt.Errorf("VarMap.Load: %s: tensor %q not in file", path, n)
		}
		if err := m.vars[n].CopyFrom(s); err != nil {
			return fmt.Errorf("VarMap.Load: %s: %w", n, err)
		}
	}
	return nil
}

// ReadCheckpoint loads every tensor of a safetensors (single file), npz or
// PyTorch zip checkpoint onto dev, dispatching on the file extension
// (`.safetensors`, `.npz`, `.npy` (single tensor named "data"), anything
// else is treated as a torch.save file). key is the .pth state-dict key.
func ReadCheckpoint(path, key string, dev device.Device) (map[string]*tensor.Tensor, error) {
	switch {
	case strings.HasSuffix(path, ".safetensors"):
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		ts, _, err := safetensors.Load(f, dev)
		return ts, err
	case strings.HasSuffix(path, ".npz"):
		return npy.ReadNpz(path, dev)
	case strings.HasSuffix(path, ".npy"):
		t, err := npy.ReadFile(path, dev)
		if err != nil {
			return nil, err
		}
		return map[string]*tensor.Tensor{"data": t}, nil
	default:
		return pth.Load(path, key, dev)
	}
}
