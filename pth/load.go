// Package pth loads PyTorch checkpoints (`torch.save` zip format, PyTorch
// ≥ 1.6): a `data.pkl` pickle describing the object graph plus one
// `data/<key>` member per storage. Tensors are reconstructed from
// `torch._utils._rebuild_tensor_v2` (and `_rebuild_parameter`,
// `torch._tensor._rebuild_from_type_v2`) records; strided or offset views
// are gathered into contiguous row-major tensors on load.
//
// Accepted top-level objects: a state dict, a checkpoint dict holding the
// state dict under a key (pass that key), or a pickled nn.Module whose
// `_parameters` / `_buffers` / `_modules` tree is flattened into dotted
// names. Counterpart of Candle's `candle_core::pickle`.
package pth

import (
	"archive/zip"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// TensorInfo describes one tensor without loading its data.
type TensorInfo struct {
	Name    string
	DType   dtype.DType
	Shape   []int
	Strides []int // in elements, as stored
	Offset  int   // storage offset in elements

	storage     string // zip member holding the storage bytes
	storageSize int    // in elements
}

// File is an opened checkpoint; tensors are read lazily by name.
type File struct {
	zr    *zip.Reader
	close io.Closer
	files map[string]*zip.File
	infos map[string]TensorInfo
}

// Open parses the checkpoint's pickle(s). key selects an entry of the
// top-level dict (e.g. "state_dict" or "model"); "" uses the top level.
func Open(path, key string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	zr, err := zip.NewReader(f, st.Size())
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("pth: %s is not a zip-format checkpoint (legacy torch.save files are not supported): %w", path, err)
	}
	pf := &File{zr: zr, close: f, files: make(map[string]*zip.File, len(zr.File)), infos: make(map[string]TensorInfo)}
	for _, zf := range zr.File {
		pf.files[zf.Name] = zf
	}
	found := false
	for _, zf := range zr.File {
		if !strings.HasSuffix(zf.Name, "data.pkl") {
			continue
		}
		found = true
		rc, err := zf.Open()
		if err != nil {
			f.Close()
			return nil, err
		}
		obj, err := unpickle(rc)
		rc.Close()
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("pth: %s: %w", zf.Name, err)
		}
		dir := strings.TrimSuffix(zf.Name, ".pkl") + "/"
		if key != "" {
			d, ok := obj.(*dict)
			if !ok {
				f.Close()
				return nil, fmt.Errorf("pth: top-level object is %T, cannot index key %q", obj, key)
			}
			v, ok := d.get(key)
			if !ok {
				f.Close()
				return nil, fmt.Errorf("pth: key %q not found (have %v)", key, d.stringKeys())
			}
			obj = v
		}
		if err := pf.collect(obj, "", dir); err != nil {
			f.Close()
			return nil, err
		}
	}
	if !found {
		f.Close()
		return nil, fmt.Errorf("pth: %s has no data.pkl member", path)
	}
	return pf, nil
}

func (d *dict) stringKeys() []string {
	var ks []string
	for _, k := range d.keys {
		if s, ok := k.(string); ok {
			ks = append(ks, s)
		}
	}
	return ks
}

// collect walks obj and records every tensor reachable through dict entries
// (state dicts) or nn.Module state (BUILD nodes with _parameters/_buffers/
// _modules). Non-tensor leaves are ignored.
func (pf *File) collect(obj any, prefix, dir string) error {
	switch o := obj.(type) {
	case *dict:
		for i, k := range o.keys {
			ks, ok := k.(string)
			if !ok {
				continue
			}
			name := prefix + ks
			if info, ok, err := tensorInfo(o.vals[i], dir); err != nil {
				return fmt.Errorf("pth: %s: %w", name, err)
			} else if ok {
				info.Name = name
				pf.infos[name] = info
			} else if d, isDict := o.vals[i].(*dict); isDict && prefix == "" && pf.looksLikeState(d) {
				// Nested state dicts (e.g. {"model": {...}}) without a key:
				// flatten with a dotted prefix.
				if err := pf.collect(d, name+".", dir); err != nil {
					return err
				}
			} else if b, isBuild := o.vals[i].(*build); isBuild {
				if err := pf.collectModule(b, name+".", dir); err != nil {
					return err
				}
			}
		}
	case *build:
		return pf.collectModule(o, prefix, dir)
	}
	return nil
}

func (pf *File) looksLikeState(d *dict) bool {
	for _, v := range d.vals {
		if _, ok, _ := tensorInfo(v, ""); ok {
			return true
		}
	}
	return false
}

// collectModule flattens a pickled nn.Module (BUILD(NEWOBJ(cls), state)).
func (pf *File) collectModule(b *build, prefix, dir string) error {
	state, ok := b.args.(*dict)
	if !ok {
		return nil
	}
	for _, bucket := range []string{"_parameters", "_buffers"} {
		if d, ok := state.get(bucket); ok {
			if dd, ok := d.(*dict); ok {
				if err := pf.collect(dd, prefix, dir); err != nil {
					return err
				}
			}
		}
	}
	if mods, ok := state.get("_modules"); ok {
		if md, ok := mods.(*dict); ok {
			for i, k := range md.keys {
				ks, _ := k.(string)
				if sub, ok := md.vals[i].(*build); ok {
					if err := pf.collectModule(sub, prefix+ks+".", dir); err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}

// tensorInfo decodes a `_rebuild_tensor_v2`-style reduce node.
func tensorInfo(obj any, dir string) (TensorInfo, bool, error) {
	r, ok := obj.(*reduce)
	if !ok {
		return TensorInfo{}, false, nil
	}
	c, ok := r.callable.(class)
	if !ok {
		return TensorInfo{}, false, nil
	}
	args, _ := r.args.(*tuple)
	switch {
	case c.module == "torch._tensor" && c.name == "_rebuild_from_type_v2":
		// (func, type, args, state)
		if args == nil || len(args.items) < 3 {
			return TensorInfo{}, false, fmt.Errorf("_rebuild_from_type_v2: bad args")
		}
		return tensorInfo(&reduce{args.items[0], args.items[2]}, dir)
	case c.module == "torch._utils" && c.name == "_rebuild_parameter":
		// (data, requires_grad, backward_hooks)
		if args == nil || len(args.items) < 1 {
			return TensorInfo{}, false, fmt.Errorf("_rebuild_parameter: bad args")
		}
		return tensorInfo(args.items[0], dir)
	case c.module == "torch._utils" && c.name == "_rebuild_tensor_v2":
	default:
		return TensorInfo{}, false, nil
	}
	// (storage, storage_offset, size, stride, requires_grad, backward_hooks[, metadata])
	if args == nil || len(args.items) < 4 {
		return TensorInfo{}, false, fmt.Errorf("_rebuild_tensor_v2: bad args")
	}
	pid, ok := args.items[0].(persID)
	if !ok {
		return TensorInfo{}, false, fmt.Errorf("_rebuild_tensor_v2: storage is %T", args.items[0])
	}
	// ('storage', storage_type, key, location, numel)
	st, ok := pid.id.(*tuple)
	if !ok || len(st.items) < 5 {
		return TensorInfo{}, false, fmt.Errorf("unexpected persistent id %v", pid.id)
	}
	stype, _ := st.items[1].(class)
	key, _ := st.items[2].(string)
	numel, _ := st.items[4].(int64)
	dt, err := storageDType(stype.name)
	if err != nil {
		return TensorInfo{}, false, err
	}
	info := TensorInfo{DType: dt, storage: dir + key, storageSize: int(numel)}
	off, _ := args.items[1].(int64)
	info.Offset = int(off)
	if info.Shape, err = ints(args.items[2]); err != nil {
		return TensorInfo{}, false, fmt.Errorf("size: %w", err)
	}
	if info.Strides, err = ints(args.items[3]); err != nil {
		return TensorInfo{}, false, fmt.Errorf("stride: %w", err)
	}
	return info, true, nil
}

func ints(o any) ([]int, error) {
	t, ok := o.(*tuple)
	if !ok {
		if l, ok := o.(*list); ok {
			t = &tuple{l.items}
		} else {
			return nil, fmt.Errorf("expected tuple, got %T", o)
		}
	}
	out := make([]int, len(t.items))
	for i, v := range t.items {
		n, ok := v.(int64)
		if !ok {
			return nil, fmt.Errorf("expected int, got %T", v)
		}
		out[i] = int(n)
	}
	return out, nil
}

func storageDType(name string) (dtype.DType, error) {
	switch name {
	case "FloatStorage":
		return dtype.F32, nil
	case "DoubleStorage":
		return dtype.F64, nil
	case "HalfStorage":
		return dtype.F16, nil
	case "BFloat16Storage":
		return dtype.BF16, nil
	case "ByteStorage", "BoolStorage":
		return dtype.U8, nil
	case "CharStorage":
		return dtype.I8, nil
	case "ShortStorage":
		return dtype.I16, nil
	case "IntStorage":
		return dtype.I32, nil
	case "LongStorage":
		return dtype.I64, nil
	}
	return 0, fmt.Errorf("unsupported storage type %q", name)
}

// Close releases the underlying file.
func (pf *File) Close() error { return pf.close.Close() }

// Infos returns the tensor descriptors keyed by name.
func (pf *File) Infos() map[string]TensorInfo { return pf.infos }

// Names lists the tensors in sorted order.
func (pf *File) Names() []string {
	names := make([]string, 0, len(pf.infos))
	for k := range pf.infos {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// Get loads one tensor onto dev as a contiguous row-major tensor.
func (pf *File) Get(name string, dev device.Device) (*tensor.Tensor, error) {
	info, ok := pf.infos[name]
	if !ok {
		return nil, fmt.Errorf("pth: no tensor %q", name)
	}
	zf, ok := pf.files[info.storage]
	if !ok {
		return nil, fmt.Errorf("pth: %s: storage member %q missing", name, info.storage)
	}
	rc, err := zf.Open()
	if err != nil {
		return nil, err
	}
	raw, err := io.ReadAll(rc)
	rc.Close()
	if err != nil {
		return nil, fmt.Errorf("pth: %s: read storage: %w", name, err)
	}
	esz := info.DType.Size()
	n := 1
	for _, d := range info.Shape {
		n *= d
	}
	data, err := gather(raw, esz, info.Shape, info.Strides, info.Offset)
	if err != nil {
		return nil, fmt.Errorf("pth: %s: %w", name, err)
	}
	if len(data) != n*esz {
		return nil, fmt.Errorf("pth: %s: gathered %d bytes, want %d", name, len(data), n*esz)
	}
	return tensor.FromBytes(data, info.DType, info.Shape, dev)
}

// gather copies a strided view out of storage bytes into row-major order.
func gather(raw []byte, esz int, shape, strides []int, offset int) ([]byte, error) {
	rank := len(shape)
	if len(strides) != rank {
		return nil, fmt.Errorf("stride rank %d vs shape rank %d", len(strides), rank)
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	// Fast path: already contiguous.
	contiguous := true
	exp := 1
	for i := rank - 1; i >= 0; i-- {
		if shape[i] != 1 && strides[i] != exp {
			contiguous = false
		}
		exp *= shape[i]
	}
	start := offset * esz
	if contiguous {
		end := start + n*esz
		if end > len(raw) {
			return nil, fmt.Errorf("storage has %d bytes, view needs %d", len(raw), end)
		}
		return raw[start:end], nil
	}
	dst := make([]byte, n*esz)
	idx := make([]int, rank)
	for e := 0; e < n; e++ {
		off := offset
		for i := 0; i < rank; i++ {
			off += idx[i] * strides[i]
		}
		if (off+1)*esz > len(raw) || off < 0 {
			return nil, fmt.Errorf("view element %d at storage offset %d is out of range", e, off)
		}
		copy(dst[e*esz:(e+1)*esz], raw[off*esz:(off+1)*esz])
		for i := rank - 1; i >= 0; i-- {
			idx[i]++
			if idx[i] < shape[i] {
				break
			}
			idx[i] = 0
		}
	}
	return dst, nil
}

// Load reads every tensor of a checkpoint onto dev. key is as in Open.
func Load(path, key string, dev device.Device) (map[string]*tensor.Tensor, error) {
	pf, err := Open(path, key)
	if err != nil {
		return nil, err
	}
	defer pf.Close()
	out := make(map[string]*tensor.Tensor, len(pf.infos))
	for _, name := range pf.Names() {
		t, err := pf.Get(name, dev)
		if err != nil {
			return nil, err
		}
		out[name] = t
	}
	return out, nil
}
