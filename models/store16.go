package models

import (
	"fmt"
	"reflect"
	"runtime"
	"runtime/debug"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// StoreWeights16 narrows an F32 model's matmul weights to 16-bit storage
// (BF16 or F16) in place: every reachable nn.Linear weight and nn.Embedding
// table, found by walking the model's fields (exported or not). Activations,
// norms, biases and every other parameter stay F32; nn.Linear then runs
// ops.MatMulW16 (the weight widened exactly inside the matmul) and the
// embeddings widen their gathered rows.
//
// For a checkpoint stored in that dtype the narrowing is lossless, so the
// model computes exactly what the F32 load does in half the weight memory
// (WebGPU, which has no 16-bit arithmetic, keeps 3–5B models in memory this
// way). Tied weights stay shared. Tables whose rows are not a whole number
// of 32-bit words (an odd width) are left F32. Each F32 weight is dropped as
// soon as all its users point at the narrowed copy, so the pass needs little
// memory beyond the model itself. Returns the number of tensors narrowed.
func StoreWeights16(model any, dt dtype.DType) (int, error) {
	return convert16(model, dt, nil)
}

// MoveToDevice16 moves a model loaded on the CPU at F32 to dev with its
// matmul weights narrowed to 16-bit storage on the way, as StoreWeights16
// does, so the F32 weights never reach the device. Every other tensor the
// model holds moves as is. An embedding table (not tied to a Linear) larger
// than one device buffer stays in host memory: its lookups gather on the
// host and upload only the rows (Gemma 4's per-layer embeddings, 4.5 GB in
// BF16, against a 2 GB WebGPU buffer cap). Device-keyed caches the model
// built on the CPU are rebuilt on first use. Returns the number of tensors
// narrowed.
func MoveToDevice16(model any, dev device.Device, dt dtype.DType) (int, error) {
	return convert16(model, dt, dev)
}

func convert16(model any, dt dtype.DType, dev device.Device) (int, error) {
	if dt != dtype.BF16 && dt != dtype.F16 {
		return 0, fmt.Errorf("StoreWeights16: want BF16 or F16, got %v", dt)
	}
	c := &w16Collect{
		seen:  map[seenKey]bool{},
		slots: map[*tensor.Tensor][]w16Slot{},
		dup:   map[**tensor.Tensor]bool{},
		all:   dev != nil,
	}
	c.walk(reflect.ValueOf(model))
	webgpu := dev != nil && device.IsWebGPU(dev)
	for _, t := range c.order {
		if device.IsWebGPU(t.Device()) {
			webgpu = true
			break
		}
	}
	var maxBytes int64
	if dev != nil && device.IsWebGPU(dev) {
		maxBytes = tensor.WebGPUMaxTensorBytes()
	}
	n := 0
	var done int64
	for i, t := range c.order {
		slots := c.slots[t]
		linear, emb := false, false
		for _, s := range slots {
			linear = linear || s.role == roleLinear
			emb = emb || s.role == roleEmbedding
		}
		nt := t
		narrowed := false
		if (linear || emb) && t.DType() == dtype.F32 && t.Rank() == 2 && t.Dim(1)%2 == 0 {
			var err error
			if nt, err = ops.Cast(t, dt); err != nil {
				return n, fmt.Errorf("StoreWeights16: %w", err)
			}
			narrowed = true
			n++
		}
		if dev != nil && !nt.Device().SameAs(dev) {
			bytes := int64(nt.DType().ByteSize(nt.NumElements()))
			hostTable := emb && !linear && maxBytes > 0 && bytes > maxBytes
			if !hostTable {
				var err error
				if nt, err = nt.To(dev); err != nil {
					return n, fmt.Errorf("StoreWeights16: %v %v: %w", t.Shape(), t.DType(), err)
				}
			}
		}
		for _, s := range slots {
			s.set(nt)
			if s.emb != nil && narrowed {
				s.emb.WidenF32 = true
			}
		}
		done += int64(t.DType().ByteSize(t.NumElements()))
		delete(c.slots, t)
		c.order[i] = nil
		if done >= 512<<20 { // let the replaced buffers go as we convert
			done = 0
			trim16(webgpu)
		}
	}
	trim16(webgpu)
	return n, nil
}

func trim16(webgpu bool) {
	if webgpu {
		tensor.TrimWebGPU() // finish the queued casts so their inputs can go
	}
	runtime.GC()
	runtime.GC() // the cleanups queued by the first cycle return the buffers
	if webgpu {
		tensor.TrimWebGPU() // release the returned buffers to the driver
	}
	debug.FreeOSMemory()
}

// seenKey identifies a visited pointer; the type matters because a struct
// and its first field share an address.
type seenKey struct {
	t reflect.Type
	p uintptr
}

const (
	roleOther = iota
	roleLinear
	roleEmbedding
)

// w16Slot is a place holding a tensor: a field / element (p) or a map entry
// (m, key). emb is set for an embedding table (which then widens its
// gathered rows).
type w16Slot struct {
	p    **tensor.Tensor
	m    reflect.Value
	key  reflect.Value
	role int
	emb  *nn.Embedding
}

func (s w16Slot) set(t *tensor.Tensor) {
	if s.p != nil {
		*s.p = t
		return
	}
	s.m.SetMapIndex(s.key, reflect.ValueOf(t))
}

type w16Collect struct {
	seen  map[seenKey]bool
	slots map[*tensor.Tensor][]w16Slot
	order []*tensor.Tensor
	dup   map[**tensor.Tensor]bool
	all   bool // every tensor (a device move), not just matmul weights
}

var (
	linearType    = reflect.TypeOf((*nn.Linear)(nil))
	embeddingType = reflect.TypeOf((*nn.Embedding)(nil))
	tensorType    = reflect.TypeOf((*tensor.Tensor)(nil))
)

func (c *w16Collect) add(t *tensor.Tensor, s w16Slot) {
	if t == nil {
		return
	}
	if s.p != nil {
		if c.dup[s.p] {
			return
		}
		c.dup[s.p] = true
	}
	if s.role == roleOther && !c.all {
		return
	}
	if _, ok := c.slots[t]; !ok {
		c.order = append(c.order, t)
	}
	c.slots[t] = append(c.slots[t], s)
}

// field returns v made usable even when it is an unexported field.
func field(v reflect.Value) reflect.Value {
	if !v.CanInterface() && v.CanAddr() {
		return reflect.NewAt(v.Type(), unsafe.Pointer(v.UnsafeAddr())).Elem()
	}
	return v
}

func (c *w16Collect) walk(v reflect.Value) {
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			return
		}
		if v.Type() == tensorType {
			if v.CanAddr() {
				p := (**tensor.Tensor)(unsafe.Pointer(v.UnsafeAddr()))
				c.add(*p, w16Slot{p: p})
			}
			return
		}
		key := seenKey{v.Type(), v.Pointer()}
		if c.seen[key] {
			return
		}
		c.seen[key] = true
		switch v.Type() {
		case linearType:
			l := (*nn.Linear)(v.UnsafePointer())
			if l.W4 == nil && l.WBlockScale == nil {
				c.add(l.W, w16Slot{p: &l.W, role: roleLinear})
			}
		case embeddingType:
			e := (*nn.Embedding)(v.UnsafePointer())
			c.add(e.Weight, w16Slot{p: &e.Weight, role: roleEmbedding, emb: e})
		}
		if c.all || (v.Type() != linearType && v.Type() != embeddingType) {
			c.walk(v.Elem())
		}
	case reflect.Interface:
		if !v.IsNil() {
			c.walk(v.Elem())
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			c.walk(field(v.Field(i)))
		}
	case reflect.Slice, reflect.Array:
		if v.Kind() == reflect.Slice && v.IsNil() {
			return
		}
		switch v.Type().Elem().Kind() {
		case reflect.Pointer, reflect.Interface, reflect.Struct, reflect.Slice, reflect.Array, reflect.Map:
		default:
			return // scalar data
		}
		for i := 0; i < v.Len(); i++ {
			c.walk(v.Index(i))
		}
	case reflect.Map:
		if v.IsNil() {
			return
		}
		if v.Type().Elem() == tensorType {
			it := v.MapRange()
			for it.Next() {
				c.add(it.Value().Interface().(*tensor.Tensor), w16Slot{m: v, key: it.Key()})
			}
			return
		}
		switch v.Type().Elem().Kind() {
		case reflect.Pointer, reflect.Interface:
		default:
			return
		}
		it := v.MapRange()
		for it.Next() {
			c.walk(it.Value())
		}
	}
}
