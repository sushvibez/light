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
	if dt != dtype.BF16 && dt != dtype.F16 {
		return 0, fmt.Errorf("StoreWeights16: want BF16 or F16, got %v", dt)
	}
	// Pass 1: every slot holding a narrowable weight, grouped by tensor.
	c := &w16Collect{seen: map[seenKey]bool{}, slots: map[*tensor.Tensor][]w16Slot{}}
	c.walk(reflect.ValueOf(model))
	webgpu := false
	for _, t := range c.order {
		if device.IsWebGPU(t.Device()) {
			webgpu = true
			break
		}
	}
	// Pass 2: narrow each tensor, repoint its slots, forget the F32 copy.
	n := 0
	var freed int64
	for i, t := range c.order {
		nt, err := ops.Cast(t, dt)
		if err != nil {
			return n, fmt.Errorf("StoreWeights16: %w", err)
		}
		for _, s := range c.slots[t] {
			*s.p = nt
			if s.emb != nil {
				s.emb.WidenF32 = true
			}
		}
		freed += int64(dtype.F32.ByteSize(t.NumElements()))
		delete(c.slots, t)
		c.order[i] = nil
		n++
		if freed >= 512<<20 { // let the replaced F32 buffers go as we convert
			freed = 0
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

// w16Slot is a field holding a weight; emb is set for an embedding table
// (which then widens its gathered rows).
type w16Slot struct {
	p   **tensor.Tensor
	emb *nn.Embedding
}

type w16Collect struct {
	seen  map[seenKey]bool
	slots map[*tensor.Tensor][]w16Slot
	order []*tensor.Tensor
}

var (
	linearType    = reflect.TypeOf((*nn.Linear)(nil))
	embeddingType = reflect.TypeOf((*nn.Embedding)(nil))
	tensorType    = reflect.TypeOf((*tensor.Tensor)(nil))
)

// add records slot p if it holds an F32 matrix with even rows.
func (c *w16Collect) add(p **tensor.Tensor, emb *nn.Embedding) {
	t := *p
	if t == nil || t.DType() != dtype.F32 || t.Rank() != 2 || t.Dim(1)%2 != 0 {
		return
	}
	if _, ok := c.slots[t]; !ok {
		c.order = append(c.order, t)
	}
	c.slots[t] = append(c.slots[t], w16Slot{p, emb})
}

func (c *w16Collect) walk(v reflect.Value) {
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() || v.Type() == tensorType {
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
				c.add(&l.W, nil)
			}
			return
		case embeddingType:
			e := (*nn.Embedding)(v.UnsafePointer())
			c.add(&e.Weight, e)
			return
		}
		c.walk(v.Elem())
	case reflect.Interface:
		if !v.IsNil() {
			c.walk(v.Elem())
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			f := v.Field(i)
			if !f.CanInterface() && f.CanAddr() {
				f = reflect.NewAt(f.Type(), unsafe.Pointer(f.UnsafeAddr())).Elem()
			}
			c.walk(f)
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
