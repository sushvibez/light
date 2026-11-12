package ops

import (
	"fmt"
	"math"
	"unsafe"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	"light/tensor"
)

// Decode-step attention over a fixed-capacity KV cache whose length lives
// on the device. The three ops below take the position as an I64 (1,)
// tensor and issue the same launches whatever its value, which is what
// lets a decode step be captured into a CUDA graph and replayed. Where a
// device has no kernel the ops read the position back and compose the
// same math from the existing ops, so CPU and CUDA agree.

// KVAppend writes k, v (B, KH, 1, HD) at sequence position pos[0] of the
// (B, KH, cap, HD) buffers kBuf, vBuf in place.
func KVAppend(kBuf, vBuf, k, v, pos *tensor.Tensor) error {
	if k.Rank() != 4 || k.Dim(2) != 1 || !k.Shape().Equal(v.Shape()) || kBuf.Rank() != 4 || !kBuf.Shape().Equal(vBuf.Shape()) {
		return fmt.Errorf("KVAppend: want k / v (B, KH, 1, HD) and buffers (B, KH, cap, HD), got %v / %v", k.Shape(), kBuf.Shape())
	}
	if kBuf.Dim(0) != k.Dim(0) || kBuf.Dim(1) != k.Dim(1) || kBuf.Dim(3) != k.Dim(3) {
		return fmt.Errorf("KVAppend: k %v does not fit buffer %v", k.Shape(), kBuf.Shape())
	}
	if k.DType() != kBuf.DType() || v.DType() != vBuf.DType() {
		return fmt.Errorf("KVAppend: dtype mismatch (%v into %v)", k.DType(), kBuf.DType())
	}
	if err := checkPos(pos, kBuf.Device()); err != nil {
		return fmt.Errorf("KVAppend: %w", err)
	}
	if !kBuf.IsContiguous() || kBuf.Offset() != 0 || !vBuf.IsContiguous() || vBuf.Offset() != 0 {
		return fmt.Errorf("KVAppend: buffers must be contiguous")
	}
	var err error
	if k, err = ensureContig(k); err != nil {
		return err
	}
	if v, err = ensureContig(v); err != nil {
		return err
	}
	if kern, lerr := registry.Lookup(registry.Key{Op: "kv_append", Dev: kBuf.Device().Kind(), DType: k.DType()}); lerr == nil {
		return kern(&registry.Context{Inputs: []*tensor.Tensor{k, v, pos}, Output: kBuf, Attrs: map[string]any{"vbuf": vBuf}})
	}
	p, err := posValue(pos)
	if err != nil {
		return err
	}
	if p < 0 || p >= kBuf.Dim(2) {
		return fmt.Errorf("KVAppend: position %d outside capacity %d", p, kBuf.Dim(2))
	}
	dk, err := kBuf.Narrow(2, p, 1)
	if err != nil {
		return err
	}
	if err := dk.Assign(k); err != nil {
		return err
	}
	dv, err := vBuf.Narrow(2, p, 1)
	if err != nil {
		return err
	}
	return dv.Assign(v)
}

// DecodeAttention is single-query attention of q (B, NH, 1, HD) against
// the first pos[0]+1 positions of kBuf, vBuf (B, KH, cap, HD) with GQA
// group NH / KH, scores scaled by scale; the result is (B, NH, 1, HD) at
// q's dtype. window > 0 keeps only the last `window` positions (sliding
// window attention); otherwise a decode token attends to everything
// cached. Scores and the softmax are F32 on CUDA; the composed fallback
// runs the batched matmul, scale and softmax at the tensor dtype.
func DecodeAttention(q, kBuf, vBuf, pos *tensor.Tensor, scale float64, window int) (*tensor.Tensor, error) {
	if q.Rank() != 4 || q.Dim(2) != 1 || kBuf.Rank() != 4 || !kBuf.Shape().Equal(vBuf.Shape()) {
		return nil, fmt.Errorf("DecodeAttention: want q (B, NH, 1, HD) and buffers (B, KH, cap, HD), got %v / %v", q.Shape(), kBuf.Shape())
	}
	B, NH, HD := q.Dim(0), q.Dim(1), q.Dim(3)
	KH, capacity := kBuf.Dim(1), kBuf.Dim(2)
	if kBuf.Dim(0) != B || kBuf.Dim(3) != HD || KH == 0 || NH%KH != 0 {
		return nil, fmt.Errorf("DecodeAttention: q %v against buffers %v", q.Shape(), kBuf.Shape())
	}
	if q.DType() != kBuf.DType() || q.DType() != vBuf.DType() {
		return nil, fmt.Errorf("DecodeAttention: dtype mismatch q %v k %v v %v", q.DType(), kBuf.DType(), vBuf.DType())
	}
	if err := checkPos(pos, q.Device()); err != nil {
		return nil, fmt.Errorf("DecodeAttention: %w", err)
	}
	if !kBuf.IsContiguous() || kBuf.Offset() != 0 || !vBuf.IsContiguous() || vBuf.Offset() != 0 {
		return nil, fmt.Errorf("DecodeAttention: buffers must be contiguous")
	}
	var err error
	if q, err = ensureContig(q); err != nil {
		return nil, err
	}
	g := NH / KH
	if kern, lerr := registry.Lookup(registry.Key{Op: "decode_attn", Dev: q.Device().Kind(), DType: q.DType()}); lerr == nil && decodeAttnKernelOK(HD, g) {
		out, err := tensor.Empty(q.DType(), []int{B, NH, 1, HD}, q.Device())
		if err != nil {
			return nil, err
		}
		if err := kern(&registry.Context{Inputs: []*tensor.Tensor{q, kBuf, vBuf, pos}, Output: out, Attrs: map[string]any{"scale": scale, "window": window}}); err != nil {
			return nil, err
		}
		return out, nil
	}
	p, err := posValue(pos)
	if err != nil {
		return nil, err
	}
	n := p + 1
	if n < 1 || n > capacity {
		return nil, fmt.Errorf("DecodeAttention: %d valid positions of %d", n, capacity)
	}
	start := 0
	if window > 0 && n > window {
		start = n - window
	}
	// Composed: fold batch and KV heads, q (B·KH, g, HD) against the
	// narrowed buffers (B·KH, n − start, HD) read in place.
	qf, err := q.Reshape(B*KH, g, HD)
	if err != nil {
		return nil, err
	}
	kn, err := kBuf.Narrow(2, start, n-start)
	if err != nil {
		return nil, err
	}
	vn, err := vBuf.Narrow(2, start, n-start)
	if err != nil {
		return nil, err
	}
	st := kn.Strides().Slice()
	k, err := kn.AsStrided([]int{B * KH, n - start, HD}, []int{st[1], st[2], st[3]}, kn.Offset())
	if err != nil {
		return nil, err
	}
	v, err := vn.AsStrided([]int{B * KH, n - start, HD}, []int{st[1], st[2], st[3]}, vn.Offset())
	if err != nil {
		return nil, err
	}
	kT, err := k.Transpose(1, 2)
	if err != nil {
		return nil, err
	}
	if q.Device().Kind() != device.KindCUDA {
		if kT, err = kT.Contiguous(); err != nil {
			return nil, err
		}
	}
	scores, err := MatMul(qf, kT) // (B·KH, g, n)
	if err != nil {
		return nil, err
	}
	if scores, err = Affine(scores, scale, 0); err != nil {
		return nil, err
	}
	attn, err := Softmax(scores, -1)
	if err != nil {
		return nil, err
	}
	ctx, err := MatMul(attn, v) // (B·KH, g, HD)
	if err != nil {
		return nil, err
	}
	return ctx.Reshape(B, NH, 1, HD)
}

// decodeAttnKernelOK mirrors the CUDA kernel's limits (head dim a multiple
// of 32, GQA group up to 16, query rows and scores within 48 KB of shared
// memory); other shapes compose.
func decodeAttnKernelOK(hd, g int) bool {
	return hd%32 == 0 && g >= 1 && g <= 16 && g*(hd+1+32)*4 <= 48*1024
}

// CanIncrementI64 reports whether IncrementI64 runs on dev.
func CanIncrementI64(dev device.Device) bool {
	if dev.Kind() == device.KindCPU {
		return true
	}
	_, err := registry.Lookup(registry.Key{Op: "inc_i64", Dev: dev.Kind(), DType: dtype.I64})
	return err == nil
}

// IncrementI64 adds one to every element of t in place.
func IncrementI64(t *tensor.Tensor) error {
	if t.DType() != dtype.I64 || !t.IsContiguous() || t.Offset() != 0 {
		return fmt.Errorf("IncrementI64: want a contiguous I64 tensor, got %v %v", t.DType(), t.Shape())
	}
	if kern, err := registry.Lookup(registry.Key{Op: "inc_i64", Dev: t.Device().Kind(), DType: dtype.I64}); err == nil {
		return kern(&registry.Context{Output: t})
	}
	if t.Device().Kind() != device.KindCPU {
		return fmt.Errorf("IncrementI64: no kernel for %v", t.Device())
	}
	b := t.Storage().CPUBytes()
	s := unsafe.Slice((*int64)(unsafe.Pointer(unsafe.SliceData(b))), t.NumElements())
	for i := range s {
		s[i]++
	}
	return nil
}

func checkPos(pos *tensor.Tensor, dev device.Device) error {
	if pos.DType() != dtype.I64 || pos.NumElements() != 1 {
		return fmt.Errorf("position must be a one-element I64 tensor, got %v %v", pos.DType(), pos.Shape())
	}
	if !pos.Device().SameAs(dev) {
		return fmt.Errorf("position on %v, tensors on %v", pos.Device(), dev)
	}
	return nil
}

// posValue reads the position back to the host (fallback paths only).
func posValue(pos *tensor.Tensor) (int, error) {
	pos, err := pos.To(device.CPU())
	if err != nil {
		return 0, err
	}
	s, err := tensor.ToSlice[int64](pos)
	if err != nil {
		return 0, err
	}
	if s[0] > math.MaxInt32 {
		return 0, fmt.Errorf("position %d out of range", s[0])
	}
	return int(s[0]), nil
}
