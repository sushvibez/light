package models

import (
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// appendKV must produce the same K / V the concat path did, across
// prefill, single-token steps, a capacity doubling and a truncate.
func TestKVCacheAppendMatchesConcat(t *testing.T) {
	testKVCacheAppend(t, device.CPU())
}

func TestKVCacheAppendMatchesConcatCUDA(t *testing.T) {
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	testKVCacheAppend(t, dev)
}

func testKVCacheAppend(t *testing.T, dev device.Device) {
	rng := rand.New(rand.NewPCG(3, 5))
	B, KH, HD := 2, 3, 4
	rnd := func(T int) *tensor.Tensor {
		d := make([]float32, B*KH*T*HD)
		for i := range d {
			d[i] = float32(rng.NormFloat64())
		}
		x, _ := tensor.New(d, []int{B, KH, T, HD}, device.CPU())
		x, _ = ops.Cast(x, dtype.BF16)
		x, _ = x.To(dev)
		return x
	}
	cache := NewKVCache(1)
	var refK, refV *tensor.Tensor
	steps := []int{5, 1, 1, 300, 1, 1} // 300 pushes past the first 256-slot buffer
	for i, T := range steps {
		k, v := rnd(T), rnd(T)
		if refK == nil {
			refK, refV = k, v
		} else {
			refK, _ = ops.Concat([]*tensor.Tensor{refK, k}, 2)
			refV, _ = ops.Concat([]*tensor.Tensor{refV, v}, 2)
		}
		gotK, gotV, err := cache.appendKV(0, k, v)
		if err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		cache.Len += T
		if gotK.Dim(2) != refK.Dim(2) {
			t.Fatalf("step %d: %d positions, want %d", i, gotK.Dim(2), refK.Dim(2))
		}
		assertSame(t, gotK, refK, "K", i)
		assertSame(t, gotV, refV, "V", i)
		if cache.K[0] != gotK || cache.kBuf[0] == nil || cache.K[0].Storage() != cache.kBuf[0].Storage() {
			t.Fatalf("step %d: K[0] is not a view of the capacity buffer", i)
		}
	}
	if err := cache.Truncate(7); err != nil {
		t.Fatal(err)
	}
	refK7, _ := refK.Narrow(2, 0, 7)
	assertSame(t, cache.K[0], refK7, "K after truncate", -1)
	k, v := rnd(2), rnd(2)
	gotK, _, err := cache.appendKV(0, k, v)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := ops.Concat([]*tensor.Tensor{refK7, k}, 2)
	assertSame(t, gotK, want, "K after truncate+append", -1)
	// Reset keeps the buffer; the next prompt starts at position 0.
	cache.Reset()
	if cache.kBuf[0] == nil || cache.K[0] != nil {
		t.Fatal("Reset should keep the capacity buffer and drop the view")
	}
	k, v = rnd(3), rnd(3)
	if gotK, _, err = cache.appendKV(0, k, v); err != nil {
		t.Fatal(err)
	}
	assertSame(t, gotK, k, "K after reset", -1)
}

func assertSame(t *testing.T, got, want *tensor.Tensor, what string, step int) {
	t.Helper()
	g, _ := got.Contiguous()
	g, _ = g.To(device.CPU())
	g, _ = ops.Cast(g, dtype.F32)
	w, _ := want.Contiguous()
	w, _ = w.To(device.CPU())
	w, _ = ops.Cast(w, dtype.F32)
	gs, ws := tensor.MustSlice[float32](g), tensor.MustSlice[float32](w)
	if len(gs) != len(ws) {
		t.Fatalf("%s step %d: %d elements, want %d", what, step, len(gs), len(ws))
	}
	for i := range gs {
		if gs[i] != ws[i] {
			t.Fatalf("%s step %d: [%d] = %v, want %v", what, step, i, gs[i], ws[i])
		}
	}
}
