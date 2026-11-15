package nn

import (
	"math/rand/v2"
	"testing"

	"light/device"
	"light/tensor"
)

// Chunked SDPA (forced with a tiny score budget) equals the one-shot result,
// with and without a per-query mask.
func TestSDPAChunkedMatchesWhole(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	mk := func(dims ...int) *tensor.Tensor {
		n := 1
		for _, d := range dims {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		x, err := tensor.New(v, dims, device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	q, k, v := mk(2, 3, 37, 8), mk(2, 3, 29, 8), mk(2, 3, 29, 8)
	mask := mk(1, 1, 37, 29)
	for _, m := range []*tensor.Tensor{nil, mask} {
		want, err := SDPA(q, k, v, m)
		if err != nil {
			t.Fatal(err)
		}
		old := sdpaChunkBytes
		sdpaChunkBytes = 2 * 3 * 29 * 4 * 5 // 5 query rows per chunk
		got, err := SDPA(q, k, v, m)
		sdpaChunkBytes = old
		if err != nil {
			t.Fatal(err)
		}
		a, b := tensor.MustSlice[float32](got), tensor.MustSlice[float32](want)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("mask=%v: element %d: %g, whole %g", m != nil, i, a[i], b[i])
			}
		}
	}
}
