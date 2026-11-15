package nn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// A SparseMoE whose experts stream through 3 slots computes exactly what
// the fully resident one does: an 11-token batch (more distinct experts
// than slots: token chunks), single-token steps (slot reuse, LRU
// eviction), with and without background prefetch; fewer slots than top-k
// is an error.
func TestExpertStreamMatchesResident(t *testing.T) {
	devs := []device.Device{device.CPU()} // MoE routing (TopK) has no WebGPU kernel yet
	const H, I, E, K = 16, 8, 8, 2
	rng := rand.New(rand.NewPCG(5, 6))
	rnd := func(n int, s float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64() * s)
		}
		return v
	}
	gw, guw, dw := rnd(E*H, 0.8), rnd(E*2*I*H, 0.3), rnd(E*H*I, 0.3)
	xs := [][]float32{rnd(11*H, 1), rnd(H, 1), rnd(H, 1), rnd(H, 1), rnd(H, 1), rnd(3*H, 1)}
	for _, dev := range devs {
		t.Run(fmt.Sprint(dev), func(t *testing.T) {
			full, err := NewSparseMoEUninit(H, I, E, K, true, dtype.F32, dev)
			if err != nil {
				t.Fatal(err)
			}
			g, _ := tensor.New(gw, []int{E, H}, dev)
			if err := full.Gate.SetWeight(g); err != nil {
				t.Fatal(err)
			}
			full.GateUp, _ = tensor.New(guw, []int{E, 2 * I, H}, dev)
			full.Down, _ = tensor.New(dw, []int{E, H, I}, dev)
			load := func(e int) (*tensor.Tensor, *tensor.Tensor, error) {
				a, _ := tensor.New(guw[e*2*I*H:(e+1)*2*I*H], []int{2 * I, H}, device.CPU())
				b, _ := tensor.New(dw[e*H*I:(e+1)*H*I], []int{H, I}, device.CPU())
				return a, b, nil
			}
			for _, prefetch := range []bool{false, true} {
				st, err := NewExpertStream(3, dev, load)
				if err != nil {
					t.Fatal(err)
				}
				streamed := &SparseMoE{Gate: full.Gate, TopK: K, NormTopK: true, Stream: st}
				md := 0.0
				for i, xv := range xs {
					x, _ := tensor.New(xv, []int{len(xv) / H, H}, dev)
					if prefetch && i > 0 {
						st.Prefetch([]int{i % E, (i + 3) % E})
					}
					want, err := full.Forward(x)
					if err != nil {
						t.Fatal(err)
					}
					got, err := streamed.Forward(x)
					if err != nil {
						t.Fatal(err)
					}
					a, _ := want.To(device.CPU())
					b, _ := got.To(device.CPU())
					av, bv := tensor.MustSlice[float32](a), tensor.MustSlice[float32](b)
					for j := range av {
						md = math.Max(md, math.Abs(float64(av[j]-bv[j])))
					}
					if n := len(st.Resident()); n > 3 {
						t.Fatalf("%d experts resident in 3 slots", n)
					}
				}
				t.Logf("prefetch %v: max diff %.3g, %d loads, %d hits", prefetch, md, st.Loads, st.Hits)
				if !(md <= 1e-5) {
					t.Errorf("prefetch %v: streamed output differs by %.3g", prefetch, md)
				}
				if st.Loads <= 3 {
					t.Errorf("only %d loads: the 11-token batch should cycle the slots", st.Loads)
				}
			}
			st, _ := NewExpertStream(1, dev, load)
			x, _ := tensor.New(xs[1], []int{1, H}, dev)
			if _, err := (&SparseMoE{Gate: full.Gate, TopK: K, Stream: st}).Forward(x); err == nil {
				t.Error("1 slot for top-2 routing should fail")
			}
		})
	}
}
