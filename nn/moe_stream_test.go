package nn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/quant"
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

// 4-bit slots (NewExpertStreamQ4) run the resident 4-bit experts' kernel
// (ops.MoEInt4): through a slot per expert the output is bit-identical to
// GateUpQ4 / DownQ4; through 3 slots (token chunks, LRU eviction, with and
// without prefetch) within rounding. Mispredicted prefetches are dropped
// by the next forward instead of staying pending, and the slot buffers
// hold the packed weights only (allocated as slots fill).
func TestExpertStreamQ4MatchesResident(t *testing.T) {
	const H, I, E, K, G = 64, 32, 8, 2, 32
	rng := rand.New(rand.NewPCG(7, 8))
	rnd := func(n int, s float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64() * s)
		}
		return v
	}
	cpu := device.CPU()
	int4 := func(N, Kd int) quant.Int4 {
		codes := make([]uint32, N*Kd/8)
		for i := range codes {
			codes[i] = rng.Uint32()
		}
		sc, zr := rnd(N*Kd/G, 0.02), make([]float32, N*Kd/G)
		for i := range zr {
			zr[i] = 8 + float32(rng.NormFloat64())
		}
		c, _ := tensor.New(codes, []int{N, Kd / 8}, cpu)
		s, _ := tensor.New(sc, []int{N, Kd / G}, cpu)
		z, _ := tensor.New(zr, []int{N, Kd / G}, cpu)
		return quant.Int4{Codes: c, Scales: s, Zeros: z, Group: G}
	}
	gu, dn := make([]quant.Int4, E), make([]quant.Int4, E)
	for e := range E {
		gu[e], dn[e] = int4(2*I, H), int4(H, I)
	}
	perExpert := int64(2*I*H/8*4 + 2*2*I*H/G*4 + H*I/8*4 + 2*H*I/G*4)
	gate, err := NewLinearUninit(H, E, false, dtype.F32, cpu)
	if err != nil {
		t.Fatal(err)
	}
	g, _ := tensor.New(rnd(E*H, 0.8), []int{E, H}, cpu)
	if err := gate.SetWeight(g); err != nil {
		t.Fatal(err)
	}
	full := &SparseMoE{Gate: gate, TopK: K, NormTopK: true, GateUpQ4: gu, DownQ4: dn}
	load := func(e int) (quant.Int4, quant.Int4, error) { return gu[e], dn[e], nil }
	xs := [][]float32{rnd(11*H, 1), rnd(H, 1), rnd(H, 1), rnd(H, 1), rnd(H, 1), rnd(3*H, 1)}
	for _, c := range []struct {
		slots    int
		prefetch bool
	}{{E, false}, {3, false}, {3, true}} {
		st, err := NewExpertStreamQ4(c.slots, cpu, load)
		if err != nil {
			t.Fatal(err)
		}
		streamed := &SparseMoE{Gate: gate, TopK: K, NormTopK: true, Stream: st}
		md := 0.0
		for i, xv := range xs {
			x, _ := tensor.New(xv, []int{len(xv) / H, H}, cpu)
			if c.prefetch && i > 0 {
				st.Prefetch([]int{i % E, (i + 3) % E, (i + 5) % E})
			}
			want, err := full.Forward(x)
			if err != nil {
				t.Fatal(err)
			}
			got, err := streamed.Forward(x)
			if err != nil {
				t.Fatal(err)
			}
			av, bv := tensor.MustSlice[float32](want), tensor.MustSlice[float32](got)
			for j := range av {
				md = math.Max(md, math.Abs(float64(av[j]-bv[j])))
			}
			if n := len(st.Resident()); n > c.slots {
				t.Fatalf("%d experts resident in %d slots", n, c.slots)
			}
			if p := st.Pending(); p != 0 {
				t.Fatalf("%d loads still pending after a forward", p)
			}
			if b := st.SlotBytes(); b > int64(c.slots)*perExpert || b%perExpert != 0 {
				t.Fatalf("slots hold %d B, %d B an expert", b, perExpert)
			}
		}
		t.Logf("%d slots, prefetch %v: max diff %.3g, %d loads, %d hits, %d prefetched (%d used), %d B in slots",
			c.slots, c.prefetch, md, st.Loads, st.Hits, st.Prefetched, st.PrefetchUsed, st.SlotBytes())
		tol := 1e-5
		if c.slots == E {
			tol = 0
		}
		if !(md <= tol) {
			t.Errorf("%d slots: streamed output differs by %.3g", c.slots, md)
		}
		if c.slots < E && st.Loads <= int64(c.slots) {
			t.Errorf("only %d loads: the 11-token batch should cycle the slots", st.Loads)
		}
		if c.prefetch && (st.Prefetched == 0 || st.PrefetchUsed >= st.Prefetched) {
			t.Errorf("prefetched %d, used %d: want some used and some dropped", st.Prefetched, st.PrefetchUsed)
		}
	}
}
