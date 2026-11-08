package ops

import (
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// The ring keeps the last Cap positions written (wrapping), and decode
// attention over its valid slots equals plain softmax attention over the
// same keys in time order; q and the written rows are strided views of a
// fused projection, as the Moshi decode path passes them. Lengths cross
// one 128-key split and the wrap point; CUDA vs CPU when present.
func TestKVRingAttnDecode(t *testing.T) {
	const B, H, D, Cap = 2, 3, 40, 300
	rng := rand.New(rand.NewPCG(1, 2))
	randT := func(dims ...int) []float32 {
		n := 1
		for _, d := range dims {
			n *= d
		}
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		return x
	}
	devs := []device.Device{device.CPU()}
	if device.CUDAAvailable() {
		g, err := device.CUDA(0)
		if err != nil {
			t.Fatal(err)
		}
		devs = append(devs, g)
	}
	scale := float32(1 / math.Sqrt(D))
	for _, steps := range []int{1, 129, 300, 413} {
		// Reference history: keys / values per (b, h, position).
		keys := randT(steps, B, 3*H, D) // per step a fused (B, 3H, 1, D) qkv
		var results [][]float32
		for _, dev := range devs {
			kr, _ := tensor.Zeros(dtype.F32, []int{B, H, Cap, D}, dev)
			vr, _ := tensor.Zeros(dtype.F32, []int{B, H, Cap, D}, dev)
			var out *tensor.Tensor
			for s := range steps {
				qkv, err := tensor.New(keys[s*B*3*H*D:(s+1)*B*3*H*D], []int{B, 3 * H, 1, D}, dev)
				if err != nil {
					t.Fatal(err)
				}
				q, _ := qkv.Narrow(1, 0, H)
				k, _ := qkv.Narrow(1, H, H)
				v, _ := qkv.Narrow(1, 2*H, H)
				if err := KVRingWrite(kr, k, s); err != nil {
					t.Fatal(err)
				}
				if err := KVRingWrite(vr, v, s); err != nil {
					t.Fatal(err)
				}
				if s == steps-1 {
					if out, err = AttnDecode(q, kr, vr, min(s+1, Cap), scale); err != nil {
						t.Fatal(err)
					}
				}
			}
			c, _ := out.To(device.CPU())
			results = append(results, tensor.MustSlice[float32](c))
		}
		// Naive attention of the last query over the last min(steps, Cap) positions.
		want := make([]float32, B*H*D)
		last := steps - 1
		first := max(0, steps-Cap)
		at := func(s, b, part, h, d int) float32 { return keys[(((s*B+b)*3*H)+part*H+h)*D+d] }
		for b := range B {
			for h := range H {
				var sc []float64
				m := math.Inf(-1)
				for s := first; s <= last; s++ {
					var dot float64
					for d := range D {
						dot += float64(at(last, b, 0, h, d)) * float64(at(s, b, 1, h, d))
					}
					sc = append(sc, dot*float64(scale))
					m = math.Max(m, dot*float64(scale))
				}
				var l float64
				for i := range sc {
					sc[i] = math.Exp(sc[i] - m)
					l += sc[i]
				}
				for d := range D {
					var a float64
					for i, s := 0, first; s <= last; i, s = i+1, s+1 {
						a += sc[i] / l * float64(at(s, b, 2, h, d))
					}
					want[(b*H+h)*D+d] = float32(a)
				}
			}
		}
		for i, got := range results {
			var md float64
			for j := range want {
				md = math.Max(md, math.Abs(float64(got[j]-want[j])))
			}
			if md > 2e-5 {
				t.Errorf("steps %d, %v: max |diff| %.2e vs naive attention", steps, devs[i].Kind(), md)
			}
		}
	}
}
