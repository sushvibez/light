package models

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/tensor"
)

// Streamed blocks under 16-bit weight storage (the 12B path on a 16 GB
// card) compute what the resident model computes, on every device.
func TestFluxStreamedBF16MatchesResident(t *testing.T) {
	devs := []device.Device{device.CPU()}
	if g, err := device.CUDA(0); err == nil {
		devs = append(devs, g)
	}
	dir := "../diffusion/testdata/flux-tiny-pipe/transformer"
	for _, dev := range devs {
		for _, slots := range []int{1, 2} { // 2: the next block uploads in the background
			t.Run(fmt.Sprintf("%s/slots=%d", dev, slots), func(t *testing.T) {
				var res, str *FluxTransformer
				err := nn.WithWeightStorage16(dtype.BF16, func() (err error) {
					if res, err = LoadFluxTransformerOpts(dir, dev, FluxTransformerLoadOptions{}); err != nil {
						return err
					}
					str, err = LoadFluxTransformerOpts(dir, dev, FluxTransformerLoadOptions{Stream: true, Slots: slots})
					return err
				})
				if err != nil {
					t.Fatal(err)
				}
				if w := res.Double[0].Img.Q.W; w.DType() != dtype.BF16 {
					t.Fatalf("resident weights %v", w.DType())
				}
				cfg := res.Cfg
				rng := rand.New(rand.NewPCG(3, 4))
				rnd := func(shape ...int) *tensor.Tensor {
					n := 1
					for _, d := range shape {
						n *= d
					}
					v := make([]float32, n)
					for i := range v {
						v[i] = float32(rng.NormFloat64())
					}
					x, err := tensor.New(v, shape, dev)
					if err != nil {
						t.Fatal(err)
					}
					return x
				}
				S, St := 40, 7
				ids := make([]float32, S*3)
				for i := range S {
					ids[i*3+1], ids[i*3+2] = float32(i/8), float32(i%8)
				}
				in := FluxInputs{Image: rnd(1, S, cfg.InChannels), Text: rnd(1, St, cfg.JointDim), Pooled: rnd(1, cfg.PooledDim),
					Timestep: []float32{0.9}, ImgIDs: ids, TxtIDs: make([]float32, St*3)}
				a, err := res.Forward(in)
				if err != nil {
					t.Fatal(err)
				}
				for pass := range 3 { // the blocks reload every pass (one slot)
					b, err := str.Forward(in)
					if err != nil {
						t.Fatal(err)
					}
					ac, _ := a.To(device.CPU())
					bc, _ := b.To(device.CPU())
					av, bv := tensor.MustSlice[float32](ac), tensor.MustSlice[float32](bc)
					md, ma := 0.0, 0.0
					for i := range av {
						md = math.Max(md, math.Abs(float64(av[i]-bv[i])))
						ma = math.Max(ma, math.Abs(float64(av[i])))
					}
					t.Logf("pass %d: streamed vs resident max diff %.3g (max |y| %.3g)", pass, md, ma)
					if md > 1e-5 {
						t.Errorf("pass %d: differ by %.3g", pass, md)
					}
				}
			})
		}
	}
}
