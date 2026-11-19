package models

import (
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// The tiny transformer as a BFL single file (scripts/verify/make_flux_bfl.py,
// the inverse of diffusers' converter, checked against it) runs exactly
// like the diffusers folder.
func TestFluxSingleFileMatchesFolder(t *testing.T) {
	cpu := device.CPU()
	ref, err := LoadFluxTransformer("../diffusion/testdata/flux-tiny-pipe/transformer", cpu)
	if err != nil {
		t.Fatal(err)
	}
	got, err := LoadFluxTransformer("testdata/flux-tiny-bfl.safetensors", cpu)
	if err != nil {
		t.Fatal(err)
	}
	if got.Cfg.NumLayers != ref.Cfg.NumLayers || (got.Cfg.NumSingleLayers != ref.Cfg.NumSingleLayers || got.Cfg.HeadDim != ref.Cfg.HeadDim || got.Cfg.InChannels != ref.Cfg.InChannels) {
		t.Errorf("config %+v, want %+v", got.Cfg, ref.Cfg)
	}
	cfg := ref.Cfg
	rng := rand.New(rand.NewPCG(1, 2))
	rnd := func(shape ...int) *tensor.Tensor {
		n := 1
		for _, d := range shape {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		x, err := tensor.New(v, shape, cpu)
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	S, St := 12, 5
	ids := func(n int) []float32 {
		v := make([]float32, n*3)
		for i := range n {
			v[i*3+1], v[i*3+2] = float32(i/4), float32(i%4)
		}
		return v
	}
	in := FluxInputs{Image: rnd(1, S, cfg.InChannels), Text: rnd(1, St, cfg.JointDim), Pooled: rnd(1, cfg.PooledDim),
		Timestep: []float32{0.7}, ImgIDs: ids(S), TxtIDs: make([]float32, St*3)}
	if cfg.GuidanceEmbeds {
		in.Guidance = []float32{3.5}
	}
	a, err := ref.Forward(in)
	if err != nil {
		t.Fatal(err)
	}
	b, err := got.Forward(in)
	if err != nil {
		t.Fatal(err)
	}
	av, bv := tensor.MustSlice[float32](a), tensor.MustSlice[float32](b)
	md := 0.0
	for i := range av {
		md = math.Max(md, math.Abs(float64(av[i]-bv[i])))
	}
	t.Logf("single file vs folder: max diff %.3g over %d values", md, len(av))
	if md > 1e-6 {
		t.Errorf("outputs differ by %.3g", md)
	}
	if a.DType() != dtype.F32 {
		t.Errorf("dtype %v", a.DType())
	}
}
