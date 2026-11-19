package models

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"light/device"
	"light/tensor"
)

// Tiny Idol (scripts/verify/ref_idol.py: idol-sdk's own modules, seeded,
// F32) against the SDK: the denoiser's velocity, a 3-step CFG sample, the
// octree decoder's logits and anchors (rand fixed at 0.5), and the
// Gaussian decoder's outputs and activated Gaussians.
func TestIdolTinyMatchesSDK(t *testing.T) {
	raw, err := os.ReadFile("testdata/idol-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Denoiser struct {
			Latent, Camera, Feature1, Feature2, VLatent, VCamera []float32
			T                                                    float32
			CondLen                                              int `json:"cond_len"`
		}
		Sample struct {
			NoiseLatent []float32 `json:"noise_latent"`
			NoiseCamera []float32 `json:"noise_camera"`
			Steps       int
			CFG, Shift  float64
			Latent      []float32
			Camera      []float32
		}
		Octree struct {
			Cond, Cells, Logits, Anchors []float32
			Level                        int
			NumPoints                    int `json:"num_points"`
		}
		Gaussians struct {
			Points, Features, XYZ, Scaling, Rotation, Opacity []float32
			FeaturesDC                                        []float32 `json:"features_dc"`
		}
	}
	// The JSON keys are snake_case; map the few multi-word ones by hand.
	var keys map[string]map[string]json.RawMessage
	if err := json.Unmarshal(raw, &keys); err != nil {
		t.Fatal(err)
	}
	get := func(sec, k string, v any) {
		if err := json.Unmarshal(keys[sec][k], v); err != nil {
			t.Fatalf("%s.%s: %v", sec, k, err)
		}
	}
	d := &ref.Denoiser
	get("denoiser", "latent", &d.Latent)
	get("denoiser", "camera", &d.Camera)
	get("denoiser", "feature1", &d.Feature1)
	get("denoiser", "feature2", &d.Feature2)
	get("denoiser", "v_latent", &d.VLatent)
	get("denoiser", "v_camera", &d.VCamera)
	get("denoiser", "t", &d.T)
	get("denoiser", "cond_len", &d.CondLen)
	s := &ref.Sample
	get("sample", "noise_latent", &s.NoiseLatent)
	get("sample", "noise_camera", &s.NoiseCamera)
	get("sample", "steps", &s.Steps)
	get("sample", "cfg", &s.CFG)
	get("sample", "shift", &s.Shift)
	get("sample", "latent", &s.Latent)
	get("sample", "camera", &s.Camera)
	oc := &ref.Octree
	get("octree", "cond", &oc.Cond)
	get("octree", "cells", &oc.Cells)
	get("octree", "logits", &oc.Logits)
	get("octree", "anchors", &oc.Anchors)
	get("octree", "level", &oc.Level)
	get("octree", "num_points", &oc.NumPoints)
	g := &ref.Gaussians
	get("gaussians", "points", &g.Points)
	get("gaussians", "features", &g.Features)
	get("gaussians", "xyz", &g.XYZ)
	get("gaussians", "features_dc", &g.FeaturesDC)
	get("gaussians", "scaling", &g.Scaling)
	get("gaussians", "rotation", &g.Rotation)
	get("gaussians", "opacity", &g.Opacity)

	devs := []device.Device{device.CPU()}
	if gpu, err := device.WebGPU(0); err == nil {
		devs = append(devs, gpu)
	}
	cmpS := func(t *testing.T, name string, got, want []float32, tol float64) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
		}
		md, mr := 0.0, 0.0
		for i := range got {
			md = math.Max(md, math.Abs(float64(got[i]-want[i])))
			mr = math.Max(mr, math.Abs(float64(want[i])))
		}
		t.Logf("%s: max diff %.3g (max |ref| %.3g)", name, md, mr)
		if md > tol*math.Max(1, mr) {
			t.Errorf("%s differs by %.3g", name, md)
		}
	}
	host := func(t *testing.T, x *tensor.Tensor) []float32 {
		t.Helper()
		c, err := x.To(device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		return tensor.MustSlice[float32](c)
	}
	for _, dev := range devs {
		t.Run(fmt.Sprint(dev), func(t *testing.T) {
			m, err := LoadIdol("testdata/idol-tiny", dev)
			if err != nil {
				t.Fatal(err)
			}
			a := m.Cfg.Denoiser.Args
			N := len(d.Latent) / a.InChannels
			nt := func(v []float32, dims ...int) *tensor.Tensor {
				x, err := tensor.New(v, dims, dev)
				if err != nil {
					t.Fatal(err)
				}
				return x
			}
			f1 := nt(d.Feature1, 1, d.CondLen, a.CondChannels)
			f2 := nt(d.Feature2, 1, d.CondLen, a.Cond2Channels)
			vl, vc, err := m.Denoiser.Forward(IdolInputs{Latent: nt(d.Latent, 1, N, a.InChannels),
				Camera: nt(d.Camera, 1, 1, a.CamChannels), T: []float32{d.T}, Feature1: f1, Feature2: f2})
			if err != nil {
				t.Fatal(err)
			}
			cmpS(t, "denoiser latent velocity", host(t, vl), d.VLatent, 1e-4)
			cmpS(t, "denoiser camera velocity", host(t, vc), d.VCamera, 1e-4)

			lat, cam, err := IdolSample(m.Denoiser, nt(s.NoiseLatent, 1, N, a.InChannels), nt(s.NoiseCamera, 1, 1, a.CamChannels),
				f1, f2, IdolSampleOptions{Steps: s.Steps, CFG: s.CFG, Shift: s.Shift})
			if err != nil {
				t.Fatal(err)
			}
			cmpS(t, "sample latent", host(t, lat), s.Latent, 2e-4)
			cmpS(t, "sample camera", host(t, cam), s.Camera, 2e-4)

			cond := nt(oc.Cond, 1, len(oc.Cond)/16, 16)
			logits, err := m.Octree.Forward(nt(oc.Cells, 1, len(oc.Cells)/3, 3), oc.Level, cond)
			if err != nil {
				t.Fatal(err)
			}
			cmpS(t, "octree logits", host(t, logits), oc.Logits, 1e-4)
			anchors, err := IdolSampleAnchors(m.Octree, cond, oc.NumPoints, m.Cfg.MaxVoxelLevel, func() float32 { return 0.5 })
			if err != nil {
				t.Fatal(err)
			}
			cmpS(t, "octree anchors", anchors, oc.Anchors, 1e-6)

			feats, err := m.Gaussian.Forward(nt(g.Points, 1, len(g.Points)/3, 3), cond)
			if err != nil {
				t.Fatal(err)
			}
			cmpS(t, "gaussian decoder", host(t, feats), g.Features, 1e-4)
			gs, err := m.Gaussian.ToGaussians(g.Points, feats)
			if err != nil {
				t.Fatal(err)
			}
			cmpS(t, "xyz", gs.XYZ, g.XYZ, 1e-4)
			cmpS(t, "features_dc", gs.FeaturesDC, g.FeaturesDC, 1e-4)
			cmpS(t, "scaling", gs.Scaling, g.Scaling, 1e-4)
			cmpS(t, "rotation", gs.Rotation, g.Rotation, 1e-4)
			cmpS(t, "opacity", gs.Opacity, g.Opacity, 1e-4)
		})
	}
}
