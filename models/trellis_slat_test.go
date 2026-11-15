package models

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/safetensors"
	"light/splat"
	"light/tensor"
)

// trellisSLatRef is testdata/trellis-tiny-slat.safetensors
// (scripts/verify/ref_trellis_slat.py).
func trellisSLatRef(t *testing.T) map[string]*tensor.Tensor {
	t.Helper()
	ref, _, err := safetensors.LoadFiles([]string{"testdata/trellis-tiny-slat.safetensors"}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	return ref
}

func trellisF32(t *testing.T, x *tensor.Tensor) []float32 {
	t.Helper()
	if x.DType() != dtype.F32 {
		t.Fatalf("%v tensor, want F32", x.DType())
	}
	return tensor.MustSlice[float32](x)
}

func trellisCoords4(t *testing.T, x *tensor.Tensor) [][4]int32 {
	t.Helper()
	if x.DType() != dtype.I32 || x.Rank() != 2 || x.Dim(1) != 4 {
		t.Fatalf("coords %v %v, want I32 (N, 4)", x.DType(), x.Shape())
	}
	v := tensor.MustSlice[int32](x)
	out := make([][4]int32, x.Dim(0))
	for i := range out {
		copy(out[i][:], v[4*i:4*i+4])
	}
	return out
}

// trellisSLatNorm reads pipeline.json's slat_normalization.
func trellisSLatNorm(t *testing.T) (mean, std []float64) {
	t.Helper()
	raw, err := os.ReadFile("testdata/trellis-tiny/pipeline.json")
	if err != nil {
		t.Fatal(err)
	}
	var pc struct {
		Args struct {
			Norm struct{ Mean, Std []float64 } `json:"slat_normalization"`
		} `json:"args"`
	}
	if err := json.Unmarshal(raw, &pc); err != nil {
		t.Fatal(err)
	}
	return pc.Args.Norm.Mean, pc.Args.Norm.Std
}

// Tiny TRELLIS stage 2 (TRELLIS's own sparse modules with plain-PyTorch
// spconv / xformers stand-ins, seeded, F32 from F16-stored torsos) against
// the reference: one SLat flow forward on 300 unsorted voxels, the 25-step
// guided SLat sample on stage 1's 1186 voxels and its un-normalisation, the
// swin Gaussian decoder's outputs, the activated Gaussians and save_ply's
// rows.
func TestTrellisSLatTinyMatchesReference(t *testing.T) {
	ref := trellisSLatRef(t)
	ref1 := loadTrellisTinyRef(t)
	mean, std := trellisSLatNorm(t)
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		t.Run(fmt.Sprint(dev), func(t *testing.T) {
			up := func(x *tensor.Tensor) *tensor.Tensor {
				y, err := x.To(dev)
				if err != nil {
					t.Fatal(err)
				}
				return y
			}
			flow, err := LoadTrellisSLatFlow("testdata/trellis-tiny/ckpts/slat_flow_tiny_fp16", dev)
			if err != nil {
				t.Fatal(err)
			}
			fg, err := NewTrellisGrid(trellisCoords4(t, ref["flow.coords"]), dev)
			if err != nil {
				t.Fatal(err)
			}
			v, err := flow.Forward(fg, up(ref["flow.x"]), trellisF32(t, ref["flow.t"]), up(ref["flow.cond"]))
			if err != nil {
				t.Fatal(err)
			}
			trellisCmp(t, "slat flow velocity", v, trellisF32(t, ref["flow.v"]), 1e-4)

			g, err := NewTrellisGrid(trellisCoords4(t, ref["sample.coords"]), dev)
			if err != nil {
				t.Fatal(err)
			}
			cond, err := tensor.New(ref1.DINOv2.Tokens, ref1.DINOv2.Shape, dev)
			if err != nil {
				t.Fatal(err)
			}
			raw, err := TrellisSLatSample(flow, g, up(ref["sample.noise"]), cond, nil,
				TrellisSampleOptions{Steps: 25, CFG: 5, Interval: [2]float64{0.5, 1}, RescaleT: 3})
			if err != nil {
				t.Fatal(err)
			}
			trellisCmp(t, "slat sample", raw, trellisF32(t, ref["sample.raw"]), 1e-4)
			slat, err := TrellisSLatUnnormalize(raw, mean, std)
			if err != nil {
				t.Fatal(err)
			}
			trellisCmp(t, "slat (un-normalised)", slat, trellisF32(t, ref["sample.slat"]), 1e-4)

			dec, err := LoadTrellisSLatDecoder("testdata/trellis-tiny/ckpts/slat_dec_gs_tiny_fp16", dev)
			if err != nil {
				t.Fatal(err)
			}
			feats, err := dec.Forward(g, up(ref["sample.slat"]))
			if err != nil {
				t.Fatal(err)
			}
			trellisCmp(t, "decoder outputs", feats, trellisF32(t, ref["dec.feats"]), 1e-4)
			gs, err := dec.Gaussians(g, feats)
			if err != nil {
				t.Fatal(err)
			}
			host := func(v []float32, dims ...int) *tensor.Tensor {
				x, err := tensor.New(v, dims, device.CPU())
				if err != nil {
					t.Fatal(err)
				}
				return x
			}
			n := len(gs.Opacity)
			trellisCmp(t, "gaussian xyz", host(gs.XYZ, n, 3), trellisF32(t, ref["gs.xyz"]), 1e-4)
			trellisCmp(t, "gaussian features_dc", host(gs.FeaturesDC, n, 3), trellisF32(t, ref["gs.features_dc"]), 1e-4)
			trellisCmp(t, "gaussian scaling", host(gs.Scaling, n, 3), trellisF32(t, ref["gs.scaling"]), 1e-4)
			trellisCmp(t, "gaussian rotation", host(gs.Rotation, n, 4), trellisF32(t, ref["gs.rotation"]), 1e-4)
			trellisCmp(t, "gaussian opacity", host(gs.Opacity, n), trellisF32(t, ref["gs.opacity"]), 1e-4)
			trellisCmp(t, "save_ply rows", host(splat.PLYRows(gs), n, 17), trellisF32(t, ref["gs.ply"]), 1e-4)

			// The decoder on Light's own sample.
			feats, err = dec.Forward(g, slat)
			if err != nil {
				t.Fatal(err)
			}
			trellisCmp(t, "decoder outputs (own slat)", feats, trellisF32(t, ref["dec.feats"]), 1e-4)
		})
	}
}

// The sparse primitives on a hand-checked case: neighbour pairs, the
// downsample's Σ / (n + 1) mean (torch.scatter_reduce with include_self),
// upsample, and the window partition.
func TestTrellisSparsePrimitives(t *testing.T) {
	dev := device.CPU()
	g, err := NewTrellisGrid([][4]int32{{0, 1, 1, 1}, {0, 0, 1, 1}, {0, 5, 5, 5}, {0, 1, 1, 0}}, dev)
	if err != nil {
		t.Fatal(err)
	}
	nb, err := g.neighbours()
	if err != nil {
		t.Fatal(err)
	}
	// Offset (−1, 0, 0) is k = 0·9 + 1·3 + 1 = 4: voxel 0's neighbour is voxel 1.
	src := tensor.MustSlice[int64](nb[4].src)
	if nb[4].n != 1 || src[0] != 1 {
		t.Errorf("offset (-1,0,0): %d pairs %v, want voxel 1", nb[4].n, src)
	}
	x, err := tensor.New([]float32{1, 2, 3, 4}, []int{4, 1}, dev)
	if err != nil {
		t.Fatal(err)
	}
	o := &iop{}
	d, dg := g.downsample(o, x)
	u, _ := dg.upsample(o, d)
	if o.err != nil {
		t.Fatal(o.err)
	}
	// Parents (0,0,0) ← voxels 0, 1, 3 and (2,2,2) ← voxel 2.
	want := []float32{(1 + 2 + 4) / 4.0, 3 / 2.0}
	got := tensor.MustSlice[float32](d)
	if dg.N() != 2 || math.Abs(float64(got[0]-want[0])) > 1e-6 || math.Abs(float64(got[1]-want[1])) > 1e-6 {
		t.Errorf("downsample %v on %v, want %v", got, dg.Coords, want)
	}
	if uv := tensor.MustSlice[float32](u); uv[0] != got[0] || uv[1] != got[0] || uv[2] != got[1] || uv[3] != got[0] {
		t.Errorf("upsample %v", uv)
	}
	for _, c := range []struct {
		shift int
		want  [][2]int // (L, n) per group
	}{{0, [][2]int{{1, 1}, {3, 1}}}, {1, [][2]int{{1, 4}}}} {
		w, err := g.windowPartition(2, c.shift)
		if err != nil {
			t.Fatal(err)
		}
		var got [][2]int
		for _, gr := range w.groups {
			got = append(got, [2]int{gr.L, gr.n})
		}
		if fmt.Sprint(got) != fmt.Sprint(c.want) {
			t.Errorf("window 2 shift %d: groups (L, n) %v, want %v", c.shift, got, c.want)
		}
	}
}

// With 16-bit weight storage the stage-2 networks stay close to the
// reference.
func TestTrellisSLatTinyStorage16(t *testing.T) {
	ref := trellisSLatRef(t)
	dev := device.CPU()
	var flow *TrellisSLatFlow
	var dec *TrellisSLatDecoder
	if err := nn.WithWeightStorage16(dtype.F16, func() (err error) {
		if flow, err = LoadTrellisSLatFlow("testdata/trellis-tiny/ckpts/slat_flow_tiny_fp16", dev); err != nil {
			return err
		}
		dec, err = LoadTrellisSLatDecoder("testdata/trellis-tiny/ckpts/slat_dec_gs_tiny_fp16", dev)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if dt := flow.InBlocks[0].Conv1.W[0].W.DType(); dt != dtype.F16 {
		t.Fatalf("sparse conv weight stored as %v, want F16", dt)
	}
	fg, err := NewTrellisGrid(trellisCoords4(t, ref["flow.coords"]), dev)
	if err != nil {
		t.Fatal(err)
	}
	v, err := flow.Forward(fg, ref["flow.x"], trellisF32(t, ref["flow.t"]), ref["flow.cond"])
	if err != nil {
		t.Fatal(err)
	}
	trellisCmp(t, "slat flow velocity (16-bit weights)", v, trellisF32(t, ref["flow.v"]), 1e-2)
	g, err := NewTrellisGrid(trellisCoords4(t, ref["sample.coords"]), dev)
	if err != nil {
		t.Fatal(err)
	}
	feats, err := dec.Forward(g, ref["sample.slat"])
	if err != nil {
		t.Fatal(err)
	}
	trellisCmp(t, "decoder outputs (16-bit weights)", feats, trellisF32(t, ref["dec.feats"]), 1e-2)
}
