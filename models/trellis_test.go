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
	"light/tensor"
)

// trellisTinyRef is testdata/trellis-tiny.json (scripts/verify/ref_trellis_ss.py).
type trellisTinyRef struct {
	DINOv2 struct {
		Pixels, Tokens []float32
		Size           int
		Shape          []int
	} `json:"dinov2"`
	Flow struct {
		X, Cond, V []float32
		T          float32
		CondLen    int `json:"cond_len"`
	} `json:"flow"`
	Sample struct {
		Noise, Latent, Logits []float32
		Steps                 int
		CFG                   float64
		Interval              [2]float64
		RescaleT              float64 `json:"rescale_t"`
		Coords                []int32
	} `json:"sample"`
}

func loadTrellisTinyRef(t *testing.T) *trellisTinyRef {
	raw, err := os.ReadFile("testdata/trellis-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var r trellisTinyRef
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	return &r
}

func trellisCmp(t *testing.T, name string, x *tensor.Tensor, want []float32, tol float64) {
	t.Helper()
	h, err := x.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](h)
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	md, mr := 0.0, 0.0
	for i := range got {
		md = math.Max(md, math.Abs(float64(got[i]-want[i])))
		mr = math.Max(mr, math.Abs(float64(want[i])))
	}
	t.Logf("%s: max diff %.3g (max |ref| %.3g, rel %.3g)", name, md, mr, md/math.Max(1, mr))
	if md > tol*math.Max(1, mr) {
		t.Errorf("%s differs by %.3g", name, md)
	}
}

// TrellisCoordsMatch compares coords with the reference, excusing voxels
// whose reference logit is within tol of 0.
func trellisCoordsMatch(t *testing.T, got [][4]int32, want []int32, logits []float32, R int, tol float64) {
	t.Helper()
	key := func(c [4]int32) int { return ((int(c[0])*R+int(c[1]))*R+int(c[2]))*R + int(c[3]) }
	ref := map[int]bool{}
	for i := 0; i+3 < len(want); i += 4 {
		ref[key([4]int32{want[i], want[i+1], want[i+2], want[i+3]})] = true
	}
	mine := map[int]bool{}
	for _, c := range got {
		mine[key(c)] = true
	}
	bad, near := 0, 0
	for k := range len(logits) {
		if ref[k] != mine[k] {
			if math.Abs(float64(logits[k])) < tol {
				near++
			} else {
				bad++
			}
		}
	}
	t.Logf("coords: %d occupied (reference %d), %d differ at |logit| < %g, %d elsewhere", len(got), len(ref), near, tol, bad)
	if bad > 0 {
		t.Errorf("%d coords differ", bad)
	}
}

// Tiny TRELLIS stage 1 (TRELLIS's own modules, seeded, F32 from F16-stored
// torsos) against the reference: DINOv2 (hub = transformers) tokens, one
// flow forward, the 25-step guided sample, the decoder logits and coords.
func TestTrellisTinyMatchesReference(t *testing.T) {
	ref := loadTrellisTinyRef(t)
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		t.Run(fmt.Sprint(dev), func(t *testing.T) {
			nt := func(v []float32, dims ...int) *tensor.Tensor {
				x, err := tensor.New(v, dims, dev)
				if err != nil {
					t.Fatal(err)
				}
				return x
			}
			dino, err := LoadDINOv2("testdata/trellis-tiny/dinov2", dev)
			if err != nil {
				t.Fatal(err)
			}
			S := ref.DINOv2.Size
			mean, std := [3]float32{0.485, 0.456, 0.406}, [3]float32{0.229, 0.224, 0.225}
			px := make([]float32, len(ref.DINOv2.Pixels))
			for i, v := range ref.DINOv2.Pixels {
				c := i / (S * S)
				px[i] = (v - mean[c]) / std[c]
			}
			h, err := dino.Hidden(nt(px, 1, 3, S, S))
			if err != nil {
				t.Fatal(err)
			}
			tok, err := TrellisImageTokens(h)
			if err != nil {
				t.Fatal(err)
			}
			trellisCmp(t, "dinov2 tokens", tok, ref.DINOv2.Tokens, 1e-4)

			flow, err := LoadTrellisFlow("testdata/trellis-tiny/ckpts/ss_flow_tiny_fp16", dev)
			if err != nil {
				t.Fatal(err)
			}
			a := flow.Args
			R := a.Resolution
			v, err := flow.Forward(nt(ref.Flow.X, 1, a.InChannels, R, R, R), []float32{ref.Flow.T},
				nt(ref.Flow.Cond, 1, ref.Flow.CondLen, a.CondChannels))
			if err != nil {
				t.Fatal(err)
			}
			trellisCmp(t, "flow velocity", v, ref.Flow.V, 1e-4)

			z, err := TrellisSample(flow, nt(ref.Sample.Noise, 1, a.InChannels, R, R, R), tok, nil,
				TrellisSampleOptions{Steps: ref.Sample.Steps, CFG: ref.Sample.CFG, Interval: ref.Sample.Interval,
					RescaleT: ref.Sample.RescaleT})
			if err != nil {
				t.Fatal(err)
			}
			trellisCmp(t, "sampled latent", z, ref.Sample.Latent, 1e-4)

			dec, err := LoadTrellisSSDecoder("testdata/trellis-tiny/ckpts/ss_dec_tiny_fp16", dev)
			if err != nil {
				t.Fatal(err)
			}
			// The decoder on the reference latent, then on ours.
			lr, err := dec.Forward(nt(ref.Sample.Latent, 1, a.InChannels, R, R, R))
			if err != nil {
				t.Fatal(err)
			}
			trellisCmp(t, "decoder logits", lr, ref.Sample.Logits, 1e-4)
			// Again with the convs split into one-plane slabs.
			saved := trellisConvChunkBytes
			trellisConvChunkBytes = 1
			lr, err = dec.Forward(nt(ref.Sample.Latent, 1, a.InChannels, R, R, R))
			trellisConvChunkBytes = saved
			if err != nil {
				t.Fatal(err)
			}
			trellisCmp(t, "decoder logits (slabs)", lr, ref.Sample.Logits, 1e-4)
			logits, err := dec.Forward(z)
			if err != nil {
				t.Fatal(err)
			}
			trellisCmp(t, "pipeline logits", logits, ref.Sample.Logits, 1e-4)
			coords, err := TrellisCoords(logits)
			if err != nil {
				t.Fatal(err)
			}
			trellisCoordsMatch(t, coords, ref.Sample.Coords, ref.Sample.Logits, logits.Dim(2), 1e-3)
		})
	}
}

// The pos_emb buffer stored in the checkpoint equals TrellisPosEmbed (the
// fallback for checkpoints without it).
func TestTrellisPosEmbed(t *testing.T) {
	hf, _, err := safetensors.LoadFiles([]string{"testdata/trellis-tiny/ckpts/ss_flow_tiny_fp16.safetensors"}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	pe := hf["pos_emb"]
	got := TrellisPosEmbed(4, pe.Dim(1))
	want := tensor.MustSlice[float32](pe)
	md := 0.0
	for i := range got {
		md = math.Max(md, math.Abs(float64(got[i]-want[i])))
	}
	t.Logf("pos_emb: max diff %.3g", md)
	if len(got) != len(want) || md > 1e-6 {
		t.Errorf("pos_emb differs by %.3g", md)
	}
}

// With 16-bit weight storage (the checkpoints' torsos are F16 already; the
// F32 rest rounds), the flow model and decoder stay close to the reference.
func TestTrellisTinyStorage16(t *testing.T) {
	ref := loadTrellisTinyRef(t)
	dev := device.CPU()
	var flow *TrellisFlow
	var dec *TrellisSSDecoder
	if err := nn.WithWeightStorage16(dtype.F16, func() (err error) {
		if flow, err = LoadTrellisFlow("testdata/trellis-tiny/ckpts/ss_flow_tiny_fp16", dev); err != nil {
			return err
		}
		dec, err = LoadTrellisSSDecoder("testdata/trellis-tiny/ckpts/ss_dec_tiny_fp16", dev)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if dt := flow.Blocks[0].FFN.L1.W.DType(); dt != dtype.F16 {
		t.Fatalf("FFN weight stored as %v, want F16", dt)
	}
	a := flow.Args
	R := a.Resolution
	nt := func(v []float32, dims ...int) *tensor.Tensor {
		x, err := tensor.New(v, dims, dev)
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	v, err := flow.Forward(nt(ref.Flow.X, 1, a.InChannels, R, R, R), []float32{ref.Flow.T},
		nt(ref.Flow.Cond, 1, ref.Flow.CondLen, a.CondChannels))
	if err != nil {
		t.Fatal(err)
	}
	trellisCmp(t, "flow velocity (16-bit weights)", v, ref.Flow.V, 1e-2)
	l, err := dec.Forward(nt(ref.Sample.Latent, 1, a.InChannels, R, R, R))
	if err != nil {
		t.Fatal(err)
	}
	trellisCmp(t, "decoder logits (16-bit weights)", l, ref.Sample.Logits, 1e-2)
}
