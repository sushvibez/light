package models

import (
	"bytes"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/safetensors"
	"light/tensor"
)

// The FLUX.2 transformer's streamed blocks (F32 and 16-bit storage, one
// and two slots) and its FP8 path (weights kept at 8 bits) compute what
// the resident F32 model computes, on every device.
func TestFlux2StreamedAndFP8MatchResident(t *testing.T) {
	src := "../diffusion/testdata/flux2-tiny-pipe/transformer"
	// An FP8 copy: 2-D weights cast to F8E4M3 (exact when widened back).
	f, err := os.Open(filepath.Join(src, "diffusion_pytorch_model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	ts, _, err := safetensors.Load(f, device.CPU())
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	fp8 := map[string]*tensor.Tensor{}
	for k, v := range ts {
		if v.Rank() == 2 {
			if v, err = ops.Cast(v, dtype.F8E4M3); err != nil {
				t.Fatal(err)
			}
		}
		fp8[k] = v
	}
	fdir := filepath.Join(t.TempDir(), "fp8")
	os.MkdirAll(fdir, 0o755)
	var buf bytes.Buffer
	if err := safetensors.Write(&buf, fp8, nil); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(fdir, "diffusion_pytorch_model.safetensors"), buf.Bytes(), 0o644)
	cfg, _ := os.ReadFile(filepath.Join(src, "config.json"))
	os.WriteFile(filepath.Join(fdir, "config.json"), cfg, 0o644)

	devs := []device.Device{device.CPU()}
	if g, err := device.CUDA(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		ref, err := LoadFlux2Transformer(src, dev)
		if err != nil {
			t.Fatal(err)
		}
		c := ref.Cfg
		rng := rand.New(rand.NewPCG(5, 6))
		rnd := func(shape ...int) *tensor.Tensor {
			n := 1
			for _, d := range shape {
				n *= d
			}
			v := make([]float32, n)
			for i := range v {
				v[i] = float32(rng.NormFloat64())
			}
			x, _ := tensor.New(v, shape, dev)
			return x
		}
		S, St, A := 24, 6, len(c.AxesDims)
		ids := make([]float32, S*A)
		for i := range S {
			ids[i*A+1], ids[i*A+2] = float32(i/6), float32(i%6)
		}
		in := Flux2Inputs{Image: rnd(1, S, c.InChannels), Text: rnd(1, St, c.JointDim), Timestep: []float32{0.6},
			ImgIDs: ids, TxtIDs: make([]float32, St*A)}
		if c.GuidanceEmbeds {
			in.Guidance = []float32{4}
		}
		want, err := ref.Forward(in)
		if err != nil {
			t.Fatal(err)
		}
		wc, _ := want.To(device.CPU())
		wv := tensor.MustSlice[float32](wc)
		check := func(name string, m *Flux2Transformer, tol float64) {
			for pass := range 2 {
				got, err := m.Forward(in)
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				gc, _ := got.To(device.CPU())
				gv := tensor.MustSlice[float32](gc)
				md := 0.0
				for i := range gv {
					md = math.Max(md, math.Abs(float64(gv[i]-wv[i])))
				}
				t.Logf("%s %s pass %d: max diff %.3g", dev, name, pass, md)
				if md > tol {
					t.Errorf("%s %s: differ by %.3g", dev, name, md)
				}
			}
		}
		for _, slots := range []int{1, 2} {
			m, err := LoadFlux2TransformerOpts(src, dev, Flux2TransformerLoadOptions{Stream: true, Slots: slots})
			if err != nil {
				t.Fatal(err)
			}
			check(fmt.Sprintf("streamed f32 slots=%d", slots), m, 1e-6)
		}
		var res16, str16 *Flux2Transformer
		err = nn.WithWeightStorage16(dtype.BF16, func() (err error) {
			if res16, err = LoadFlux2TransformerOpts(src, dev, Flux2TransformerLoadOptions{}); err != nil {
				return
			}
			str16, err = LoadFlux2TransformerOpts(src, dev, Flux2TransformerLoadOptions{Stream: true, Slots: 1})
			return
		})
		if err != nil {
			t.Fatal(err)
		}
		r16, _ := res16.Forward(in)
		r16c, _ := r16.To(device.CPU())
		wv16 := tensor.MustSlice[float32](r16c)
		got16, _ := str16.Forward(in)
		g16c, _ := got16.To(device.CPU())
		md := 0.0
		for i, g := range tensor.MustSlice[float32](g16c) {
			md = math.Max(md, math.Abs(float64(g-wv16[i])))
		}
		t.Logf("%s streamed bf16 vs resident bf16: max diff %.3g", dev, md)
		if md > 1e-6 {
			t.Errorf("%s bf16 streamed differs by %.3g", dev, md)
		}
		for _, stream := range []bool{false, true} {
			m, err := LoadFlux2TransformerOpts(fdir, dev, Flux2TransformerLoadOptions{KeepFP8: true, Stream: stream})
			if err != nil {
				t.Fatal(err)
			}
			if !stream {
				if w := m.Double[0].Img.Q.W; w.DType() != dtype.F8E4M3 {
					t.Errorf("fp8 weight stored as %v", w.DType())
				}
			}
			widened, err := LoadFlux2Transformer(fdir, dev)
			if err != nil {
				t.Fatal(err)
			}
			a, _ := m.Forward(in)
			b, _ := widened.Forward(in)
			ac, _ := a.To(device.CPU())
			bc, _ := b.To(device.CPU())
			bv := tensor.MustSlice[float32](bc)
			md := 0.0
			for i, g := range tensor.MustSlice[float32](ac) {
				md = math.Max(md, math.Abs(float64(g-bv[i])))
			}
			t.Logf("%s fp8 kept (streamed=%v) vs widened at load: max diff %.3g", dev, stream, md)
			if md > 1e-6 {
				t.Errorf("%s fp8 streamed=%v differs by %.3g", dev, stream, md)
			}
		}
	}
}
