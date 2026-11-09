package models

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// A seeded tiny-random FluxTransformer2DModel (2 double + 2 single blocks,
// guidance embedder, 3-axis RoPE; scripts/verify/ref_flux.py) against
// diffusers' forward pass, on the CPU and WebGPU.
func TestFluxTransformerTinyMatchesDiffusers(t *testing.T) {
	raw, err := os.ReadFile("testdata/refs/flux-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		X        []float32 `json:"x"`
		Txt      []float32 `json:"txt"`
		Pooled   []float32 `json:"pooled"`
		ImgIDs   []float32 `json:"img_ids"`
		TxtIDs   []float32 `json:"txt_ids"`
		Out      []float32 `json:"out"`
		XShape   []int     `json:"x_shape"`
		TxtShape []int     `json:"txt_shape"`
		Timestep []float32 `json:"timestep"`
		Guidance []float32 `json:"guidance"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		m, err := LoadFluxTransformer("testdata/flux-tiny", dev)
		if err != nil {
			t.Fatal(err)
		}
		mk := func(v []float32, dims ...int) *tensor.Tensor {
			x, err := tensor.New(v, dims, dev)
			if err != nil {
				t.Fatal(err)
			}
			return x
		}
		y, err := m.Forward(FluxInputs{
			Image:    mk(ref.X, ref.XShape...),
			Text:     mk(ref.Txt, ref.TxtShape...),
			Pooled:   mk(ref.Pooled, 1, len(ref.Pooled)),
			Timestep: ref.Timestep, Guidance: ref.Guidance,
			ImgIDs: ref.ImgIDs, TxtIDs: ref.TxtIDs,
		})
		if err != nil {
			t.Fatal(err)
		}
		if y, err = y.To(device.CPU()); err != nil {
			t.Fatal(err)
		}
		got := tensor.MustSlice[float32](y)
		if len(got) != len(ref.Out) {
			t.Fatalf("%v: %d values, want %d", dev, len(got), len(ref.Out))
		}
		md := 0.0
		for i := range got {
			md = math.Max(md, math.Abs(float64(got[i]-ref.Out[i])))
		}
		t.Logf("%v: max |light - diffusers| %.3g over %d values", dev, md, len(got))
		if md > 1e-4 {
			t.Errorf("%v: output differs by %.3g", dev, md)
		}
	}
}

// A tiny Flux-style AutoencoderKL decoder (16 latent channels, no
// post-quant conv, 2 blocks) against diffusers' decode, on the CPU.
func TestFluxVAETinyMatchesDiffusers(t *testing.T) {
	raw, err := os.ReadFile("testdata/refs/flux-tiny-vae.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Z      []float32 `json:"z"`
		ZShape []int     `json:"z_shape"`
		Image  []float32 `json:"image"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	cpu := device.CPU()
	v, err := LoadFluxVAE("testdata/flux-tiny-vae", cpu)
	if err != nil {
		t.Fatal(err)
	}
	z, err := tensor.New(ref.Z, ref.ZShape, cpu)
	if err != nil {
		t.Fatal(err)
	}
	img, err := v.Decoder.Decode(z)
	if err != nil {
		t.Fatal(err)
	}
	got := tensor.MustSlice[float32](img)
	if len(got) != len(ref.Image) {
		t.Fatalf("%d values, want %d", len(got), len(ref.Image))
	}
	md := 0.0
	for i := range got {
		md = math.Max(md, math.Abs(float64(got[i]-ref.Image[i])))
	}
	t.Logf("max |light - diffusers| %.3g over %d values", md, len(got))
	if md > 1e-4 {
		t.Errorf("image differs by %.3g", md)
	}
}

// The SD 1.5 decoder built through VAEDecoderConfig has diffusers'
// parameter count (decoder + post_quant_conv of the default AutoencoderKL).
func TestVAEDecoderSD15ParamCount(t *testing.T) {
	v, err := NewVAEDecoder(true, dtype.F32, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if got, want := v.ParamCount(), 49490199; got != want {
		t.Fatalf("ParamCount %d, want %d", got, want)
	}
}

// A seeded tiny-random Flux2Transformer2DModel (klein-style, 2 + 2 blocks,
// 4-axis RoPE; scripts/verify/ref_flux.py --make-tiny-flux2) against
// diffusers' forward, on the CPU and WebGPU.
func TestFlux2TransformerTinyMatchesDiffusers(t *testing.T) {
	raw, err := os.ReadFile("testdata/refs/flux2-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		X        []float32 `json:"x"`
		Txt      []float32 `json:"txt"`
		ImgIDs   []float32 `json:"img_ids"`
		TxtIDs   []float32 `json:"txt_ids"`
		Out      []float32 `json:"out"`
		XShape   []int     `json:"x_shape"`
		TxtShape []int     `json:"txt_shape"`
		Timestep []float32 `json:"timestep"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		m, err := LoadFlux2Transformer("testdata/flux2-tiny", dev)
		if err != nil {
			t.Fatal(err)
		}
		x, _ := tensor.New(ref.X, ref.XShape, dev)
		txt, _ := tensor.New(ref.Txt, ref.TxtShape, dev)
		y, err := m.Forward(Flux2Inputs{Image: x, Text: txt, Timestep: ref.Timestep, ImgIDs: ref.ImgIDs, TxtIDs: ref.TxtIDs})
		if err != nil {
			t.Fatal(err)
		}
		if y, err = y.To(device.CPU()); err != nil {
			t.Fatal(err)
		}
		got := tensor.MustSlice[float32](y)
		md := 0.0
		for i := range got {
			md = math.Max(md, math.Abs(float64(got[i]-ref.Out[i])))
		}
		t.Logf("%v: max |light - diffusers| %.3g over %d values", dev, md, len(got))
		if len(got) != len(ref.Out) || md > 1e-4 {
			t.Errorf("%v: output differs by %.3g", dev, md)
		}
	}
}

// A tiny AutoencoderKL encoder (with quant_conv) against diffusers'
// encode().latent_dist.mode() on an odd-sized image, on the CPU and WebGPU.
func TestVAEEncoderTinyMatchesDiffusers(t *testing.T) {
	raw, err := os.ReadFile("testdata/refs/vae-tiny-enc.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Image       []float32 `json:"image"`
		ImageShape  []int     `json:"image_shape"`
		Latent      []float32 `json:"latent"`
		LatentShape []int     `json:"latent_shape"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		v, err := LoadFluxVAE("testdata/vae-tiny-enc", dev)
		if err != nil {
			t.Fatal(err)
		}
		if v.Encoder == nil {
			t.Fatal("no encoder loaded")
		}
		x, _ := tensor.New(ref.Image, ref.ImageShape, dev)
		z, err := v.Encoder.Mode(x)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(z.Shape().Dims(), ref.LatentShape) {
			t.Fatalf("latent %v, want %v", z.Shape(), ref.LatentShape)
		}
		if z, err = z.To(device.CPU()); err != nil {
			t.Fatal(err)
		}
		got := tensor.MustSlice[float32](z)
		md := 0.0
		for i := range got {
			md = math.Max(md, math.Abs(float64(got[i]-ref.Latent[i])))
		}
		t.Logf("%v: max |light - diffusers| %.3g over %d values", dev, md, len(got))
		if md > 1e-4 {
			t.Errorf("%v: latent differs by %.3g", dev, md)
		}
	}
}
