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

// A seeded tiny-random ZImageTransformer2DModel (caption and image both
// padded to 32 with the learned pad tokens; scripts/verify/ref_flux.py
// --make-tiny-zimage) against diffusers' forward, on the CPU and WebGPU.
func TestZImageTransformerTinyMatchesDiffusers(t *testing.T) {
	raw, err := os.ReadFile("testdata/refs/zimage-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Latents []float32 `json:"latents"`
		LShape  []int     `json:"latents_shape"`
		Caption []float32 `json:"caption"`
		CShape  []int     `json:"caption_shape"`
		T       float32   `json:"t"`
		Out     []float32 `json:"out"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		m, err := LoadZImageTransformer("testdata/zimage-tiny", dev)
		if err != nil {
			t.Fatal(err)
		}
		lat, _ := tensor.New(ref.Latents, ref.LShape, dev)
		cp, _ := tensor.New(ref.Caption, ref.CShape, dev)
		y, err := m.Forward(lat, cp, ref.T)
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

// A tiny flow-map Flux3Transformer2DModel (scripts/verify/ref_flux3.py)
// against diffusers: text-to-image and editing (reference image tokens),
// with a timestep and a target timestep, on the CPU and WebGPU.
func TestFlux3TransformerTinyMatchesDiffusers(t *testing.T) {
	raw, err := os.ReadFile("testdata/flux3-tiny/ref.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Img, Txt, Cond   []float32
		ImgIDs           []float32 `json:"img_ids"`
		TxtIDs           []float32 `json:"txt_ids"`
		CondIDs          []float32 `json:"cond_ids"`
		Timestep, Target float32
		Si, St, Sc       int
		T2I              []float32 `json:"t2i"`
		Edit             []float32 `json:"edit"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		t.Run(dev.String(), func(t *testing.T) {
			m, err := LoadFlux3Transformer("testdata/flux3-tiny/transformer", dev)
			if err != nil {
				t.Fatal(err)
			}
			img, _ := tensor.New(ref.Img, []int{1, ref.Si, 8}, dev)
			txt, _ := tensor.New(ref.Txt, []int{1, ref.St, 24}, dev)
			cond, _ := tensor.New(ref.Cond, []int{1, ref.Sc, 8}, dev)
			in := Flux3Inputs{Image: img, Text: txt, Timestep: []float32{ref.Timestep},
				TargetTimestep: []float32{ref.Target}, ImgIDs: ref.ImgIDs, TxtIDs: ref.TxtIDs}
			for _, c := range []struct {
				name string
				cond bool
				want []float32
			}{{"text-to-image", false, ref.T2I}, {"edit", true, ref.Edit}} {
				in.Cond, in.CondIDs = nil, nil
				if c.cond {
					in.Cond, in.CondIDs = cond, ref.CondIDs
				}
				out, err := m.Forward(in)
				if err != nil {
					t.Fatal(err)
				}
				if out, err = out.To(device.CPU()); err != nil {
					t.Fatal(err)
				}
				got := tensor.MustSlice[float32](out)
				md := 0.0
				for i := range c.want {
					md = math.Max(md, math.Abs(float64(got[i]-c.want[i])))
				}
				t.Logf("%s: max diff %.3g over %d values", c.name, md, len(got))
				if len(got) != len(c.want) || md > 1e-4 {
					t.Errorf("%s differs by %.3g", c.name, md)
				}
			}
		})
	}
}

// A tiny QwenImage21Transformer2DModel (scripts/verify/ref_qwenimage21.py)
// against diffusers: text-to-image with right-padded text (block-causal
// mask, 3-axis RoPE with a zero-centred image grid, t = 0 text modulation),
// on the CPU and WebGPU.
func TestQwenImage21TransformerTinyMatchesDiffusers(t *testing.T) {
	raw, err := os.ReadFile("testdata/qwenimage21-tiny/ref.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Txt, Img, Target []float32
		L, Valid, H, W   int
		Timestep         float32
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		t.Run(dev.String(), func(t *testing.T) {
			m, err := LoadQwenImage21Transformer("testdata/qwenimage21-tiny/transformer", dev)
			if err != nil {
				t.Fatal(err)
			}
			txt, _ := tensor.New(ref.Txt, []int{1, ref.L, 24}, dev)
			img, _ := tensor.New(ref.Img, []int{1, ref.H * ref.W, 8}, dev)
			out, err := m.Forward(QwenImage21Inputs{Image: img, Text: txt, TextLen: ref.Valid, H: ref.H, W: ref.W, Timestep: ref.Timestep})
			if err != nil {
				t.Fatal(err)
			}
			if out, err = out.To(device.CPU()); err != nil {
				t.Fatal(err)
			}
			got := tensor.MustSlice[float32](out)
			md := 0.0
			for i := range ref.Target {
				md = math.Max(md, math.Abs(float64(got[i]-ref.Target[i])))
			}
			t.Logf("target: max diff %.3g over %d values", md, len(got))
			if len(got) != len(ref.Target) || md > 1e-4 {
				t.Errorf("target differs by %.3g", md)
			}
		})
	}
}

// A tiny AutoencoderKLQwenImage21 (scripts/verify/ref_qwenimage_vae.py)
// decoding one frame against diffusers, on the CPU and WebGPU.
func TestQwenImageVAETinyMatchesDiffusers(t *testing.T) {
	raw, err := os.ReadFile("testdata/qwenimage21-tiny/vae.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Latents, Image []float32
		Z, H, W        int
		ImageShape     []int `json:"image_shape"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		t.Run(dev.String(), func(t *testing.T) {
			v, err := LoadQwenImageVAE("testdata/qwenimage21-tiny/vae", dev)
			if err != nil {
				t.Fatal(err)
			}
			z, _ := tensor.New(ref.Latents, []int{1, ref.Z, ref.H, ref.W}, dev)
			img, err := v.Decode(z)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(img.Shape().Dims(), ref.ImageShape) {
				t.Fatalf("image %v, want %v", img.Shape(), ref.ImageShape)
			}
			if img, err = img.To(device.CPU()); err != nil {
				t.Fatal(err)
			}
			got := tensor.MustSlice[float32](img)
			md := 0.0
			for i := range ref.Image {
				md = math.Max(md, math.Abs(float64(got[i]-ref.Image[i])))
			}
			t.Logf("image: max diff %.3g over %d values", md, len(got))
			if md > 1e-4 {
				t.Errorf("image differs by %.3g", md)
			}
		})
	}
}
