package models

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"light/device"
	"light/npy"
	"light/tensor"
)

// A tiny BiRefNet (scripts/verify/ref_birefnet.py: idol-sdk's module with
// Swin-L shrunk to 16 wide, window 4, two blocks per stage): the sigmoid
// map for a 64 x 128 input, and remove_background's uint8 alpha for a
// 37 x 50 image (resized to 64 x 128 and back).
func TestBiRefNetTinyMatchesIdolSDK(t *testing.T) {
	raw, err := os.ReadFile("testdata/birefnet-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Input      []float32
		InputShape []int `json:"input_shape"`
		Forward    []float32
		RGBRaw     []int `json:"rgb"`
		RGBShape   []int `json:"rgb_shape"`
		Alpha      []int
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		t.Run(fmt.Sprint(dev), func(t *testing.T) {
			m, err := LoadBiRefNet("testdata/birefnet-tiny/birefnet.safetensors", dev)
			if err != nil {
				t.Fatal(err)
			}
			x, err := tensor.New(ref.Input, ref.InputShape, dev)
			if err != nil {
				t.Fatal(err)
			}
			y, err := m.Forward(x)
			if err != nil {
				t.Fatal(err)
			}
			yc, err := y.To(device.CPU())
			if err != nil {
				t.Fatal(err)
			}
			got := tensor.MustSlice[float32](yc)
			if len(got) != len(ref.Forward) {
				t.Fatalf("forward: %d values, want %d", len(got), len(ref.Forward))
			}
			md, sd := 0.0, 0.0
			for i := range got {
				d := math.Abs(float64(got[i] - ref.Forward[i]))
				md, sd = math.Max(md, d), sd+d
			}
			t.Logf("forward: max diff %.3g, mean %.3g over %d", md, sd/float64(len(got)), len(got))
			if md > 1e-4 {
				t.Errorf("forward differs by %.3g", md)
			}

			h, w := ref.RGBShape[0], ref.RGBShape[1]
			rgb := make([]float32, 3*h*w)
			for i := range h * w {
				for c := range 3 {
					rgb[c*h*w+i] = float32(ref.RGBRaw[i*3+c]) / 255
				}
			}
			m.InputSize = [2]int{64, 128}
			a, err := m.RemoveBackground(rgb, w, h)
			if err != nil {
				t.Fatal(err)
			}
			off := 0
			for i := range a {
				if d := int(a[i]) - ref.Alpha[i]; d != 0 {
					off++
					if d > 1 || d < -1 {
						t.Errorf("alpha[%d] = %d, want %d", i, a[i], ref.Alpha[i])
					}
				}
			}
			t.Logf("remove_background: %d / %d alpha values off by one (truncation at a boundary)", off, len(a))
			if off > len(a)/100 {
				t.Errorf("%d alpha values differ", off)
			}
		})
	}
}

// Real weights against idol-sdk on CPU: LIGHT_BIREFNET_WEIGHTS is
// speridlabs/idol's birefnet.safetensors and LIGHT_BIREFNET_REF a folder of
// <name>_input.npy / _map.npy / _rgb.png / _alpha.png written by idol-sdk's
// BiRefNet (the normalised 1024² input, its sigmoid map, the short-side-1024
// image and remove_background's alpha).
func TestBiRefNetRealMatchesIdolSDK(t *testing.T) {
	wpath, dir := os.Getenv("LIGHT_BIREFNET_WEIGHTS"), os.Getenv("LIGHT_BIREFNET_REF")
	if wpath == "" || dir == "" {
		t.Skip("set LIGHT_BIREFNET_WEIGHTS and LIGHT_BIREFNET_REF")
	}
	names, err := filepath.Glob(filepath.Join(dir, "*_input.npy"))
	if err != nil || len(names) == 0 {
		t.Fatalf("no *_input.npy in %s", dir)
	}
	t0 := time.Now()
	m, err := LoadBiRefNet(wpath, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("loaded in %.1f s", time.Since(t0).Seconds())
	for _, in := range names {
		name := strings.TrimSuffix(in, "_input.npy")
		x, err := npy.ReadFile(in, device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		y, err := m.Forward(x)
		if err != nil {
			t.Fatal(err)
		}
		dt := time.Since(t0)
		want, err := npy.ReadFile(name+"_map.npy", device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		got, wv := tensor.MustSlice[float32](y), tensor.MustSlice[float32](want)
		md, sd := 0.0, 0.0
		for i := range got {
			d := math.Abs(float64(got[i] - wv[i]))
			md, sd = math.Max(md, d), sd+d
		}
		t.Logf("%s: forward %.1f s, map max diff %.3g, mean %.3g over %d", filepath.Base(name), dt.Seconds(), md, sd/float64(len(got)), len(got))
		if md > 1e-2 || sd/float64(len(got)) > 1e-4 {
			t.Errorf("%s: map differs (max %.3g, mean %.3g)", name, md, sd/float64(len(got)))
		}

		img := readPNG(t, name+"_rgb.png")
		ref := readPNG(t, name+"_alpha.png")
		b := img.Bounds()
		w, h := b.Dx(), b.Dy()
		rgb := make([]float32, 3*w*h)
		for yy := range h {
			for xx := range w {
				c := color.NRGBAModel.Convert(img.At(b.Min.X+xx, b.Min.Y+yy)).(color.NRGBA)
				i := yy*w + xx
				rgb[i], rgb[w*h+i], rgb[2*w*h+i] = float32(c.R)/255, float32(c.G)/255, float32(c.B)/255
			}
		}
		t0 = time.Now()
		a, err := m.RemoveBackground(rgb, w, h)
		if err != nil {
			t.Fatal(err)
		}
		dt = time.Since(t0)
		off, maxd := 0, 0
		for yy := range h {
			for xx := range w {
				r, _, _, _ := ref.At(b.Min.X+xx, b.Min.Y+yy).RGBA()
				d := int(a[yy*w+xx]) - int(r>>8)
				if d != 0 {
					off++
					maxd = max(maxd, d, -d)
				}
			}
		}
		t.Logf("%s: remove_background %.1f s, %d x %d alpha: %d values differ, max by %d", filepath.Base(name), dt.Seconds(), w, h, off, maxd)
		if maxd > 3 {
			t.Errorf("%s: alpha differs by up to %d", name, maxd)
		}
	}
}

func readPNG(t *testing.T, path string) image.Image {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}
