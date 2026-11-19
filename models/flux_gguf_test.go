package models

import (
	"bytes"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
	"light/gguf"
	"light/quant"
	"light/safetensors"
	"light/tensor"
)

// The tiny transformer as a GGUF (BFL names, 2-D weights Q8_0 and Q4_0,
// biases and norms F32) runs like the same weights dequantized into a
// safetensors file: the quantized loader, the byte-level fused splits and
// nn.Linear's quantized matmul, resident and streamed, CPU and CUDA.
func TestFluxGGUFMatchesDequantized(t *testing.T) {
	f, err := os.Open("testdata/flux-tiny-bfl.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	src, _, err := safetensors.Load(f, device.CPU())
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	w := gguf.NewWriter()
	w.AddString("general.architecture", "flux")
	deq := map[string]*tensor.Tensor{}
	i := 0
	for name, x := range src {
		q := x
		if x.Rank() == 2 && x.Dim(1)%32 == 0 {
			to := dtype.Q8_0
			if i%2 == 1 {
				to = dtype.Q4_0
			}
			i++
			if q, err = quant.Quantize(x, to); err != nil {
				t.Fatal(err)
			}
			d, err := quant.Dequantize(q)
			if err != nil {
				t.Fatal(err)
			}
			deq[name] = d
		} else {
			deq[name] = x
		}
		dims := make([]int, x.Rank()) // ggml order: innermost first
		for j, d := range x.Shape().Dims() {
			dims[len(dims)-1-j] = d
		}
		gt, err := gguf.GGMLType(q.DType())
		if err != nil {
			t.Fatal(err)
		}
		if err := w.AddTensor(name, dims, gt, q.Storage().CPUBytes()); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := w.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	gpath := filepath.Join(dir, "flux-tiny.gguf")
	os.WriteFile(gpath, raw, 0o644)
	os.WriteFile(gpath+".config.json", mustReadFile(t, "testdata/flux-tiny-bfl.safetensors.config.json"), 0o644)
	spath := filepath.Join(dir, "flux-tiny-deq.safetensors")
	var buf bytes.Buffer
	if err := safetensors.Write(&buf, deq, nil); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(spath, buf.Bytes(), 0o644)
	os.WriteFile(spath+".config.json", mustReadFile(t, "testdata/flux-tiny-bfl.safetensors.config.json"), 0o644)

	devs := []device.Device{device.CPU()}
	if g, err := device.CUDA(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		for _, stream := range []bool{false, true} {
			ref, err := LoadFluxTransformer(spath, dev)
			if err != nil {
				t.Fatal(err)
			}
			got, err := LoadFluxTransformerOpts(gpath, dev, FluxTransformerLoadOptions{Stream: stream})
			if err != nil {
				t.Fatal(err)
			}
			if !stream {
				if wq := got.Double[0].Img.Q.W; !wq.DType().IsQuantized() {
					t.Errorf("gguf weight loaded as %v", wq.DType())
				}
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
				x, _ := tensor.New(v, shape, dev)
				return x
			}
			S, St := 12, 5
			ids := make([]float32, S*3)
			for i := range S {
				ids[i*3+1], ids[i*3+2] = float32(i/4), float32(i%4)
			}
			in := FluxInputs{Image: rnd(1, S, cfg.InChannels), Text: rnd(1, St, cfg.JointDim), Pooled: rnd(1, cfg.PooledDim),
				Timestep: []float32{0.7}, ImgIDs: ids, TxtIDs: make([]float32, St*3)}
			a, err := ref.Forward(in)
			if err != nil {
				t.Fatal(err)
			}
			b, err := got.Forward(in)
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
			t.Logf("%s streamed=%v: gguf vs dequantized max diff %.3g (max |y| %.3g)", dev, stream, md, ma)
			if md > 1e-4 {
				t.Errorf("%s streamed=%v: differ by %.3g", dev, stream, md)
			}
		}
	}
}

func mustReadFile(t *testing.T, p string) []byte {
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
