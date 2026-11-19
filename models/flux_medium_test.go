package models

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/tensor"
)

// FLUX.1's real geometry on a small random model (scripts/verify/
// ref_flux_medium.py: 128-wide heads, rope axes 16 / 56 / 56, 1024 image
// and 256 text tokens, guidance) against diffusers, on every device.
func TestFluxMediumGeometryMatchesDiffusers(t *testing.T) {
	dir := "testdata/flux-medium"
	if d := os.Getenv("LIGHT_FLUX_MEDIUM_DIR"); d != "" { // e.g. a full-width one made with ref_flux_medium.py <dir> 24 0
		dir = d
	}
	raw, err := os.ReadFile(dir + "/forward.json")
	if err != nil {
		t.Skip(err)
	}
	var ref struct {
		Image, Text, Pooled []float32
		Timestep, Guidance  float32
		ImgIDs              []float32 `json:"img_ids"`
		TxtIDs              []float32 `json:"txt_ids"`
		S, St               int
		Out                 []float32
		OutShape            []int `json:"out_shape"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.CUDA(0); err == nil {
		devs = append(devs, g)
	}
	// noaxes: FLUX.1's own config.json has no axes_dims_rope; the default
	// (16, 56, 56) must survive the streamed loader too.
	noaxes := filepath.Join(t.TempDir(), "noaxes")
	os.MkdirAll(noaxes, 0o755)
	os.Symlink(filepath.Join(mustAbs(t, dir), "diffusion_pytorch_model.safetensors"), filepath.Join(noaxes, "diffusion_pytorch_model.safetensors"))
	cfgRaw, _ := os.ReadFile(filepath.Join(dir, "config.json"))
	var cfgMap map[string]any
	json.Unmarshal(cfgRaw, &cfgMap)
	delete(cfgMap, "axes_dims_rope")
	cfgRaw, _ = json.Marshal(cfgMap)
	os.WriteFile(filepath.Join(noaxes, "config.json"), cfgRaw, 0o644)
	loads := map[string]struct {
		s16    bool
		stream bool
		dir    string
		tol    float64
	}{"f32": {false, false, dir, 1e-3}, "bf16": {true, false, dir, 2e-2}, "bf16-streamed": {true, true, dir, 2e-2},
		"noaxes-streamed": {false, true, noaxes, 1e-3}}
	for _, dev := range devs {
		for name, lo := range loads {
			t.Run(dev.String()+"/"+name, func(t *testing.T) {
				var m *FluxTransformer
				load := func() (err error) {
					m, err = LoadFluxTransformerOpts(lo.dir, dev, FluxTransformerLoadOptions{Stream: lo.stream})
					return
				}
				var err error
				if lo.s16 {
					err = nn.WithWeightStorage16(dtype.BF16, load)
				} else {
					err = load()
				}
				if err != nil {
					t.Fatal(err)
				}
				mk := func(v []float32, shape ...int) *tensor.Tensor {
					x, err := tensor.New(v, shape, dev)
					if err != nil {
						t.Fatal(err)
					}
					return x
				}
				y, err := m.Forward(FluxInputs{Image: mk(ref.Image, 1, ref.S, m.Cfg.InChannels), Text: mk(ref.Text, 1, ref.St, m.Cfg.JointDim),
					Pooled: mk(ref.Pooled, 1, m.Cfg.PooledDim), Timestep: []float32{ref.Timestep}, Guidance: []float32{ref.Guidance},
					ImgIDs: ref.ImgIDs, TxtIDs: ref.TxtIDs})
				if err != nil {
					t.Fatal(err)
				}
				c, _ := y.To(device.CPU())
				got := tensor.MustSlice[float32](c)
				if len(got) != len(ref.Out) {
					t.Fatalf("out %v, want %v", y.Shape(), ref.OutShape)
				}
				var sum, md, ma float64
				for i := range got {
					d := math.Abs(float64(got[i] - ref.Out[i]))
					sum += d
					md = math.Max(md, d)
					ma = math.Max(ma, math.Abs(float64(ref.Out[i])))
				}
				t.Logf("mean |Δ| %.3g, max %.3g (reference max |y| %.3g)", sum/float64(len(got)), md, ma)
				if md > lo.tol {
					t.Errorf("differs by %.3g", md)
				}
			})
		}
	}
}

func mustAbs(t *testing.T, p string) string {
	a, err := filepath.Abs(p)
	if err != nil {
		t.Fatal(err)
	}
	return a
}
