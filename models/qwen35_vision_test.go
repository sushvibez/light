package models

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

func readF32File(t *testing.T, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

// Qwen3.5-0.8B's vision tower against transformers' model.visual in F32
// on the processor's patches of a synthetic 320 x 256 image
// (scripts/verify/ref_qwen35_vision.py), on the CPU and WebGPU.
func TestQwen35VisionMatchesTransformers(t *testing.T) {
	const dir = "testdata/qwen35-vision/"
	raw, err := os.ReadFile(dir + "meta.json")
	if err != nil {
		t.Fatal(err)
	}
	var meta struct {
		Repo   string   `json:"repo"`
		PVS    []int    `json:"pixel_values_shape"`
		Grid   [][3]int `json:"grid_thw"`
		Merged []int    `json:"merged_shape"`
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	pv, want := readF32File(t, dir+"pixel_values.f32"), readF32File(t, dir+"merged.f32")
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		m, err := LoadQwen35VisionFromHF(meta.Repo, dev)
		if err != nil {
			t.Skip(err)
		}
		x, err := tensor.New(pv, meta.PVS, dev)
		if err != nil {
			t.Fatal(err)
		}
		y, err := m.Forward(x, meta.Grid)
		if err != nil {
			t.Fatal(err)
		}
		if y, err = y.To(device.CPU()); err != nil {
			t.Fatal(err)
		}
		got := tensor.MustSlice[float32](y)
		if len(got) != len(want) {
			t.Fatalf("%v: %d values, want %d", dev, len(got), len(want))
		}
		md, ma := 0.0, 0.0
		for i := range got {
			md = math.Max(md, math.Abs(float64(got[i]-want[i])))
			ma = math.Max(ma, math.Abs(float64(want[i])))
		}
		t.Logf("%v: max |light - transformers| %.3g (max |ref| %.3g) over %d values", dev, md, ma, len(got))
		// The residual stream reaches ~2100 in the last blocks (ViT massive
		// activations), where F32 summation order alone moves values by
		// ~1e-4 relative; block 0 agrees to 1.3e-5 and the merged tokens
		// (|ref| <= 3.8) to ~1e-3.
		if md > 2e-3 {
			t.Errorf("%v: merged tokens differ by %.3g", dev, md)
		}
	}
}

// Image + text through Qwen3.5-0.8B (vision tower, placeholder scatter,
// interleaved MRoPE) against transformers F32 (scripts/verify/ref_qwen35_vl.py):
// MRoPE position ids equal, prefill top-5, and 16 greedy tokens.
func TestQwen35ImageChatMatchesTransformers(t *testing.T) {
	const dir = "testdata/qwen35-vision/"
	raw, err := os.ReadFile(dir + "vl.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Repo     string     `json:"repo"`
		IDs      []int64    `json:"ids"`
		Grid     [][3]int   `json:"grid_thw"`
		PVS      []int      `json:"pixel_values_shape"`
		Pos      [3][]int64 `json:"position_ids"`
		Delta    int64      `json:"delta"`
		Top5     []int      `json:"top5_ids"`
		Top5Lgt  []float64  `json:"top5_logits"`
		Greedy   []int64    `json:"greedy"`
		GreedyTx string     `json:"greedy_text"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	cpu := device.CPU()
	lm, err := LoadQwen35FromHF(ref.Repo, cpu, dtype.F32)
	if err != nil {
		t.Skip(err)
	}
	vis, err := LoadQwen35VisionFromHF(ref.Repo, cpu)
	if err != nil {
		t.Fatal(err)
	}
	pos, delta, err := Qwen35RopeIndex(ref.IDs, lm.Cfg.ImageTokenID, ref.Grid, vis.Cfg.SpatialMergeSize)
	if err != nil {
		t.Fatal(err)
	}
	for s := range 3 {
		if !slices.Equal(pos[s], ref.Pos[s]) {
			t.Fatalf("MRoPE stream %d differs:\n got %v\nwant %v", s, pos[s], ref.Pos[s])
		}
	}
	if delta != ref.Delta {
		t.Fatalf("rope delta %d, want %d", delta, ref.Delta)
	}
	px, err := tensor.New(readF32File(t, dir+"pixel_values.f32"), ref.PVS, cpu)
	if err != nil {
		t.Fatal(err)
	}
	img, err := vis.Forward(px, ref.Grid)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tensor.New(ref.IDs, []int{1, len(ref.IDs)}, cpu)
	if err != nil {
		t.Fatal(err)
	}
	cache := NewKVCache(lm.NumLayers())
	logits, err := lm.ForwardMultimodal(ids, img, pos, cache)
	if err != nil {
		t.Fatal(err)
	}
	last := lastLogitsRow(t, logits)
	got := top5(last)
	md := 0.0
	for i, id := range ref.Top5 {
		md = math.Max(md, math.Abs(float64(last[id])-ref.Top5Lgt[i]))
	}
	t.Logf("top-5 %v (ref %v), logits within %.3g", got, ref.Top5, md)
	if got[0] != ref.Top5[0] || md > 1e-2 {
		t.Fatalf("prefill differs")
	}
	var gen []int64
	next := int64(got[0])
	for range len(ref.Greedy) {
		gen = append(gen, next)
		p := int64(cache.Len) + delta
		tok, _ := tensor.New([]int64{next}, []int{1, 1}, cpu)
		lg, err := lm.ForwardMultimodal(tok, nil, [3][]int64{{p}, {p}, {p}}, cache)
		if err != nil {
			t.Fatal(err)
		}
		next = int64(top5(lastLogitsRow(t, lg))[0])
	}
	t.Logf("greedy %v\n   ref %v (%q)", gen, ref.Greedy, ref.GreedyTx)
	if !slices.Equal(gen, ref.Greedy) {
		t.Errorf("greedy reply differs")
	}
}

// lastLogitsRow is the last position's logits of (1, T, V) as float32s.
func lastLogitsRow(t *testing.T, logits *tensor.Tensor) []float32 {
	t.Helper()
	l, err := logits.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	all := tensor.MustSlice[float32](l)
	V := l.Dim(-1)
	return all[len(all)-V:]
}
