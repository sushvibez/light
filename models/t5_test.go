package models

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// google/t5-v1_1-small (gated-GELU, like Flux's T5-v1.1-XXL) against
// transformers' T5EncoderModel F32 (scripts/verify/ref_t5.py): every hidden
// state of a 17-token sentence, on the CPU and WebGPU. Skips without the
// cached checkpoint.
func TestT5EncoderMatchesTransformers(t *testing.T) {
	raw, err := os.ReadFile("testdata/refs/t5-v1_1-small.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Repo   string    `json:"repo"`
		IDs    []int64   `json:"ids"`
		DModel int       `json:"d_model"`
		Hidden []float32 `json:"hidden"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		m, err := LoadT5EncoderFromHF(ref.Repo, dev, dtype.F32)
		if err != nil {
			t.Skip(err)
		}
		ids, err := tensor.New(ref.IDs, []int{1, len(ref.IDs)}, dev)
		if err != nil {
			t.Fatal(err)
		}
		h, err := m.Forward(ids)
		if err != nil {
			t.Fatal(err)
		}
		if h, err = h.To(device.CPU()); err != nil {
			t.Fatal(err)
		}
		got := tensor.MustSlice[float32](h)
		if len(got) != len(ref.Hidden) {
			t.Fatalf("%v: got %d values, want %d", dev, len(got), len(ref.Hidden))
		}
		md := 0.0
		for i := range got {
			md = math.Max(md, math.Abs(float64(got[i]-ref.Hidden[i])))
		}
		t.Logf("%v: max |light - transformers| %.3g over %d values", dev, md, len(got))
		if md > 1e-4 {
			t.Errorf("%v: hidden states differ by %.3g", dev, md)
		}
	}
}

// Bucket edges against transformers' _relative_position_bucket (32
// buckets, max distance 128).
func TestT5Bucket(t *testing.T) {
	want := map[int]int{0: 0, 1: 17, -1: 1, 7: 23, -7: 7, 8: 24, -8: 8, 12: 25, 20: 26, -40: 12, 127: 31, -128: 15, 500: 31}
	for rel, w := range want {
		if got := t5Bucket(rel, 32, 128); got != w {
			t.Errorf("bucket(%d) = %d, want %d", rel, got, w)
		}
	}
}
