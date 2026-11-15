package models

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"light/device"
	"light/tensor"
)

// A tiny Kimi K3.1 (scripts/verify/ref_kimi_k31.py: a float64 port of vLLM's
// kimi_k3 forward; KDA and gated DSA layers, Attention Residuals over blocks
// of 2 layers, a dense first layer then latent MoE layers with grouped
// routing and shared experts, SiTU-GLU) in the released naming and storage
// (BF16, F32 KDA state parameters, MXFP4 routed experts): every position's
// logits for a (2, 12) prompt, long enough for the DSA top-k (5) to drop keys.
func TestKimiK31TinyMatchesReference(t *testing.T) {
	raw, err := os.ReadFile("testdata/kimi-k31-tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		IDs    [][]int64
		Logits []float32
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	B, S := len(ref.IDs), len(ref.IDs[0])
	var ids []int64
	for _, r := range ref.IDs {
		ids = append(ids, r...)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		t.Run(fmt.Sprint(dev), func(t *testing.T) {
			m, err := LoadKimiK31("testdata/kimi-k31-tiny", dev)
			if err != nil {
				t.Fatal(err)
			}
			tok, err := tensor.New(ids, []int{B, S}, dev)
			if err != nil {
				t.Fatal(err)
			}
			out, err := m.Forward(tok)
			if err != nil {
				t.Fatal(err)
			}
			c, _ := out.To(device.CPU())
			got := tensor.MustSlice[float32](c)
			if len(got) != len(ref.Logits) {
				t.Fatalf("%d logits, want %d", len(got), len(ref.Logits))
			}
			md := 0.0
			for i := range got {
				md = math.Max(md, math.Abs(float64(got[i]-ref.Logits[i])))
			}
			t.Logf("logits: max diff %.3g over %d", md, len(got))
			if !(md <= 1e-4) { // NaN fails too
				t.Errorf("logits differ by %.3g", md)
			}
		})
	}
}
