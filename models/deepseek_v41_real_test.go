package models

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"light/device"
	"light/tensor"
)

// DeepSeek-V4.1-Flash on real weights, truncated to its first layers
// (scripts/colab/deepseek_v41_check.py writes the folder and the reference:
// DeepSeek's model.py on the dequantized weights in F32): the last
// position's logits and top-10, with 16 routed experts per layer resident
// (LIGHT_EXPERT_SLOTS unless set).
// Runs when LIGHT_V41_TRUNC names that folder.
func TestDeepseekV41TruncatedMatchesReference(t *testing.T) {
	dir := os.Getenv("LIGHT_V41_TRUNC")
	if dir == "" {
		t.Skip("LIGHT_V41_TRUNC not set")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "ref.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		IDs        []int64   `json:"ids"`
		LastLogits []float32 `json:"last_logits"`
		Top10      []int     `json:"top10"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	if os.Getenv("LIGHT_EXPERT_SLOTS") == "" {
		t.Setenv("LIGHT_EXPERT_SLOTS", "16")
	}
	m, err := LoadDeepseekV41(dir, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tensor.New(ref.IDs, []int{1, len(ref.IDs)}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	out, err := m.Forward(ids)
	if err != nil {
		t.Fatal(err)
	}
	all := tensor.MustSlice[float32](out)
	V := len(ref.LastLogits)
	last := all[len(all)-V:]
	md, mr := 0.0, 0.0
	for i, v := range last {
		md = math.Max(md, math.Abs(float64(v-ref.LastLogits[i])))
		mr = math.Max(mr, math.Abs(float64(ref.LastLogits[i])))
		if v != v {
			md = math.Inf(1)
		}
	}
	idx := make([]int, V)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return last[idx[a]] > last[idx[b]] })
	t.Logf("last logits: max diff %.3g (max |ref| %.3g); top10 %v, reference %v", md, mr, idx[:10], ref.Top10)
	if idx[0] != ref.Top10[0] {
		t.Errorf("top-1 %d, reference %d", idx[0], ref.Top10[0])
	}
	if !(md <= 1e-3*mr) { // both F32 on the same dequantized weights
		t.Errorf("last logits differ by %.3g of %.3g", md, mr)
	}
}
