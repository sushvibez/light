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

// GLM-5.3-Flash on real weights, truncated to its first layers
// (scripts/colab/glm5_check.py writes the folder and transformers'
// reference with the FP8 weights dequantized to BF16): the last position's
// logits and top-10. Runs when LIGHT_GLM5_TRUNC names that folder.
func TestGLM5FlashTruncatedMatchesTransformers(t *testing.T) {
	dir := os.Getenv("LIGHT_GLM5_TRUNC")
	if dir == "" {
		t.Skip("LIGHT_GLM5_TRUNC not set")
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
	m, err := LoadGLM5(dir, device.CPU())
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
	if md > 0.05*mr { // BF16-dequantized reference vs Light's exact FP8 · scale in F32
		t.Errorf("last logits differ by %.3g of %.3g", md, mr)
	}
}
