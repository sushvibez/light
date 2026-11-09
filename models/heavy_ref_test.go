package models

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"light/device"
)

// ggufSearchDir is where quantized checkpoints live (engine.Catalog scans
// the same place).
func ggufSearchDir() string {
	if d := os.Getenv("LIGHT_GGUF_DIR"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache/light/gguf")
}

// quantTolerance is the default |logit − ref| budget by GGUF quant level
// when the reference file does not set one: BF16 references vs Q8 / Q6 /
// Q5 / Q4 / Q3 / Q2 weights.
func quantTolerance(file string) float64 {
	f := strings.ToLower(file)
	switch {
	case strings.Contains(f, "q8"), strings.Contains(f, "f16"), strings.Contains(f, "bf16"):
		return 0.5
	case strings.Contains(f, "q6"), strings.Contains(f, "q5"):
		return 1.0
	case strings.Contains(f, "q4"):
		return 2.0
	default:
		return 3.0
	}
}

// TestGGUFAgainstReferenceJSON verifies every quantized checkpoint on disk
// that has a reference under testdata/refs (written by
// scripts/verify/ref_json.py, on Colab for the big ones): the reference's
// top-1 must be Light's top-1 with its logit within the tolerance, the
// other reference top-k logits within twice that, and the greedy
// continuation must agree on its first tokens (quantization noise may flip later near-ties). Skips
// references whose GGUF is not on disk; LIGHT_HEAVY_TESTS=1 is required for
// files over 4 GB.
func TestGGUFAgainstReferenceJSON(t *testing.T) {
	refs, err := LoadRefJSONs("testdata/refs")
	if err != nil {
		t.Fatal(err)
	}
	if len(refs) == 0 {
		t.Skip("no references in testdata/refs")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	ran := 0
	for _, ref := range refs {
		if ref.GGUF == "" {
			continue
		}
		files, _ := filepath.Glob(filepath.Join(ggufSearchDir(), ref.GGUF))
		for _, file := range files {
			name := filepath.Base(ref.Path) + "/" + filepath.Base(file)
			t.Run(name, func(t *testing.T) {
				st, _ := os.Stat(file)
				if st != nil && st.Size() > 4<<30 && os.Getenv("LIGHT_HEAVY_TESTS") == "" {
					t.Skip("set LIGHT_HEAVY_TESTS=1 (GGUF over 4 GB)")
				}
				releaseModelMemory(t)
				checkGGUFAgainstRef(t, ref, file, dev)
			})
			ran++
		}
	}
	if ran == 0 {
		t.Skip("none of the referenced GGUF files is in " + ggufSearchDir())
	}
}

func checkGGUFAgainstRef(t *testing.T, ref *RefJSON, file string, dev device.Device) {
	m, err := LoadLlamaGGUF(file, dev)
	if err != nil {
		t.Fatal(err)
	}
	tol := ref.Tolerance
	if tol == 0 {
		tol = quantTolerance(file)
	}
	cache := NewKVCache(m.Cfg.NumLayers)
	row := lastLogits(t, m, ref.IDs, cache, dev)
	got := top5(row)
	t.Logf("%s (%s %s) vs %s: top-5 %v, ref %d %d %d %d %d; tol %.1f", ref.Repo, ref.DType, ref.Source.Device, filepath.Base(file),
		got, ref.Top[0].ID, ref.Top[1].ID, ref.Top[2].ID, ref.Top[3].ID, ref.Top[4].ID, tol)
	if got[0] != ref.Top[0].ID {
		t.Errorf("top-1 = %d (%.3f), reference %d (%.3f)", got[0], row[got[0]], ref.Top[0].ID, ref.Top[0].Logit)
	}
	// Rank 1 must sit within the tolerance; lower ranks get twice that,
	// since quantization noise spreads the tail of the distribution more
	// than its mode (Q4 on a 1B model moves a rank-4 logit by ~2).
	for i, w := range ref.Top {
		budget := tol
		if i > 0 {
			budget = 2 * tol
		}
		if d := math.Abs(float64(row[w.ID] - w.Logit)); d > budget {
			t.Errorf("logit[%d %q] = %.3f, reference %.3f (tolerance %.1f)", w.ID, w.Text, row[w.ID], w.Logit, budget)
		}
	}
	// Greedy: the first min(4, len) tokens must agree; log the rest.
	var out []int64
	step := row
	for len(out) < len(ref.Greedy) {
		next := int64(top5(step)[0])
		out = append(out, next)
		if next == ref.Greedy[len(out)-1] && len(out) < len(ref.Greedy) {
			step = lastLogits(t, m, []int64{next}, cache, dev)
			continue
		}
		break
	}
	agree := 0
	for i := range out {
		if i < len(ref.Greedy) && out[i] == ref.Greedy[i] {
			agree++
		} else {
			break
		}
	}
	t.Logf("greedy agrees on %d/%d tokens (%q)", agree, len(ref.Greedy), ref.GreedyText)
	if agree < min(4, len(ref.Greedy)) {
		t.Errorf("greedy diverged after %d tokens: got %v, reference %v", agree, out, ref.Greedy)
	}
}
