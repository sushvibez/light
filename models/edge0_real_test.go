package models

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	"strings"
	"testing"
	"time"

	"light/device"
	"light/dtype"
	"light/safetensors"
	"light/tensor"
)

// Edge0/Edge0-35B-A3B-preview on its real weights against mlx-lm
// (scripts/colab/edge0_check.py writes the reference: mlx-lm's qwen3_5_moe
// in F32 on the 4-bit checkpoint, greedy decoding of a chat prompt, the
// logits row behind every token). Two cases, each loaded fresh:
//
//	base  LIGHT_LORA=off, the config's top-8 routing
//	lora  the adjacent lora_edge0_35b.safetensors and its K = 4
//
// Each prefills the prompt, compares the last row (top-5 ids in order, the
// largest |Δlogit| over the reference top-5 and over the vocabulary), then
// feeds the reference tokens and checks Light's argmax at every step
// (greedy agreement) and its rows. Reports the load time, the Go heap and
// the process RSS, prefill time and decode tokens/s.
//
// Runs when LIGHT_EDGE0_REF names the reference folder (the model folder
// is ref.json's "model" unless LIGHT_EDGE0_MODEL is set). Streams the
// routed experts with LIGHT_EXPERT_SLOTS; LIGHT_EDGE0_W16=1 narrows the
// dense weights to BF16 storage (the CLI's CPU default);
// LIGHT_EDGE0_HEAPPROF=<file> writes heap profiles after load and decode.
// Go heap about 21 GB resident; streamed, 2.3 GB plus the F32 slots (200 MB
// a layer for 16). Select a case with -run 'TestEdge0Real/lora$'.
func TestEdge0Real(t *testing.T) {
	dir := os.Getenv("LIGHT_EDGE0_REF")
	if dir == "" {
		t.Skip("LIGHT_EDGE0_REF not set")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "ref.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Model      string  `json:"model"`
		IDs        []int64 `json:"ids"`
		BaseTokens []int64 `json:"base_tokens"`
		LoRATokens []int64 `json:"lora_tokens"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	model := ref.Model
	if v := os.Getenv("LIGHT_EDGE0_MODEL"); v != "" {
		model = v
	}
	rows, _, err := safetensors.LoadFiles([]string{filepath.Join(dir, "ref.safetensors")}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name, lora string
		topK       int
		tokens     []int64
	}{{"base", "off", 8, ref.BaseTokens}, {"lora", "", 4, ref.LoRATokens}} {
		t.Run(c.name, func(t *testing.T) {
			t.Setenv("LIGHT_LORA", c.lora)
			want := tensor.MustSlice[float32](rows[c.name+"_logits"])
			V := len(want) / len(c.tokens)
			runEdge0Real(t, model, ref.IDs, c.tokens, want, V, c.topK, c.lora == "")
		})
		runtime.GC()
		debug.FreeOSMemory()
	}
}

func runEdge0Real(t *testing.T, model string, ids, tokens []int64, want []float32, V, topK int, lora bool) {
	cpu := device.CPU()
	t0 := time.Now()
	m, err := LoadQwen35FromHF(model, cpu, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	// F32 matches mlx-lm to rounding (5e-5 here); with BF16 storage only the
	// greedy tokens are checked (the rounded router weights can swap a
	// near-tied expert: top-5 |Δlogit| 0.6 on the LoRA run, 20 / 20 tokens).
	tol := 0.1
	if os.Getenv("LIGHT_EDGE0_W16") == "1" {
		tol = math.Inf(1)
		if _, err := StoreWeights16(m, dtype.BF16); err != nil {
			t.Fatal(err)
		}
	}
	edge0HeapProfile("load")
	b := m.Blocks[0]
	if b.MoE.TopK != topK || (b.InQKV.LoRA != nil) != lora || m.VocabSize() != V {
		t.Fatalf("top-%d (want %d), LoRA %v (want %v), vocab %d (want %d)", b.MoE.TopK, topK, b.InQKV.LoRA != nil, lora, m.VocabSize(), V)
	}
	t.Logf("loaded in %.1fs, %d layers, top-%d, LoRA %v, streamed %v; %s", time.Since(t0).Seconds(), m.NumLayers(), topK, lora, b.MoE.Stream != nil, edge0Mem())

	// Prefill: the last row against the reference.
	cache := NewKVCache(m.NumLayers())
	t1 := time.Now()
	row := lastLogits(t, m, ids, cache, cpu)
	prefill := time.Since(t1).Seconds()
	ref0 := want[:V]
	g5, w5 := top5(row), top5(ref0)
	d5, dAll := 0.0, 0.0
	for _, i := range w5 {
		d5 = math.Max(d5, math.Abs(float64(row[i]-ref0[i])))
	}
	for i := range ref0 {
		dAll = math.Max(dAll, math.Abs(float64(row[i]-ref0[i])))
	}
	t.Logf("prefill %d tokens in %.1fs: top-5 %v (mlx %v), max |Δlogit| top-5 %.3g, vocab %.3g", len(ids), prefill, g5, w5, d5, dAll)
	if g5[0] != w5[0] || d5 > tol {
		t.Errorf("prefill: argmax %d (mlx %d), top-5 |Δ| %.3g", g5[0], w5[0], d5)
	}

	// Decode the reference tokens: greedy agreement and every row.
	agree, worst, worstTop5 := 0, 0.0, 0.0
	var diverged []string
	t2 := time.Now()
	for j, tok := range tokens {
		r := want[j*V : (j+1)*V]
		if j > 0 {
			row = lastLogits(t, m, []int64{tokens[j-1]}, cache, cpu)
			for i := range r {
				worst = math.Max(worst, math.Abs(float64(row[i]-r[i])))
			}
			for _, i := range top5(r) {
				worstTop5 = math.Max(worstTop5, math.Abs(float64(row[i]-r[i])))
			}
		}
		if g := top5(row)[0]; int64(g) == tok {
			agree++
		} else {
			sorted := top5(r)
			diverged = append(diverged, fmt.Sprintf("step %d: light %d, mlx %d (mlx top-2 margin %.3g)", j, g, tok, r[sorted[0]]-r[sorted[1]]))
		}
	}
	dec := time.Since(t2).Seconds()
	tps := float64(len(tokens)-1) / dec
	var stream string
	var loads, hits int64
	for _, bl := range m.Blocks {
		if s := bl.MoE.Stream; s != nil {
			loads, hits = loads+s.Loads, hits+s.Hits
		}
	}
	if b.MoE.Stream != nil {
		stream = fmt.Sprintf(", expert loads %d, hits %d", loads, hits)
	}
	t.Logf("greedy agreement %d/%d; decode rows max |Δlogit| top-5 %.3g, vocab %.3g; %.2f tok/s%s; %s",
		agree, len(tokens), worstTop5, worst, tps, stream, edge0Mem())
	edge0HeapProfile("decode")
	runtime.KeepAlive(m) // the heap figures above include the model
	for _, d := range diverged {
		t.Log("diverged:", d)
	}
	if agree != len(tokens) || worstTop5 > tol {
		t.Errorf("greedy agreement %d/%d, top-5 |Δ| %.3g", agree, len(tokens), worstTop5)
	}
}

// edge0HeapProfile writes a heap profile to $LIGHT_EDGE0_HEAPPROF.<stage>
// when that is set.
func edge0HeapProfile(stage string) {
	hp := os.Getenv("LIGHT_EDGE0_HEAPPROF")
	if hp == "" {
		return
	}
	runtime.GC()
	if f, err := os.Create(hp + "." + stage); err == nil {
		pprof.WriteHeapProfile(f)
		f.Close()
	}
}

// edge0Mem is the live Go heap and, on Linux, the
// process's resident and peak resident set.
func edge0Mem() string {
	// Two collections: a dead tensor's storage is freed by a cleanup that
	// runs after the collection which found the tensor unreachable.
	runtime.GC()
	time.Sleep(100 * time.Millisecond)
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	s := fmt.Sprintf("Go heap %.1f GB", float64(ms.HeapAlloc)/(1<<30))
	if raw, err := os.ReadFile("/proc/self/status"); err == nil {
		for _, l := range strings.Split(string(raw), "\n") {
			if strings.HasPrefix(l, "VmRSS:") || strings.HasPrefix(l, "VmHWM:") {
				s += ", " + strings.Join(strings.Fields(l), " ")
			}
		}
	}
	return s
}
