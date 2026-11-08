package models

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"light/device"
	"light/dtype"
	"light/gguf"
	"light/ops"
	"light/quant"
	"light/tensor"
)

// writeTinyQwen35GGUF converts the tiny HF checkpoint (testdata/qwen35/tiny:
// 2 key / 4 value heads in the DeltaNet layers) into a `qwen35` GGUF the way
// llama.cpp's converter does (conversion/qwen.py, Qwen3_5TextModel): +1 on
// every norm except linear_attn.norm, ssm_a = −exp(A_log), conv squeezed to
// (C, K), and the value heads reordered from grouped to tiled order in every
// tensor that indexes them. 2-D weights are written as linDT (F32 or a block
// format); norms, conv, ssm_a / ssm_dt stay F32.
func writeTinyQwen35GGUF(t *testing.T, linDT dtype.DType) string {
	t.Helper()
	cfg, err := fetchQwen35Config(qwen35TinyDir)
	if err != nil {
		t.Fatal(err)
	}
	hf, err := loadSafetensorsAuto(qwen35TinyDir)
	if err != nil {
		t.Fatal(err)
	}
	get := func(name string) ([]float32, []int) {
		x, ok := hf[name]
		if !ok {
			t.Fatalf("missing %s", name)
		}
		c, err := ops.Cast(x, dtype.F32)
		if err != nil {
			t.Fatal(err)
		}
		return slices.Clone(tensor.MustSlice[float32](c)), c.Shape().Dims()
	}
	Hk, Hv, dk, dv := cfg.LinearKHeads, cfg.LinearVHeads, cfg.LinearKDim, cfg.LinearVDim
	r := Hv / Hk
	// tiled maps a value-head-indexed position (head j in grouped order
	// g·r + i, element e) to its tiled position (i·Hk + g, e).
	tiled := func(j, e, hd int) int {
		g, i := j/r, j%r
		return (i*Hk+g)*hd + e
	}
	reorderRows := func(x []float32, rows, cols, first, hd int) []float32 {
		out := slices.Clone(x)
		for j := 0; j < (rows-first)/hd; j++ {
			for e := range hd {
				src, dst := first+j*hd+e, first+tiled(j, e, hd)
				copy(out[dst*cols:(dst+1)*cols], x[src*cols:(src+1)*cols])
			}
		}
		return out
	}
	reorderCols := func(x []float32, rows, cols, hd int) []float32 {
		out := slices.Clone(x)
		for row := range rows {
			for j := 0; j < cols/hd; j++ {
				for e := range hd {
					out[row*cols+tiled(j, e, hd)] = x[row*cols+j*hd+e]
				}
			}
		}
		return out
	}

	w := gguf.NewWriter()
	w.AddString("general.architecture", "qwen35")
	for k, v := range map[string]int{
		"block_count": cfg.NumLayers, "embedding_length": cfg.HiddenSize, "feed_forward_length": cfg.IntermediateSize,
		"attention.head_count": cfg.NumHeads, "attention.head_count_kv": cfg.NumKVHeads,
		"attention.key_length": cfg.HeadDim, "attention.value_length": cfg.HeadDim,
		"rope.dimension_count": cfg.rotaryDim(), "context_length": cfg.MaxPosition,
		"ssm.conv_kernel": cfg.ConvKernel, "ssm.state_size": dk, "ssm.group_count": Hk,
		"ssm.time_step_rank": Hv, "ssm.inner_size": Hv * dv, "full_attention_interval": 2,
	} {
		w.AddU32("qwen35."+k, uint32(v))
	}
	w.AddF32("qwen35.rope.freq_base", float32(cfg.RopeBase))
	w.AddF32("qwen35.attention.layer_norm_rms_epsilon", float32(cfg.RmsEps))
	add := func(name string, x []float32, dims []int, dt dtype.DType) {
		tt, err := tensor.New(x, dims, device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		if dt.IsQuantized() {
			if tt, err = quant.Quantize(tt, dt); err != nil {
				t.Fatalf("%s: %v", name, err)
			}
		}
		rev := make([]int, len(dims))
		for i, d := range dims {
			rev[len(dims)-1-i] = d
		}
		code, err := gguf.GGMLType(tt.DType())
		if err != nil {
			t.Fatal(err)
		}
		if err := w.AddTensor(name, rev, code, tt.Storage().CPUBytes()); err != nil {
			t.Fatal(err)
		}
	}
	lin := func(name, hfName string) {
		x, d := get(hfName)
		add(name, x, d, linDT)
	}
	plusOne := func(name, hfName string) {
		x, d := get(hfName)
		for i := range x {
			x[i]++
		}
		add(name, x, d, dtype.F32)
	}
	const pre = "model."
	lin("token_embd.weight", pre+"embed_tokens.weight")
	lin("output.weight", "lm_head.weight")
	plusOne("output_norm.weight", pre+"norm.weight")
	for i, isLin := range cfg.LinearLayers {
		p, b := fmt.Sprintf("%slayers.%d.", pre, i), fmt.Sprintf("blk.%d.", i)
		plusOne(b+"attn_norm.weight", p+"input_layernorm.weight")
		plusOne(b+"post_attention_norm.weight", p+"post_attention_layernorm.weight")
		lin(b+"ffn_gate.weight", p+"mlp.gate_proj.weight")
		lin(b+"ffn_up.weight", p+"mlp.up_proj.weight")
		lin(b+"ffn_down.weight", p+"mlp.down_proj.weight")
		if !isLin {
			sa := p + "self_attn."
			lin(b+"attn_q.weight", sa+"q_proj.weight")
			lin(b+"attn_k.weight", sa+"k_proj.weight")
			lin(b+"attn_v.weight", sa+"v_proj.weight")
			lin(b+"attn_output.weight", sa+"o_proj.weight")
			plusOne(b+"attn_q_norm.weight", sa+"q_norm.weight")
			plusOne(b+"attn_k_norm.weight", sa+"k_norm.weight")
			continue
		}
		la := p + "linear_attn."
		x, d := get(la + "in_proj_qkv.weight")
		add(b+"attn_qkv.weight", reorderRows(x, d[0], d[1], 2*Hk*dk, dv), d, linDT)
		x, d = get(la + "in_proj_z.weight")
		add(b+"attn_gate.weight", reorderRows(x, d[0], d[1], 0, dv), d, linDT)
		x, d = get(la + "in_proj_a.weight")
		add(b+"ssm_alpha.weight", reorderRows(x, d[0], d[1], 0, 1), d, linDT)
		x, d = get(la + "in_proj_b.weight")
		add(b+"ssm_beta.weight", reorderRows(x, d[0], d[1], 0, 1), d, linDT)
		x, d = get(la + "out_proj.weight")
		add(b+"ssm_out.weight", reorderCols(x, d[0], d[1], dv), d, linDT)
		x, d = get(la + "conv1d.weight") // (C, 1, K) → (C, K)
		C, K := d[0], d[2]
		add(b+"ssm_conv1d.weight", reorderRows(x, C, K, 2*Hk*dk, dv), []int{C, K}, dtype.F32)
		x, d = get(la + "A_log")
		for i := range x {
			x[i] = -float32(math.Exp(float64(x[i])))
		}
		add(b+"ssm_a", reorderRows(x, d[0], 1, 0, 1), d, dtype.F32)
		x, d = get(la + "dt_bias")
		add(b+"ssm_dt.bias", reorderRows(x, d[0], 1, 0, 1), d, dtype.F32)
		x, d = get(la + "norm.weight")
		add(b+"ssm_norm.weight", x, d, dtype.F32)
	}
	data, err := w.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "tiny-qwen35-"+linDT.String()+".gguf")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestQwen35GGUFTiny loads the converted tiny checkpoint and compares it with
// the HF load of the same weights: in F32 the logits must agree to float
// rounding (the reorder is a pure permutation and VHeadsTiled undoes it), in
// Q8_0 to quantization error; both must keep the transformers top-1 and
// greedy continuation. Running the tiled file without VHeadsTiled must
// change the logits, so the test would notice the flag being ignored.
func TestQwen35GGUFTiny(t *testing.T) {
	devs := []device.Device{device.CPU()}
	if device.CUDAAvailable() {
		if gpu, err := device.CUDA(0); err == nil {
			devs = append(devs, gpu)
		}
	}
	ids := qwen35TinyIDs()
	for _, dev := range devs {
		ref, err := LoadQwen35FromHF(qwen35TinyDir, dev, dtype.F32)
		if err != nil {
			t.Fatal(err)
		}
		want := lastLogits(t, ref, ids, NewKVCache(ref.NumLayers()), dev)
		type fixture struct {
			name string
			dt   dtype.DType
			tol  float64
			path func(t *testing.T) string
		}
		cases := []fixture{
			{"go-f32", dtype.F32, 1e-4, func(t *testing.T) string { return writeTinyQwen35GGUF(t, dtype.F32) }},
			{"go-q8_0", dtype.Q8_0, 2e-2, func(t *testing.T) string { return writeTinyQwen35GGUF(t, dtype.Q8_0) }},
		}
		// The same checkpoint converted by llama.cpp's own converter
		// (scripts/verify/qwen35_gguf_fixture.py), when present: pins the
		// rules writeTinyQwen35GGUF reimplements to what llama.cpp writes.
		for _, f := range []struct {
			file string
			dt   dtype.DType
			tol  float64
		}{{"tiny-llamacpp-f32.gguf", dtype.F32, 1e-4}, {"tiny-llamacpp-q8_0.gguf", dtype.Q8_0, 2e-2}} {
			p := filepath.Join("testdata", "qwen35", f.file)
			if _, err := os.Stat(p); err == nil {
				cases = append(cases, fixture{f.file, f.dt, f.tol, func(*testing.T) string { return p }})
			}
		}
		for _, tc := range cases {
			t.Run(fmt.Sprintf("%v/%s", dev, tc.name), func(t *testing.T) {
				m, err := LoadQwen35GGUF(tc.path(t), dev)
				if err != nil {
					t.Fatal(err)
				}
				c := m.Cfg
				if !c.VHeadsTiled || c.LinearVHeads != 4 || c.LinearKHeads != 2 || c.rotaryDim() != cfgRot(t) ||
					!c.AttnOutputGate || c.TieEmbed || !slices.Equal(c.LinearLayers, []bool{true, false, true, false}) {
					t.Fatalf("config %+v", c)
				}
				if got := m.Blocks[0].InQKV.W.DType(); got != tc.dt {
					t.Fatalf("attn_qkv dtype %v, want %v", got, tc.dt)
				}
				cache := NewKVCache(m.NumLayers())
				row := lastLogits(t, m, ids, cache, dev)
				d := maxAbsDiff(want, row)
				t.Logf("max |GGUF − HF| = %.2e; top-5 %v", d, top5(row))
				if d > tc.tol {
					t.Errorf("GGUF logits differ from the HF load by %.2e (tol %g)", d, tc.tol)
				}
				// The tiny model's top two logits are 3e-4 apart, inside Q8_0's
				// error, so only F32 must reproduce the exact ranking.
				if tc.dt == dtype.F32 && top5(row)[0] != qwen35TinyRef[0].id {
					t.Errorf("top-1 %d, want %d", top5(row)[0], qwen35TinyRef[0].id)
				}
				first := slices.Clone(row)
				var greedy []int64
				for len(greedy) < len(qwen35TinyGreedy) {
					next := int64(top5(row)[0])
					greedy = append(greedy, next)
					row = lastLogits(t, m, []int64{next}, cache, dev)
				}
				if tc.dt == dtype.F32 && !slices.Equal(greedy, qwen35TinyGreedy) {
					t.Errorf("greedy %v, want %v", greedy, qwen35TinyGreedy)
				}
				m.Cfg.VHeadsTiled = false
				if d := maxAbsDiff(first, lastLogits(t, m, ids, NewKVCache(m.NumLayers()), dev)); d < 1e-3 {
					t.Errorf("tiled weights read in grouped order give the same logits (diff %.2e)", d)
				}
			})
		}
	}
}

func cfgRot(t *testing.T) int {
	t.Helper()
	cfg, err := fetchQwen35Config(qwen35TinyDir)
	if err != nil {
		t.Fatal(err)
	}
	return cfg.rotaryDim()
}

// Qwen3.5-27B as unsloth's Q4_K_M GGUF (16.7 GB: Q4_K / Q5_K / Q6_K / Q8_0)
// on CUDA. 64 layers (48 DeltaNet), 16 key / 48 value heads in tiled order,
// 24 query / 4 KV heads of 256. Fits a 22 GiB L4 as well as an A100.
//
// Reference: transformers 5.21 on the BF16 checkpoint (54 GB, so on the CPU
// of a Colab High-RAM runtime):
//
//	python scripts/verify/ref_qwen35.py Qwen/Qwen3.5-27B bf16 --chat
//
// Q4_K_M is not BF16, so the comparison is loose: the reference top-1, the
// reference logits within 1.5, and the greedy continuation agreeing on a
// prefix. Reference from an A100 80 GB (BF16 on the GPU); the prompt ids are
// the 9B's (same tokenizer and template).
//
// colab-download: unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-Q4_K_M.gguf
const qwen3527BGGUF = "Qwen3.5-27B-Q4_K_M.gguf"

var (
	qwen3527BChatIDs = qwen359BChatIDs
	qwen3527BRef     = []struct {
		id    int
		logit float32
	}{{90700, 29.625}, {760, 21.75}, {31248, 17.625}, {79420, 17.25}, {50534, 15.9375}}
	// "Thinking Process:\n\n1.  **Analyze the Request:**\n    *"
	qwen3527BGreedy = []int64{90700, 8340, 25, 271, 16, 13, 220, 2972, 2014, 53983, 279, 5952, 64700, 198, 262, 348}
)

func TestQwen3527BGGUFMatchesTransformers(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (16.7 GB GGUF)")
	}
	home, _ := os.UserHomeDir()
	files, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--unsloth--Qwen3.5-27B-GGUF/snapshots/*/"+qwen3527BGGUF))
	if len(files) == 0 {
		t.Skip("unsloth/Qwen3.5-27B-GGUF " + qwen3527BGGUF + " not in HF cache")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	requireGPUMemGB(t, 0, 19)
	releaseModelMemory(t)
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	m, err := LoadQwen35GGUF(files[0], dev)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	linear := 0
	for _, l := range c.LinearLayers {
		if l {
			linear++
		}
	}
	if c.NumLayers != 64 || linear != 48 || c.LinearLayers[3] || c.LinearKHeads != 16 || c.LinearVHeads != 48 ||
		c.LinearKDim != 128 || c.LinearVDim != 128 || c.HeadDim != 256 || c.rotaryDim() != 64 || c.NumHeads != 24 ||
		c.NumKVHeads != 4 || !c.AttnOutputGate || c.TieEmbed || !c.VHeadsTiled || c.HiddenSize != 5120 {
		t.Fatalf("config %+v", c)
	}
	t.Logf("attn_qkv %v, ffn_down %v, lm head %v", m.Blocks[0].InQKV.W.DType(), m.Blocks[0].Down.W.DType(), m.LMHead.W.DType())

	cache := NewKVCache(m.NumLayers())
	row := lastLogits(t, m, qwen3527BChatIDs, cache, dev)
	got := top5(row)
	t.Logf("top-5 %v logits %.3f %.3f %.3f %.3f %.3f", got, row[got[0]], row[got[1]], row[got[2]], row[got[3]], row[got[4]])
	// Chunked (prefill) and recurrent delta rule on the quantized model.
	m.deltaMode = ops.GatedDeltaRecurrent
	rec := lastLogits(t, m, qwen3527BChatIDs, NewKVCache(m.NumLayers()), dev)
	m.deltaMode = ops.GatedDeltaAuto
	if d := maxAbsDiff(row, rec); top5(rec)[0] != got[0] || d > 0.05 {
		t.Errorf("recurrent prefill top-1 %d (chunked %d), logits differ by up to %.3g", top5(rec)[0], got[0], d)
	} else {
		t.Logf("chunked vs recurrent prefill: max |Δ| %.3g", d)
	}

	first := slices.Clone(row)
	var greedy []int64
	for len(greedy) < 16 {
		next := int64(top5(row)[0])
		greedy = append(greedy, next)
		row = lastLogits(t, m, []int64{next}, cache, dev)
	}
	t.Logf("greedy %v", greedy)

	if got[0] != qwen3527BRef[0].id {
		t.Errorf("top-1 %d (%.3f), want %d (%.3f)", got[0], first[got[0]], qwen3527BRef[0].id, qwen3527BRef[0].logit)
	}
	for _, w := range qwen3527BRef {
		if d := first[w.id] - w.logit; d > 1.5 || d < -1.5 {
			t.Errorf("logit[%d] = %.3f, want %.3f (Q4_K_M vs BF16 tol 1.5)", w.id, first[w.id], w.logit)
		}
	}
	agree := 0
	for agree < len(greedy) && agree < len(qwen3527BGreedy) && greedy[agree] == qwen3527BGreedy[agree] {
		agree++
	}
	t.Logf("greedy: first %d of %d agree with transformers BF16", agree, len(qwen3527BGreedy))
	if agree < 4 {
		t.Errorf("greedy %v diverges from transformers BF16 %v after %d tokens", greedy, qwen3527BGreedy, agree)
	}
}
