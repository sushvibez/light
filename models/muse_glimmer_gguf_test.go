package models

import (
	"fmt"
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

// writeTinyMuseGlimmerGGUF converts the tiny HF checkpoint into a
// `muse-glimmer` GGUF following llama.cpp (conversion/muse_glimmer.py):
// +1 on the four layer norms, q / k rows permuted to the interleaved RoPE
// layout, attn_q_norm synthesized as qk_scale_factor and attn_k_norm as
// ones, sliding layers as a period, output_multiplier as logit_scale.
// 2-D weights are written as linDT.
func writeTinyMuseGlimmerGGUF(t *testing.T, linDT dtype.DType) string {
	t.Helper()
	cfg, err := fetchMuseGlimmerConfig(museGlimmerTinyDir)
	if err != nil {
		t.Fatal(err)
	}
	hf, err := loadSafetensorsAuto(museGlimmerTinyDir)
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
	// toInterleaved is llama.cpp's _unpermute_for_rope: HF row
	// h·hd + s·hd/2 + i lands at row h·hd + 2i + s.
	toInterleaved := func(x []float32, rows, cols, heads int) []float32 {
		out := make([]float32, len(x))
		hd := rows / heads
		half := hd / 2
		for h := range heads {
			for s := range 2 {
				for i := range half {
					src, dst := h*hd+s*half+i, h*hd+2*i+s
					copy(out[dst*cols:(dst+1)*cols], x[src*cols:(src+1)*cols])
				}
			}
		}
		return out
	}
	const arch = "muse-glimmer."
	w := gguf.NewWriter()
	w.AddString("general.architecture", "muse-glimmer")
	for k, v := range map[string]int{
		"block_count": cfg.NumLayers, "embedding_length": cfg.HiddenSize, "feed_forward_length": cfg.IntermediateSize,
		"attention.head_count": cfg.NumHeads, "attention.head_count_kv": cfg.NumKVHeads,
		"attention.key_length": cfg.HeadDim, "attention.value_length": cfg.HeadDim, "context_length": cfg.MaxPosition,
		"attention.sliding_window": cfg.SlidingWindow, "attention.sliding_window_pattern": 4,
	} {
		w.AddU32(arch+k, uint32(v))
	}
	w.AddF32(arch+"rope.freq_base", float32(cfg.LayerRopeBase[0]))
	w.AddF32(arch+"attention.layer_norm_rms_epsilon", float32(cfg.RmsEps))
	w.AddF32(arch+"final_logit_softcapping", float32(cfg.FinalLogitSoftcap))
	w.AddF32(arch+"logit_scale", float32(cfg.OutputMultiplier))
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
	norm := func(name, hfName string, shift float32) {
		x, d := get(hfName)
		for i := range x {
			x[i] += shift
		}
		add(name, x, d, dtype.F32)
	}
	const pre = "model.language_model."
	x, d := get(pre + "embed_tokens.weight")
	add("token_embd.weight", x, d, linDT)
	x, d = get("lm_head.weight")
	add("output.weight", x, d, linDT)
	norm("output_norm.weight", pre+"norm.weight", 0)
	for i := range cfg.NumLayers {
		p, b := fmt.Sprintf("%slayers.%d.", pre, i), fmt.Sprintf("blk.%d.", i)
		norm(b+"attn_norm.weight", p+"input_layernorm.weight", 1)
		norm(b+"post_attention_norm.weight", p+"post_attention_layernorm.weight", 1)
		norm(b+"ffn_norm.weight", p+"pre_feedforward_layernorm.weight", 1)
		norm(b+"post_ffw_norm.weight", p+"post_feedforward_layernorm.weight", 1)
		x, d = get(p + "self_attn.q_proj.weight")
		add(b+"attn_q.weight", toInterleaved(x, d[0], d[1], cfg.NumHeads), d, linDT)
		x, d = get(p + "self_attn.k_proj.weight")
		add(b+"attn_k.weight", toInterleaved(x, d[0], d[1], cfg.NumKVHeads), d, linDT)
		for name, hfName := range map[string]string{
			"attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj", "attn_gate": "self_attn.gate_proj",
			"ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
		} {
			x, d = get(p + hfName + ".weight")
			add(b+name+".weight", x, d, linDT)
		}
		qn, kn := make([]float32, cfg.HeadDim), make([]float32, cfg.HeadDim)
		for j := range qn {
			qn[j], kn[j] = float32(cfg.QScale), 1
		}
		add(b+"attn_q_norm.weight", qn, []int{cfg.HeadDim}, dtype.F32)
		add(b+"attn_k_norm.weight", kn, []int{cfg.HeadDim}, dtype.F32)
	}
	data, err := w.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "tiny-muse-glimmer-"+linDT.String()+".gguf")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestMuseGlimmerGGUFTiny: the tiny checkpoint converted as llama.cpp does
// must reproduce the HF load (F32: float rounding; Q8_0: quantization error)
// and the transformers greedy continuation (F32).
func TestMuseGlimmerGGUFTiny(t *testing.T) {
	devs := []device.Device{device.CPU()}
	if device.CUDAAvailable() {
		if gpu, err := device.CUDA(0); err == nil {
			devs = append(devs, gpu)
		}
	}
	ids := museGlimmerTinyIDs
	for _, dev := range devs {
		ref, err := LoadMuseGlimmerFromHF(museGlimmerTinyDir, dev, dtype.F32)
		if err != nil {
			t.Fatal(err)
		}
		want := lastLogits(t, ref, ids, NewKVCache(ref.NumLayers()), dev)
		for _, tc := range []struct {
			dt  dtype.DType
			tol float64
		}{{dtype.F32, 1e-5}, {dtype.Q8_0, 1e-2}} { // Q8_0: embedding and every Linear quantized
			t.Run(fmt.Sprintf("%v/%v", dev, tc.dt), func(t *testing.T) {
				m, err := LoadMuseGlimmerGGUF(writeTinyMuseGlimmerGGUF(t, tc.dt), dev)
				if err != nil {
					t.Fatal(err)
				}
				c := m.Cfg
				if !slices.Equal(c.SlidingLayers, []bool{true, true, true, false}) ||
					!slices.Equal(c.LayerRopeBase, []float64{10000, 10000, 10000, 0}) || !c.UntiedHead || c.QScale != 0 {
					t.Fatalf("config %+v", c)
				}
				if got := m.Blocks[0].QProj.W.DType(); got != tc.dt {
					t.Fatalf("attn_q dtype %v, want %v", got, tc.dt)
				}
				cache := NewKVCache(m.NumLayers())
				row := lastLogits(t, m, ids, cache, dev)
				d := maxAbsDiff(want, row)
				t.Logf("max |GGUF − HF| = %.2e; top-5 %v", d, top5(row))
				if d > tc.tol {
					t.Errorf("GGUF logits differ from the HF load by %.2e (tol %g)", d, tc.tol)
				}
				if tc.dt != dtype.F32 {
					return
				}
				var greedy []int64
				for len(greedy) < len(museGlimmerTinyGreedy) {
					next := int64(top5(row)[0])
					greedy = append(greedy, next)
					row = lastLogits(t, m, []int64{next}, cache, dev)
				}
				if !slices.Equal(greedy, museGlimmerTinyGreedy) {
					t.Errorf("greedy %v, want %v", greedy, museGlimmerTinyGreedy)
				}
			})
		}
	}
}

// Muse Glimmer 30B as unsloth's UD-Q4_K_XL GGUF (15.9 GB: Q4_K / Q5_K + F32
// norms) on CUDA: 52 layers (13 global NoPE), 32 query / 2 KV heads of 128
// on a 6656-wide model. Fits a 22 GiB L4 as well as an A100.
//
// Reference: transformers 5.21 on the BF16 checkpoint (60 GB: GPU on an
// 80 GB A100, else the CPU of a High-RAM runtime):
//
//	python scripts/verify/ref_muse_glimmer.py meta-models/Muse-Glimmer-30B bf16 --chat
//
// Q4 against BF16, so the bar is loose: reference top-1, logits within 1.5,
// greedy agreeing on a prefix. Reference from an A100 80 GB (BF16 on the
// GPU). The chat template renders a harmony-style prompt (default system
// block included) ending in an assistant header; the model answers
// " to=self<|message|>" and echoes the question — odd, but it is what the
// checkpoint does, and Light must do the same.
//
// colab-download: unsloth/Muse-Glimmer-30B-GGUF Muse-Glimmer-30B-UD-Q4_K_XL.gguf
const museGlimmer30BGGUF = "Muse-Glimmer-30B-UD-Q4_K_XL.gguf"

var (
	museGlimmer30BChatIDs = []int64{200000, 200022, 15651, 200023, 4662, 583, 262, 81080, 43910, 1574, 34956, 300, 9762, 38,
		2244, 1574, 15, 14757, 73965, 38, 392, 2540, 706, 392, 1556, 4205, 200008, 200022, 1556, 200023, 51015, 310, 1085,
		17705, 5959, 290, 21752, 373, 10950, 26, 200008, 200022, 140680}
	museGlimmer30BRef = []struct {
		id    int
		logit float32
	}{{328, 18.125}, {5445, -0.2598}, {13532, -1.6094}, {5740, -2.9688}, {34158, -3.2031}}
	// " to=self<|message|>Explain in one sentence why the sky is blue.\n\nExplain in one"
	museGlimmer30BGreedy = []int64{328, 19669, 200023, 51015, 310, 1085, 17705, 5959, 290, 21752, 373, 10950, 1574, 51015, 310, 1085}
)

func TestMuseGlimmer30BGGUFMatchesTransformers(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (15.9 GB GGUF)")
	}
	home, _ := os.UserHomeDir()
	files, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--unsloth--Muse-Glimmer-30B-GGUF/snapshots/*/"+museGlimmer30BGGUF))
	if len(files) == 0 {
		t.Skip("unsloth/Muse-Glimmer-30B-GGUF " + museGlimmer30BGGUF + " not in HF cache")
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
	m, err := LoadMuseGlimmerGGUF(files[0], dev)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	global := 0
	for _, s := range c.SlidingLayers {
		if !s {
			global++
		}
	}
	if c.NumLayers != 52 || global != 13 || c.SlidingLayers[3] || !c.SlidingLayers[0] || c.LayerRopeBase[3] != 0 ||
		c.LayerRopeBase[0] != 500000 || c.NumHeads != 32 || c.NumKVHeads != 2 || c.HeadDim != 128 ||
		c.HiddenSize != 6656 || c.SlidingWindow != 2048 || !c.UntiedHead || c.FinalLogitSoftcap != 20 {
		t.Fatalf("config %+v", c)
	}
	t.Logf("attn_q %v, ffn_down %v, lm head %v", m.Blocks[0].QProj.W.DType(), m.Blocks[0].DownProj.W.DType(), m.LMHead.W.DType())
	cache := NewKVCache(m.NumLayers())
	row := lastLogits(t, m, museGlimmer30BChatIDs, cache, dev)
	got := top5(row)
	t.Logf("top-5 %v logits %.3f %.3f %.3f %.3f %.3f", got, row[got[0]], row[got[1]], row[got[2]], row[got[3]], row[got[4]])
	first := slices.Clone(row)
	var greedy []int64
	for len(greedy) < len(museGlimmer30BGreedy) {
		next := int64(top5(row)[0])
		greedy = append(greedy, next)
		row = lastLogits(t, m, []int64{next}, cache, dev)
	}
	t.Logf("greedy %v", greedy)
	if got[0] != museGlimmer30BRef[0].id {
		t.Errorf("top-1 %d (%.3f), want %d (%.3f)", got[0], first[got[0]], museGlimmer30BRef[0].id, museGlimmer30BRef[0].logit)
	}
	for _, w := range museGlimmer30BRef {
		if d := first[w.id] - w.logit; d > 1.5 || d < -1.5 {
			t.Errorf("logit[%d] = %.3f, want %.3f (Q4 vs BF16 tol 1.5)", w.id, first[w.id], w.logit)
		}
	}
	agree := 0
	for agree < len(greedy) && greedy[agree] == museGlimmer30BGreedy[agree] {
		agree++
	}
	t.Logf("greedy: first %d of %d agree with transformers BF16", agree, len(greedy))
	if agree < 4 {
		t.Errorf("greedy %v diverges from transformers BF16 %v after %d tokens", greedy, museGlimmer30BGreedy, agree)
	}
}

// LoadChatModelGGUF (the engine's GGUF entry point) dispatches on
// general.architecture.
func TestLoadChatModelGGUFDispatch(t *testing.T) {
	cpu := device.CPU()
	m, err := LoadChatModelGGUF(writeTinyMuseGlimmerGGUF(t, dtype.F32), cpu)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := m.(*Gemma3); !ok {
		t.Errorf("muse-glimmer GGUF loaded as %T, want *Gemma3", m)
	}
	q, err := LoadChatModelGGUF(writeTinyQwen35GGUF(t, dtype.F32), cpu)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := q.(*Qwen35); !ok {
		t.Errorf("qwen35 GGUF loaded as %T, want *Qwen35", q)
	}
}
