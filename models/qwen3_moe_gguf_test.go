package models

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
	"light/gguf"
	"light/ops"
	"light/quant"
	"light/tensor"
)

// writeTinyQwen3MoEGGUF converts the tiny HF checkpoint into a llama.cpp-
// style `qwen3moe` GGUF (attention / norms / embeddings F32, experts in
// `expDT`) so the GGUF loader and the fused quantized MoE kernels can be
// checked against the same transformers reference as the HF path.
func writeTinyQwen3MoEGGUF(t *testing.T, expDT dtype.DType) string {
	t.Helper()
	hf, err := loadSafetensorsAuto(qwen3MoERepo)
	if err != nil {
		t.Fatal(err)
	}
	f32 := func(name string) *tensor.Tensor {
		x, ok := hf[name]
		if !ok {
			t.Fatalf("missing %s", name)
		}
		c, err := ops.Cast(x, dtype.F32)
		if err != nil {
			t.Fatal(err)
		}
		return c
	}
	w := gguf.NewWriter()
	w.AddString("general.architecture", "qwen3moe")
	for k, v := range map[string]uint32{
		"qwen3moe.embedding_length": 64, "qwen3moe.block_count": 2, "qwen3moe.attention.head_count": 4,
		"qwen3moe.attention.head_count_kv": 2, "qwen3moe.rope.dimension_count": 16, "qwen3moe.context_length": 4096,
		"qwen3moe.expert_count": 4, "qwen3moe.expert_used_count": 2, "qwen3moe.expert_feed_forward_length": 64,
		"qwen3moe.feed_forward_length": 128,
	} {
		w.AddU32(k, v)
	}
	w.AddF32("qwen3moe.rope.freq_base", 1e6)
	w.AddF32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6)
	add := func(name string, x *tensor.Tensor) {
		dims := x.Shape().Dims()
		rev := make([]int, len(dims))
		for i, d := range dims {
			rev[len(dims)-1-i] = d
		}
		code, err := gguf.GGMLType(x.DType())
		if err != nil {
			t.Fatal(err)
		}
		c, _ := x.Contiguous()
		if err := w.AddTensor(name, rev, code, c.Storage().CPUBytes()); err != nil {
			t.Fatal(err)
		}
	}
	add("token_embd.weight", f32("model.embed_tokens.weight"))
	add("output.weight", f32("lm_head.weight"))
	add("output_norm.weight", f32("model.norm.weight"))
	for i := range 2 {
		p := "model.layers." + string(rune('0'+i)) + "."
		b := "blk." + string(rune('0'+i)) + "."
		add(b+"attn_q.weight", f32(p+"self_attn.q_proj.weight"))
		add(b+"attn_k.weight", f32(p+"self_attn.k_proj.weight"))
		add(b+"attn_v.weight", f32(p+"self_attn.v_proj.weight"))
		add(b+"attn_output.weight", f32(p+"self_attn.o_proj.weight"))
		add(b+"attn_q_norm.weight", f32(p+"self_attn.q_norm.weight"))
		add(b+"attn_k_norm.weight", f32(p+"self_attn.k_norm.weight"))
		add(b+"attn_norm.weight", f32(p+"input_layernorm.weight"))
		add(b+"ffn_norm.weight", f32(p+"post_attention_layernorm.weight"))
		add(b+"ffn_gate_inp.weight", f32(p+"mlp.router.weight"))
		gu := f32(p + "mlp.experts.gate_up_proj") // (E, 2I, H)
		I := gu.Dim(1) / 2
		gate, _ := gu.Narrow(1, 0, I)
		up, _ := gu.Narrow(1, I, I)
		gate, _ = gate.Contiguous()
		up, _ = up.Contiguous()
		down := f32(p + "mlp.experts.down_proj") // (E, H, I)
		for name, x := range map[string]*tensor.Tensor{"ffn_gate_exps": gate, "ffn_up_exps": up, "ffn_down_exps": down} {
			q := x
			if expDT.IsQuantized() {
				var err error
				if q, err = quant.Quantize(x, expDT); err != nil {
					t.Fatal(err)
				}
			}
			add(b+name+".weight", q)
		}
	}
	data, err := w.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "tiny-qwen3moe-"+expDT.String()+".gguf")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func checkQwen3MoEGGUF(t *testing.T, path string, dev device.Device, expDT dtype.DType, logitTol float64) {
	t.Helper()
	m, err := LoadLlamaGGUF(path, dev)
	if err != nil {
		t.Fatal(err)
	}
	if !m.Cfg.QKNorm || m.Cfg.NumExperts != 4 || m.Cfg.NumExpertsPerTok != 2 || !m.Cfg.NormTopKProb || m.Blocks[0].MoE == nil {
		t.Fatalf("GGUF MoE config not applied: %+v", m.Cfg)
	}
	if got := m.Blocks[0].MoE.GateUp.DType(); got != expDT {
		t.Fatalf("expert dtype %v, want %v", got, expDT)
	}
	row := lastLogits(t, m, qwen3MoEPromptIDs, NewKVCache(m.NumLayers()), dev)
	got := top5(row)
	if got[0] != qwen3MoERefF32[0].id {
		t.Errorf("top-1 id %d (%.5f), want %d", got[0], row[got[0]], qwen3MoERefF32[0].id)
	}
	for _, w := range qwen3MoERefF32 {
		if d := math.Abs(float64(row[w.id] - w.logit)); d > logitTol {
			t.Errorf("logit[%d] = %.5f, want %.5f (tol %g)", w.id, row[w.id], w.logit, logitTol)
		}
	}
	t.Logf("%v experts %v top-5 %v logits %.5f %.5f %.5f %.5f %.5f", dev, expDT, got, row[got[0]], row[got[1]], row[got[2]], row[got[3]], row[got[4]])
}

func TestQwen3MoEGGUF(t *testing.T) {
	qwen3MoECached(t)
	for _, tc := range []struct {
		dt  dtype.DType
		tol float64
	}{{dtype.F32, 2e-3}, {dtype.Q8_0, 1e-2}, {dtype.Q4_0, 5e-2}} {
		path := writeTinyQwen3MoEGGUF(t, tc.dt)
		checkQwen3MoEGGUF(t, path, device.CPU(), tc.dt, tc.tol)
		if device.CUDAAvailable() {
			dev, err := device.CUDA(0)
			if err != nil {
				t.Fatal(err)
			}
			checkQwen3MoEGGUF(t, path, dev, tc.dt, tc.tol)
		}
	}
}
