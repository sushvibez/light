package models

import (
	"testing"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// Key layouts: diffusers / PEFT, kohya (lora_unet_, "_" for "."), BFL
// module names, and the fused projections splitting by rows.
func TestFluxLoRAKeys(t *testing.T) {
	cases := []struct{ key, mod, part string }{
		{"transformer.transformer_blocks.0.attn.to_q.lora_A.weight", "transformer_blocks.0.attn.to_q", "A"},
		{"transformer.single_transformer_blocks.3.proj_out.lora_B.weight", "single_transformer_blocks.3.proj_out", "B"},
		{"lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight", "double_blocks_0_img_attn_qkv", "A"},
		{"lora_unet_single_blocks_7_linear1.alpha", "single_blocks_7_linear1", "alpha"},
		{"lora_transformer_transformer_blocks_2_attn_to_out_0.lora_up.weight", "transformer_blocks.2.attn.to_out.0", "B"},
		{"lora_transformer_single_transformer_blocks_1_norm_linear.lora_down.weight", "single_transformer_blocks.1.norm.linear", "A"},
		{"diffusion_model.double_blocks.1.txt_mlp.0.lora.down.weight", "double_blocks.1.txt_mlp.0", "A"},
	}
	for _, c := range cases {
		mod, part, ok := splitLoRAKey(c.key)
		if !ok || mod != c.mod || part != c.part {
			t.Errorf("%s: (%q, %q, %v), want (%q, %q)", c.key, mod, part, ok, c.mod, c.part)
		}
	}
	for _, k := range []string{"lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight", "text_encoder.layers.0.q.lora_A.weight", "transformer.x_embedder.weight"} {
		if _, _, ok := splitLoRAKey(k); ok {
			t.Errorf("%s: accepted", k)
		}
	}
	targets := map[string][]string{
		"double_blocks_0_img_attn_qkv":   {"transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_k", "transformer_blocks.0.attn.to_v"},
		"double_blocks.4.txt_attn.proj":  {"transformer_blocks.4.attn.to_add_out"},
		"double_blocks.4.txt_mod.lin":    {"transformer_blocks.4.norm1_context.linear"},
		"double_blocks.2.img_mlp.2":      {"transformer_blocks.2.ff.net.2"},
		"single_blocks_7_linear1":        {"single_transformer_blocks.7.attn.to_q", "single_transformer_blocks.7.attn.to_k", "single_transformer_blocks.7.attn.to_v", "single_transformer_blocks.7.proj_mlp"},
		"single_blocks.7.modulation.lin": {"single_transformer_blocks.7.norm.linear"},
		"final_layer.linear":             {"proj_out"},
		"img_in":                         {"x_embedder"},
		"transformer_blocks.1.ff.net.2":  {"transformer_blocks.1.ff.net.2"},
	}
	for mod, want := range targets {
		got, _ := fluxLoRATargets(mod)
		if len(got) != len(want) {
			t.Errorf("%s: %v, want %v", mod, got, want)
			continue
		}
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("%s: %v, want %v", mod, got, want)
				break
			}
		}
	}
	if got, _ := fluxLoRATargets("double_blocks.0.img_attn.norm"); got != nil {
		t.Errorf("norm: %v", got)
	}

	// Fused: qkv B (3·out, r) splits in thirds; linear1 B (7D, r) into (D, D, D, 4D).
	cpu := device.CPU()
	a, _ := tensor.Zeros(dtype.F32, []int{2, 8}, cpu)
	bqkv, _ := tensor.Zeros(dtype.F32, []int{24, 2}, cpu)
	b1, _ := tensor.Zeros(dtype.F32, []int{56, 2}, cpu)
	alpha, _ := tensor.New([]float32{4}, []int{1}, cpu)
	ad, err := ParseFluxLoRA(map[string]*tensor.Tensor{
		"lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": a, "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight": bqkv,
		"lora_unet_double_blocks_0_img_attn_qkv.alpha":       alpha,
		"lora_unet_single_blocks_0_linear1.lora_down.weight": a, "lora_unet_single_blocks_0_linear1.lora_up.weight": b1,
		"lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": a,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(ad.Skipped) != 1 {
		t.Errorf("skipped %v", ad.Skipped)
	}
	for name, rows := range map[string]int{"transformer_blocks.0.attn.to_q": 8, "transformer_blocks.0.attn.to_v": 8,
		"single_transformer_blocks.0.attn.to_k": 8, "single_transformer_blocks.0.proj_mlp": 32} {
		m := ad.Modules[name]
		if m == nil || m.B.Dim(0) != rows || m.A.Dim(1) != 8 {
			t.Errorf("%s: %+v", name, m)
		}
	}
	if ad.Modules["transformer_blocks.0.attn.to_q"].Alpha != 4 || ad.Modules["single_transformer_blocks.0.proj_mlp"].Alpha != 0 {
		t.Error("alpha not carried per module")
	}
}
