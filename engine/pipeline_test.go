package engine

import (
	"os"
	"path/filepath"
	"testing"

	"light/device"
	"light/dtype"
	"light/safetensors"
	"light/tensor"
)

// The library lists safetensors by folder and reads what the keys say.
func TestLibraryFiles(t *testing.T) {
	root := t.TempDir()
	t.Setenv("LIGHT_MODELS_DIR", root)
	write := func(kind, name string, keys map[string]dtype.DType) {
		ts := map[string]*tensor.Tensor{}
		for k, dt := range keys {
			x, err := tensor.Zeros(dt, []int{2, 2}, device.CPU())
			if err != nil {
				t.Fatal(err)
			}
			ts[k] = x
		}
		if err := os.MkdirAll(filepath.Join(root, kind), 0o755); err != nil {
			t.Fatal(err)
		}
		f, err := os.Create(filepath.Join(root, kind, name))
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		if err := safetensors.Write(f, ts, nil); err != nil {
			t.Fatal(err)
		}
	}
	write("checkpoints", "flux1-schnell.safetensors", map[string]dtype.DType{"double_blocks.0.img_attn.qkv.weight": dtype.BF16, "single_blocks.0.linear1.weight": dtype.BF16, "img_in.weight": dtype.F32})
	write("checkpoints", "flux-diffusers.safetensors", map[string]dtype.DType{"transformer_blocks.0.attn.add_q_proj.weight": dtype.BF16, "time_text_embed.text_embedder.linear_1.weight": dtype.BF16})
	write("loras", "style.safetensors", map[string]dtype.DType{"transformer.single_transformer_blocks.3.attn.to_q.lora_A.weight": dtype.F16, "transformer.single_transformer_blocks.3.attn.to_q.lora_B.weight": dtype.F16})
	write("loras", "kohya.safetensors", map[string]dtype.DType{"lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": dtype.F16})
	write("vae", "ae.safetensors", map[string]dtype.DType{"decoder.up.0.block.0.conv1.weight": dtype.F32, "encoder.down.0.block.0.conv1.weight": dtype.F32})
	write("clip", "clip_l.safetensors", map[string]dtype.DType{"text_model.encoder.layers.0.self_attn.q_proj.weight": dtype.F16})
	write("clip", "t5xxl.safetensors", map[string]dtype.DType{"encoder.block.0.layer.0.SelfAttention.q.weight": dtype.BF16})
	os.WriteFile(filepath.Join(root, "clip", "notes.txt"), []byte("x"), 0o644)

	want := map[string][3]string{ // name → dtype, family, layout
		"flux1-schnell.safetensors":  {"bf16", "flux", "bfl"},
		"flux-diffusers.safetensors": {"bf16", "flux", "diffusers"},
		"style.safetensors":          {"f16", "lora", "peft"},
		"kohya.safetensors":          {"f16", "lora", "kohya"},
		"ae.safetensors":             {"f32", "vae", ""},
		"clip_l.safetensors":         {"f16", "clip", ""},
		"t5xxl.safetensors":          {"bf16", "t5", ""},
	}
	seen := 0
	for _, kind := range LibraryKinds {
		files, err := LibraryFiles(kind)
		if err != nil {
			t.Fatal(err)
		}
		for _, f := range files {
			w, ok := want[f.Name]
			if !ok {
				t.Errorf("%s/%s: unexpected", kind, f.Name)
				continue
			}
			seen++
			if got := [3]string{f.DType, f.Family, f.Layout}; got != w {
				t.Errorf("%s: got %v, want %v", f.Name, got, w)
			}
			if f.Format != "safetensors" || f.Bytes == 0 {
				t.Errorf("%s: format %q, %d bytes", f.Name, f.Format, f.Bytes)
			}
		}
	}
	if seen != len(want) {
		t.Errorf("saw %d files, want %d", seen, len(want))
	}
	if _, err := LibraryFiles("unet"); err == nil {
		t.Error("unknown kind accepted")
	}
	if files, err := LibraryFiles("checkpoints"); err != nil || files == nil {
		t.Errorf("missing folder: %v, %v", files, err)
	}
}

// A spec with every slot from the same repo resolves to that repo's folders.
func TestFluxComponentsFromRepo(t *testing.T) {
	dir := t.TempDir()
	for _, sub := range []string{"transformer", "vae", "text_encoder", "text_encoder_2"} {
		os.MkdirAll(filepath.Join(dir, sub), 0o755)
	}
	comp, err := fluxComponents(ImageOptions{Repo: dir, Pipeline: &PipelineSpec{Family: "flux", VAE: PipelineSlot{Path: filepath.Join(dir, "vae")}}})
	if err != nil {
		t.Fatal(err)
	}
	for _, sub := range []string{"transformer", "vae", "text_encoder", "text_encoder_2"} {
		if comp[sub] != filepath.Join(dir, sub) {
			t.Errorf("%s: %q", sub, comp[sub])
		}
	}
	f := filepath.Join(dir, "flux1.safetensors")
	os.WriteFile(f, []byte("x"), 0o644)
	if _, err := fluxComponents(ImageOptions{Repo: dir, Pipeline: &PipelineSpec{Transformer: PipelineSlot{Path: f}}}); err == nil {
		t.Error("single file accepted")
	}
}
