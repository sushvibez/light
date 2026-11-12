#!/usr/bin/env python3
"""End-to-end QwenImage21Pipeline text-to-image reference (diffusers 0.42.1
+ transformers, CPU, F32) for diffusion/qwenimage21_test.go.

    .venv/bin/python scripts/verify/ref_qwenimage21_pipe.py <out_dir> <out.json>

Builds a seeded tiny Qwen-Image 2.1 pipeline - Qwen-Image-2.1's processor
(tokenizer files only), a tiny Qwen3-VL text encoder (BF16 weights), a tiny
QwenImage21Transformer2DModel and a tiny 16x RGBA AutoencoderKLQwenImage21 -
with Qwen-Image-2.1's scheduler config, saves it, reloads it from disk and
runs it for 3 steps at 64 x 96 from explicit packed latents (no guidance).
"""
import json, os, shutil, sys
import torch
from diffusers import (AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline,
                       QwenImage21Transformer2DModel)
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

out_dir, out_json = sys.argv[1], sys.argv[2]
repo = "Qwen/Qwen-Image-2.1"
proc = AutoProcessor.from_pretrained(repo, subfolder="processor")
sched = FlowMatchEulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")
D, Z = 16, 8

torch.manual_seed(101)
te = Qwen3VLForConditionalGeneration(Qwen3VLConfig(
    text_config=dict(vocab_size=151936, hidden_size=D, intermediate_size=32, num_hidden_layers=3,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=8, rms_norm_eps=1e-6,
                     max_position_embeddings=1024, rope_theta=5000000.0,
                     rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True}),
    vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=D,
                       deepstack_visual_indexes=[0]),
    tie_word_embeddings=True)).eval()
with torch.no_grad():
    for name, p in te.named_parameters():
        p.copy_(1 + 0.2 * torch.randn_like(p)) if "norm" in name else p.normal_(0, 0.1)
tr = QwenImage21Transformer2DModel(patch_size=1, in_channels=Z, out_channels=Z, num_layers=2, attention_head_dim=16,
                                   num_attention_heads=2, context_in_dim=D, mlp_ratio=3, axes_dims_rope=(4, 6, 6),
                                   causal_condition=True).eval()
with torch.no_grad():
    for name, p in tr.named_parameters():
        if "norm" in name:
            p.copy_(0.2 * torch.randn_like(p) + (0 if "text_norm" in name else 1))
        else:
            p.normal_(0, 0.08)
torch.manual_seed(102)
vae = AutoencoderKLQwenImage21(base_dim=4, decoder_base_dim=6, z_dim=Z, dim_mult=[1, 2, 2, 2, 2], num_res_blocks=1,
                               temperal_downsample=[False, True, True, True], is_residual=True, in_channels=4,
                               out_channels=4, latents_mean=(0.1 * torch.randn(Z)).tolist(),
                               latents_std=(1 + 0.2 * torch.rand(Z)).tolist()).eval()
with torch.no_grad():
    for name, p in vae.named_parameters():
        p.copy_(1 + 0.2 * torch.randn_like(p)) if "gamma" in name else p.normal_(0, 0.15)

pipe = QwenImage21Pipeline(scheduler=sched, vae=vae, text_encoder=te, processor=proc, transformer=tr)
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
pipe.save_pretrained(out_dir, safe_serialization=True)
for f in os.listdir(os.path.join(out_dir, "text_encoder")):
    if f.endswith(".safetensors"):
        f = os.path.join(out_dir, "text_encoder", f)
        save_file({k: v.to(torch.bfloat16) for k, v in load_file(f).items()}, f, metadata={"format": "pt"})
pipe = QwenImage21Pipeline.from_pretrained(out_dir, torch_dtype=torch.float32)
shutil.rmtree(os.path.join(out_dir, "processor"))  # the Go test reads Qwen-Image-2.1's tokenizer

prompt, H, W, steps = "a red cube on a blue table, studio light", 64, 96, 3
h, w = H // 16, W // 16
torch.manual_seed(103)
packed = torch.randn(1, h * w, Z)
with torch.no_grad():
    embeds, emask, _ = pipe._get_qwen_prompt_embeds(prompt, device="cpu")
    img = pipe(prompt=prompt, height=H, width=W, num_inference_steps=steps, latents=packed, output_type="pt").images
sigmas = pipe.scheduler.sigmas.tolist()
text = pipe.prompt_template_t2i.format(prompt)
ids = proc.tokenizer(text).input_ids
unpacked = packed.transpose(1, 2).reshape(1, Z, h, w)
print("ids", len(ids), "drop", pipe._drop_idx, "embeds", tuple(embeds.shape), "image", tuple(img.shape), "sigmas", sigmas)
r = lambda a: [round(v, 7) for v in a.flatten().tolist()]
json.dump({"prompt": prompt, "height": H, "width": W, "steps": steps, "ids": ids, "embeds": r(embeds),
           "embeds_shape": list(embeds.shape), "sigmas": sigmas, "latents": r(unpacked),
           "latents_shape": list(unpacked.shape), "image": r(img), "image_shape": list(img.shape)},
          open(out_json, "w"))
