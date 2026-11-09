#!/usr/bin/env python3
"""End-to-end FluxPipeline reference (diffusers, CPU, F32) for
diffusion/flux_test.go.

    .venv/bin/python scripts/verify/ref_flux_pipe.py <out_dir> <out.json>

Copies hf-internal-testing/tiny-flux-pipe to <out_dir> with its VAE
replaced by a seeded tiny AutoencoderKL that uses 32 GroupNorm groups (the
tiny pipe's VAE uses 1, which Light's VAE blocks do not take), then runs
FluxPipeline for 2 steps at 32 x 32 with explicit packed latents and writes
the token ids, text embeddings, sigmas and the final image (output_type
"pt", in [0, 1]) to <out.json>.
"""
import json, os, shutil, sys
import torch
from diffusers import AutoencoderKL, FluxPipeline
from huggingface_hub import snapshot_download

out_dir, out_json = sys.argv[1], sys.argv[2]
src = snapshot_download("hf-internal-testing/tiny-flux-pipe")
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
shutil.copytree(src, out_dir, ignore=shutil.ignore_patterns("README.md", ".gitattributes"))
torch.manual_seed(2)
vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=1, block_out_channels=(32,),
                    down_block_types=("DownEncoderBlock2D",), up_block_types=("UpDecoderBlock2D",),
                    layers_per_block=1, norm_num_groups=32, use_quant_conv=False, use_post_quant_conv=False,
                    scaling_factor=1.5035, shift_factor=0.0609).eval()
with torch.no_grad():
    for p in vae.parameters():
        p.add_(0.05 * torch.randn_like(p))
shutil.rmtree(os.path.join(out_dir, "vae"))
vae.save_pretrained(os.path.join(out_dir, "vae"), safe_serialization=True)
from safetensors.torch import load_file, save_file
f = os.path.join(out_dir, "vae", "diffusion_pytorch_model.safetensors")
save_file({k: v for k, v in load_file(f).items() if k.startswith("decoder.")}, f)

pipe = FluxPipeline.from_pretrained(out_dir, torch_dtype=torch.float32)
pipe.vae = vae
prompt, H, W, steps, max_len = "a red cube on a blue table", 32, 32, 2, 16
torch.manual_seed(3)
latents = torch.randn(1, (H // 2) * (W // 2), 4)  # packed: vae scale 1, 1 latent channel
clip_ids = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True).input_ids
t5_ids = pipe.tokenizer_2(prompt, padding="max_length", max_length=max_len, truncation=True).input_ids
with torch.no_grad():
    pe, pooled, _ = pipe.encode_prompt(prompt, None, max_sequence_length=max_len)
    img = pipe(prompt, height=H, width=W, num_inference_steps=steps, max_sequence_length=max_len,
               latents=latents, output_type="pt").images
print("image", tuple(img.shape), "mean", img.mean().item(), "sigmas", pipe.scheduler.sigmas.tolist())
r = lambda a: [round(v, 7) for v in a.flatten().tolist()]
json.dump({"prompt": prompt, "height": H, "width": W, "steps": steps, "max_seq_len": max_len,
           "clip_ids": clip_ids, "t5_ids": t5_ids, "latents": r(latents), "latents_shape": list(latents.shape),
           "prompt_embeds": r(pe), "pooled": r(pooled), "sigmas": pipe.scheduler.sigmas.tolist(),
           "image": r(img), "image_shape": list(img.shape)}, open(out_json, "w"))
