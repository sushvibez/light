#!/usr/bin/env python3
"""Tiny AutoencoderKLQwenImage21 decode reference (diffusers 0.42.1, CPU,
F32) for models/qwenimage_vae_test.go.

    .venv/bin/python scripts/verify/ref_qwenimage_vae.py <out_dir> <out.json>

A seeded tiny residual VAE (decoder base 12, dim_mult 1/2/2 with one
temporal upsample, patch size 2) decodes one frame of 4 x 5 latents.
"""
import json, os, shutil, sys
import torch
from diffusers import AutoencoderKLQwenImage21

out_dir, out_json = sys.argv[1], sys.argv[2]
torch.manual_seed(91)
z = 4
v = AutoencoderKLQwenImage21(base_dim=8, decoder_base_dim=12, z_dim=z, dim_mult=[1, 2, 2], num_res_blocks=1,
                             temperal_downsample=[False, True], is_residual=True, in_channels=12, out_channels=12,
                             patch_size=2, latents_mean=[0.0] * z, latents_std=[1.0] * z).eval()
with torch.no_grad():
    for name, p in v.named_parameters():
        if "gamma" in name:
            p.copy_(1 + 0.2 * torch.randn_like(p))
        else:
            p.normal_(0, 0.15)
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
v.save_pretrained(out_dir, safe_serialization=True)
torch.manual_seed(92)
lat = torch.randn(1, z, 1, 4, 5)
with torch.no_grad():
    img = v.decode(lat).sample  # (1, 3, 1, H, W)
print("image", tuple(img.shape), img.abs().max().item())
json.dump({"latents": lat.flatten().tolist(), "z": z, "h": 4, "w": 5, "image": img.flatten().tolist(),
           "image_shape": [1, 3, img.shape[3], img.shape[4]]}, open(out_json, "w"))
