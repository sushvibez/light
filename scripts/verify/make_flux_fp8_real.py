#!/usr/bin/env python3
"""Casts a diffusers FLUX.1 transformer folder to a community-style FP8
single file: every 2-D weight float8_e4m3fn (no scales), the rest as is,
diffusers names. The 12B transformer becomes 12 GB.

    .venv/bin/python scripts/verify/make_flux_fp8_real.py black-forest-labs/FLUX.1-schnell \\
        ~/.cache/light/models/checkpoints/flux1-schnell-fp8.safetensors
"""
import glob, os, sys
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

repo, out = sys.argv[1], os.path.expanduser(sys.argv[2])
d = snapshot_download(repo, allow_patterns=["transformer/*"])
sd, n8, total = {}, 0, 0
for shard in sorted(glob.glob(os.path.join(d, "transformer", "*.safetensors"))):
    with safe_open(shard, "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            total += t.numel()
            if t.dim() == 2 and k.endswith(".weight"):
                t = t.to(torch.float8_e4m3fn)
                n8 += t.numel()
            sd[k] = t.contiguous()
save_file(sd, out, metadata={"format": "pt", "light": "fp8 weight-only cast of " + repo})
print("wrote", out, f"{n8 / 1e9:.2f}B of {total / 1e9:.2f}B params in fp8,", os.path.getsize(out) >> 30, "GB")
