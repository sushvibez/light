#!/usr/bin/env python3
"""Casts a diffusers FLUX.2 transformer folder to a weight-only FP8 folder
(2-D weights float8_e4m3fn, config.json copied), loadable in place of
the repo's transformer (diffusion.Flux2LoadOptions.Components).

    .venv/bin/python scripts/verify/make_flux2_fp8_real.py black-forest-labs/FLUX.2-klein-4B \\
        ~/.cache/light/models/checkpoints/flux2-klein-4b-fp8
"""
import glob, os, shutil, sys
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

repo, out = sys.argv[1], os.path.expanduser(sys.argv[2])
d = snapshot_download(repo, allow_patterns=["transformer/*"])
os.makedirs(out, exist_ok=True)
shutil.copy(os.path.join(d, "transformer", "config.json"), os.path.join(out, "config.json"))
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
p = os.path.join(out, "diffusion_pytorch_model.safetensors")
save_file(sd, p, metadata={"format": "pt", "light": "fp8 weight-only cast of " + repo})
print("wrote", p, f"{n8 / 1e9:.2f}B of {total / 1e9:.2f}B params in fp8,", os.path.getsize(p) >> 20, "MB")
