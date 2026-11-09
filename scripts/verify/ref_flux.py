#!/usr/bin/env python3
"""Flux transformer reference (diffusers, CPU, F32) for models/flux_test.go.

    .venv/bin/python scripts/verify/ref_flux.py --make-tiny <dir> <out.json>

Writes a seeded tiny-random FluxTransformer2DModel (2 double + 2 single
blocks, guidance embedder, 3-axis RoPE) to <dir> with save_pretrained, runs
one forward pass on seeded inputs (8 text tokens, a 4 x 6 latent grid) and
writes the inputs and the output to <out.json>.
"""
import json, sys
import torch
from diffusers import FluxTransformer2DModel

assert sys.argv[1] == "--make-tiny"
out_dir, out_json = sys.argv[2], sys.argv[3]
torch.manual_seed(0)
cfg = dict(patch_size=1, in_channels=16, out_channels=None, num_layers=2, num_single_layers=2,
           attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
           pooled_projection_dim=24, guidance_embeds=True, axes_dims_rope=(4, 6, 6))
m = FluxTransformer2DModel(**cfg).eval()
with torch.no_grad():
    for name, p in m.named_parameters():
        if "norm" in name and name.endswith("weight"):
            p.add_(0.2 * torch.randn_like(p))  # RMSNorm weights start at 1
        else:
            p.normal_(0, 0.08)                   # incl. the zero-initialised AdaLN / out layers
m.save_pretrained(out_dir, safe_serialization=True)

B, St, H, W = 1, 8, 4, 6
x = torch.randn(B, H * W, 16)
txt = torch.randn(B, St, 32)
pooled = torch.randn(B, 24)
t = torch.tensor([0.7])
g = torch.tensor([3.5])
txt_ids = torch.zeros(St, 3)
img_ids = torch.zeros(H, W, 3)
img_ids[..., 1] = torch.arange(H)[:, None]
img_ids[..., 2] = torch.arange(W)[None, :]
img_ids = img_ids.reshape(H * W, 3)
with torch.no_grad():
    y = m(hidden_states=x, encoder_hidden_states=txt, pooled_projections=pooled, timestep=t,
          img_ids=img_ids, txt_ids=txt_ids, guidance=g, return_dict=False)[0]
print("out", tuple(y.shape), "mean abs", y.abs().mean().item())
r = lambda a: [round(v, 7) for v in a.flatten().tolist()]
json.dump({"x": r(x), "x_shape": list(x.shape), "txt": r(txt), "txt_shape": list(txt.shape),
           "pooled": r(pooled), "timestep": t.tolist(), "guidance": g.tolist(),
           "img_ids": r(img_ids), "txt_ids": r(txt_ids), "out": r(y), "out_shape": list(y.shape)},
          open(out_json, "w"))
