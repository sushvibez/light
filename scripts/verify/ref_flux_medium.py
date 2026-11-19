#!/usr/bin/env python3
"""A random FluxTransformer2DModel with FLUX.1's real geometry (128-wide
heads, rope axes 16 / 56 / 56, 1024 image + 256 text tokens, guidance
embedder) but two heads and one block of each kind, for
models/flux_medium_test.go: its weights (diffusers layout), inputs and
output.

    .venv/bin/python scripts/verify/ref_flux_medium.py models/testdata/flux-medium
"""
import json, os, sys
import torch
from diffusers import FluxTransformer2DModel
from safetensors.torch import save_file

out = sys.argv[1]
heads = int(sys.argv[2]) if len(sys.argv) > 2 else 2        # 24: FLUX.1's width
singles = int(sys.argv[3]) if len(sys.argv) > 3 else 1
os.makedirs(out, exist_ok=True)
torch.manual_seed(5)
cfg = dict(patch_size=1, in_channels=64, num_layers=1, num_single_layers=singles, attention_head_dim=128,
           num_attention_heads=heads, joint_attention_dim=96, pooled_projection_dim=40, guidance_embeds=True,
           axes_dims_rope=(16, 56, 56))
m = FluxTransformer2DModel(**cfg).eval()
with torch.no_grad():
    for p in m.parameters():
        p.mul_(0.5)  # tame the random init
sd = {k: v.contiguous() for k, v in m.state_dict().items()}
save_file(sd, os.path.join(out, "diffusion_pytorch_model.safetensors"))
json.dump({**cfg, "_class_name": "FluxTransformer2DModel"}, open(os.path.join(out, "config.json"), "w"))

H = W = 32  # 512² at vae scale 8, packed 2x2
S, St = H * W, 256
x = torch.randn(1, S, 64)
txt = torch.randn(1, St, 96)
pooled = torch.randn(1, 40)
img_ids = torch.zeros(S, 3)
img_ids[:, 1] = torch.arange(H).repeat_interleave(W)
img_ids[:, 2] = torch.arange(W).repeat(H)
txt_ids = torch.zeros(St, 3)
t = torch.tensor([0.75])
g = torch.tensor([3.5])
with torch.no_grad():
    y = m(hidden_states=x, encoder_hidden_states=txt, pooled_projections=pooled, timestep=t, guidance=g,
          img_ids=img_ids, txt_ids=txt_ids, return_dict=False)[0]
r = lambda a: [round(v, 7) for v in a.flatten().tolist()]
json.dump({"image": r(x), "text": r(txt), "pooled": r(pooled), "timestep": 0.75, "guidance": 3.5,
           "img_ids": r(img_ids), "txt_ids": r(txt_ids), "S": S, "St": St, "out": r(y), "out_shape": list(y.shape)},
          open(os.path.join(out, "forward.json"), "w"))
print("wrote", out, "out", tuple(y.shape), "mean |y|", y.abs().mean().item())
