#!/usr/bin/env python3
"""Tiny QwenImage21Transformer2DModel reference (diffusers 0.42.1, CPU, F32)
for models/qwenimage21_test.go.

    .venv/bin/python scripts/verify/ref_qwenimage21.py <out_dir> <out.json>

A seeded tiny model (2 blocks, 2 heads of 16, axes 4/6/6, causal_condition)
run text-to-image: 7 text tokens of which 5 are valid (right padding masked
as keys), then a 4 x 6 target image block. Records the target tokens'
output.
"""
import json, os, shutil, sys
import torch
from diffusers import QwenImage21Transformer2DModel

out_dir, out_json = sys.argv[1], sys.argv[2]
torch.manual_seed(81)
m = QwenImage21Transformer2DModel(patch_size=1, in_channels=8, out_channels=8, num_layers=2, attention_head_dim=16,
                                  num_attention_heads=2, context_in_dim=24, mlp_ratio=3, axes_dims_rope=(4, 6, 6),
                                  eps=1e-6, causal_condition=True).eval()
with torch.no_grad():
    for name, p in m.named_parameters():
        if "norm" in name:
            p.copy_(0.2 * torch.randn_like(p) + (0 if "text_norm" in name else 1))
        else:
            p.normal_(0, 0.08)
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
m.save_pretrained(out_dir, safe_serialization=True)

L, valid, h, w = 7, 5, 4, 6
torch.manual_seed(82)
txt = torch.randn(1, L, 24)
img = torch.randn(1, h * w, 8)
img_mask = torch.tensor([[False] * L + [True] * (h * w // 4)])
txt_mask = torch.tensor([[True] * valid + [False] * (L - valid)])
with torch.no_grad():
    out = m(hidden_states=img, encoder_hidden_states=txt, timestep=torch.tensor([0.6]), img_shapes=[[(1, h, w)]],
            img_mask=img_mask, encoder_hidden_states_mask=txt_mask).sample
target = out[:, L:]
print("out", tuple(out.shape), "target", tuple(target.shape), target.abs().max().item())
f = lambda t: t.flatten().tolist()
json.dump({"txt": f(txt), "img": f(img), "L": L, "valid": valid, "h": h, "w": w, "timestep": 0.6,
           "target": f(target)}, open(out_json, "w"))
