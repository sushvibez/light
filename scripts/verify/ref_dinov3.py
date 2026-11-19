#!/usr/bin/env python3
"""Tiny DINOv3 ViT references (transformers DINOv3ViTModel, CPU, F32) for
models/dinov3_test.go: a gated (SwiGLU, as ViT-H+) and a plain-MLP model,
seeded and saved with save_pretrained, run on a 48 x 64 image.

    python scripts/verify/ref_dinov3.py <out_dir> <out.json>
"""
import json, os, sys

import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

out_dir, out_json = sys.argv[1], sys.argv[2]
r = lambda t: [round(v, 7) for v in t.detach().flatten().tolist()]
out = {}
for name, gated, act in (("gated", True, "silu"), ("plain", False, "gelu")):
    torch.manual_seed(5 if gated else 6)
    cfg = DINOv3ViTConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=192,
                          patch_size=16, num_register_tokens=4, use_gated_mlp=gated, hidden_act=act,
                          layerscale_value=0.5, rope_theta=100.0, image_size=64, query_bias=True,
                          key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True)
    m = DINOv3ViTModel(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lambda1" in n:
                p.copy_(0.5 + 0.2 * torch.randn_like(p))
            elif "norm" in n and n.endswith("weight"):
                p.copy_(1 + 0.2 * torch.randn_like(p))
            else:
                p.normal_(0, 0.1)
    d = os.path.join(out_dir, name)
    m.save_pretrained(d, safe_serialization=True)
    m = DINOv3ViTModel.from_pretrained(d).eval()
    x = torch.rand(1, 3, 48, 64)
    with torch.no_grad():
        h = m(pixel_values=x).last_hidden_state
    out[name] = {"pixels": r(x), "hidden": r(h), "shape": list(h.shape),
                 "idol_tokens": r(torch.nn.functional.layer_norm(h, h.shape[-1:]))}
    print(name, tuple(h.shape))
json.dump(out, open(out_json, "w"))
