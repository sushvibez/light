#!/usr/bin/env python3
"""Tiny Flux3Transformer2DModel reference (diffusers, CPU, F32) for
models/flux3_test.go.

    .venv/bin/python scripts/verify/ref_flux3.py <out_dir> <out.json>

Builds a seeded tiny flow-map Flux3Transformer2DModel (2 per-mode layers,
2 single layers, 2 heads of 16, axes 4/4/4/4), saves it to <out_dir> and
records two forwards: text-to-image (image + text tokens) and editing (with
reference image tokens), each with a timestep and a target timestep.
"""
import json, os, shutil, sys
import torch
from diffusers import Flux3Transformer2DModel

out_dir, out_json = sys.argv[1], sys.argv[2]
torch.manual_seed(51)
m = Flux3Transformer2DModel(in_channels=8, cond_in_channels=8, num_layers=2, num_single_layers=2,
                            attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
                            timestep_channels=256, mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4), rope_theta=10000,
                            eps=1e-6, flow_map=True).eval()
with torch.no_grad():
    for name, p in m.named_parameters():
        if "norm" in name:
            p.copy_(1 + 0.2 * torch.randn_like(p))
        else:
            p.normal_(0, 0.08)
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
m.save_pretrained(out_dir, safe_serialization=True)

Si, St, Sc = 12, 5, 6
torch.manual_seed(52)
img = torch.randn(1, Si, 8)
txt = torch.randn(1, St, 24)
cond = torch.randn(1, Sc, 8)


def ids(n, t, h0, w0, w):
    return torch.tensor([[t, h0 + i // w, w0 + i % w, 0] for i in range(n)], dtype=torch.float32)


img_ids = ids(Si, 0, 0, 0, 4)
cond_ids = ids(Sc, 10, 0, 0, 3)
txt_ids = torch.tensor([[0, 0, 0, l] for l in range(St)], dtype=torch.float32)
ts, tgt = torch.tensor([0.7]), torch.tensor([0.4])
with torch.no_grad():
    t2i = m(hidden_states=img, encoder_hidden_states=txt, timestep=ts, img_ids=img_ids, txt_ids=txt_ids,
            target_timestep=tgt).sample
    edit = m(hidden_states=img, encoder_hidden_states=txt, timestep=ts, img_ids=img_ids, txt_ids=txt_ids,
             image_cond_hidden_states=cond, image_cond_ids=cond_ids, target_timestep=tgt).sample
print("t2i", tuple(t2i.shape), t2i.abs().max().item(), "edit", edit.abs().max().item())
f = lambda t: t.flatten().tolist()
json.dump({"img": f(img), "txt": f(txt), "cond": f(cond), "img_ids": f(img_ids), "txt_ids": f(txt_ids),
           "cond_ids": f(cond_ids), "timestep": ts.item(), "target": tgt.item(), "si": Si, "st": St, "sc": Sc,
           "t2i": f(t2i), "edit": f(edit)}, open(out_json, "w"))
