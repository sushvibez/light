#!/usr/bin/env python3
"""Sliding-window attention reference: Qwen2-0.5B-Instruct with a tiny window.

Transformers applies the window to layers >= max_window_layers when
use_sliding_window is set, so window=8 / max_window_layers=0 exercises every
layer on a 15-token prompt. Prints top-5 of the last position (F32, eager).

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify/ref_swa.py 8
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoConfig

window = int(sys.argv[1]) if len(sys.argv) > 1 else 8
repo = "Qwen/Qwen2-0.5B-Instruct"
cfg = AutoConfig.from_pretrained(repo)
cfg.use_sliding_window = True
cfg.sliding_window = window
cfg.max_window_layers = 0
if hasattr(cfg, "layer_types"):
    cfg.layer_types = ["sliding_attention"] * cfg.num_hidden_layers
model = AutoModelForCausalLM.from_pretrained(repo, config=cfg, torch_dtype=torch.float32, attn_implementation="eager").eval()
ids = [151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198]
with torch.no_grad():
    full = model(torch.tensor([ids])).logits[0, -1]
topv, topi = full.topk(5)
print("window", window, "layer_types", getattr(cfg, "layer_types", None)[:2] if hasattr(cfg, "layer_types") else None)
for v, i in zip(topv.tolist(), topi.tolist()):
    print(f"  id={i:<7} logit={v:.4f}")
