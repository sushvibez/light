#!/usr/bin/env python3
"""Qwen3-MoE reference logits (transformers) for models/qwen3_moe_test.go.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify/ref_qwen3_moe.py [repo] [f32|bf16] [cpu|cuda]

Default repo is the tiny random hf-internal-testing checkpoint (fused
`experts.gate_up_proj` / `experts.down_proj` layout, router stored as
`mlp.router.weight`). transformers 5.21 names the router `mlp.gate.weight`
and would otherwise re-initialise it at random, so the router weights are
copied in from the safetensors file explicitly before running.
"""
import os, sys, torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

repo = sys.argv[1] if len(sys.argv) > 1 else "hf-internal-testing/tiny-random-Qwen3MoeForCausalLM"
dt = torch.bfloat16 if (len(sys.argv) > 2 and sys.argv[2] == "bf16") else torch.float32
dev = sys.argv[3] if len(sys.argv) > 3 else "cpu"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=dt, attn_implementation="eager").to(dev).eval()
try:
    path = hf_hub_download(repo, "model.safetensors")
    with safe_open(path, "pt") as f:
        for i, layer in enumerate(model.model.layers):
            k = f"model.layers.{i}.mlp.router.weight"
            if k in f.keys() and hasattr(layer.mlp, "gate") and hasattr(layer.mlp.gate, "weight"):
                with torch.no_grad():
                    layer.mlp.gate.weight.copy_(f.get_tensor(k).to(dt).to(dev))
except Exception as e:  # sharded hub checkpoints already use mlp.gate.weight
    print("router patch skipped:", e)
prompt = "The quick brown fox jumps over the lazy dog because it wanted to reach the other side of the river"
ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
print("ids", ids.tolist()[0])
with torch.no_grad():
    logits = model(ids).logits[0].float()
for pos in [0, ids.shape[1] - 1]:
    v, i = logits[pos].topk(5)
    print(f"dtype={dt} pos={pos} top5", [(int(a), round(float(b), 5)) for a, b in zip(i, v)])
