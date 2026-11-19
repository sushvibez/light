#!/usr/bin/env python3
"""Edge0-35B-A3B-preview real-weight reference with mlx-lm (Colab, high-mem CPU).

    pip install "mlx[cpu]" mlx-lm
    hf download Edge0/Edge0-35B-A3B-preview --exclude "*.mp4" --exclude "*.jpg" \
        --exclude "prerouter*" --local-dir <model_dir>
    python scripts/colab/edge0_check.py <model_dir> <out_dir> [n_tokens]

mlx-lm's qwen3_5_moe on the MLX 4-bit checkpoint, every float parameter
(scales, biases, norms) widened to F32 so the reference computes in F32
like Light's F32 path. Two runs on a rendered chat prompt:

  base  the checkpoint as published (top-8 routing, no adapter)
  lora  lora_edge0_35b.safetensors applied the way Light applies it
        (models/mlx_load.go): y = base(x) + (alpha / r) (x A^T) B^T on each
        named linear (F32), and the routed width set to the adapter's K
        metadata (4)

Each run greedy-decodes n_tokens (default 20) and records the logits row
behind every token: row 0 is the prompt's last position, row j the
position of token j-1. <out_dir>/ref.json (ids, tokens, timings) and
<out_dir>/ref.safetensors (<run>_logits, n_tokens x vocab, F32) feed
models/edge0_real_test.go (LIGHT_EDGE0_REF=<out_dir>).
"""
import json, os, sys, time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from safetensors.numpy import save_file

model_dir, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
os.makedirs(out, exist_ok=True)

model, tok = load(model_dir)
float_types = (mx.bfloat16, mx.float16)
model.update(tree_map(lambda x: x.astype(mx.float32) if isinstance(x, mx.array) and x.dtype in float_types else x,
                      model.parameters()))
mx.eval(model.parameters())
print("dtypes", sorted({str(v.dtype) for _, v in tree_flatten(model.parameters())}), flush=True)

msgs = [{"role": "user", "content": "Explain in two short sentences why the sky looks blue during the day."}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
ids = tok.encode(prompt, add_special_tokens=False)
print("prompt", repr(prompt), len(ids), "tokens", flush=True)


def greedy(tag):
    cache = model.make_cache()
    rows, toks = [], []
    t0 = time.time()
    logits = model(mx.array([ids]), cache=cache)[0, -1].astype(mx.float32)
    mx.eval(logits)
    t_prefill = time.time() - t0
    t1 = time.time()
    for i in range(N):
        row = np.array(logits)
        rows.append(row)
        nxt = int(row.argmax())
        toks.append(nxt)
        if i == N - 1:
            break
        logits = model(mx.array([[nxt]]), cache=cache)[0, -1].astype(mx.float32)
        mx.eval(logits)
    dt = time.time() - t1
    top5 = np.argsort(-rows[0])[:5]
    print(f"{tag}: top-5 {top5.tolist()} {rows[0][top5].tolist()}", flush=True)
    print(f"{tag}: tokens {toks} {tok.decode(toks)!r}; prefill {t_prefill:.1f}s, decode {(N - 1) / dt:.2f} tok/s",
          flush=True)
    return np.stack(rows).astype(np.float32), toks, {"prefill_s": t_prefill, "decode_tok_s": (N - 1) / dt}


ref = {"model": os.path.abspath(model_dir), "prompt": prompt, "ids": ids, "n": N}
tensors = {}
tensors["base_logits"], ref["base_tokens"], ref["base_time"] = greedy("base")


class LoRALinear(nn.Module):
    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base, self.scale = base, scale
        self.a_t, self.b_t = a.astype(mx.float32).T, b.astype(mx.float32).T

    def __call__(self, x):
        return self.base(x) + self.scale * ((x.astype(mx.float32) @ self.a_t) @ self.b_t).astype(x.dtype)


def strip(name):  # models/mlx_load.go loraModule
    while True:
        for p in ("base_model.model.", "language_model.", "model."):
            if name.startswith(p):
                name = name[len(p):]
                break
        else:
            return name


lora_path = os.path.join(model_dir, "lora_edge0_35b.safetensors")
lw, meta = mx.load(lora_path, return_metadata=True)
meta = {**meta, **json.loads(meta.get("__metadata__", "{}"))}
scale, K = float(meta["alpha"]) / float(meta["r"]), int(meta.get("K", 0))
lm = model.language_model
mods = sorted({strip(k.rsplit(".", 1)[0]) for k in lw})
for mod in mods:
    full = [k for k in lw if strip(k.rsplit(".", 1)[0]) == mod]
    a = lw[next(k for k in full if k.endswith("lora_A"))]
    b = lw[next(k for k in full if k.endswith("lora_B"))]
    parent, path = (lm, ["lm_head"]) if mod == "lm_head" else (lm.model, mod.split("."))
    for p in path[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    leaf = path[-1]
    setattr(parent, leaf, LoRALinear(getattr(parent, leaf), a, b, scale))
if K:
    for layer in lm.model.layers:
        layer.mlp.top_k = K
print(f"lora: {len(mods)} modules, scale {scale}, top-{K}", flush=True)
ref["lora"] = {"modules": len(mods), "scale": scale, "top_k": K}
tensors["lora_logits"], ref["lora_tokens"], ref["lora_time"] = greedy("lora")

save_file(tensors, os.path.join(out, "ref.safetensors"))
json.dump(ref, open(os.path.join(out, "ref.json"), "w"), indent=1)
print("wrote", out)
