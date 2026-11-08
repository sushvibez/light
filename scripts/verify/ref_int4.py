#!/usr/bin/env python3
"""GPTQ / AWQ reference logits for models/int4_test.go, without GPU kernels.

    .venv/bin/python scripts/verify/ref_int4.py <repo> {gptq|awq} [--chat]

transformers runs GPTQ / AWQ checkpoints only through GPU kernel packages
(gptqmodel, autoawq). This script instead unpacks every 4-bit linear with
plain PyTorch, following the published layouts:

  GPTQ ("gptq" format): qweight I32 (K/8, N), nibble j of row r = row 8r+j;
        qzeros I32 (K/G, N/8) packed along N, holding zero - 1; scales (K/G, N);
        W[k, n] = s[g_idx[k], n] * (q[k, n] - zero[g_idx[k], n])
  AWQ (GEMM):  qweight / qzeros I32 (K, N/8) / (K/G, N/8) packed along N in
        the order [0, 2, 4, 6, 1, 3, 5, 7]; W = s * (q - z)

into an F32 Qwen2 / Llama model and runs transformers on it. Light loads the
packed checkpoint and dequantizes with its own code, so the two agree to
float rounding when both read the format the same way. (The layouts were
first checked against the BF16 originals: right unpacking gives relative
weight errors 0.16 / 0.31, a wrong nibble order or zero offset 0.46 / 1.49,
and the dequantized models write coherent text.)
"""
import json, sys, torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

repo, fmt = sys.argv[1], sys.argv[2]
chat = "--chat" in sys.argv


def unpack_rows(x):
    return torch.stack([(x >> (4 * j)) & 0xF for j in range(8)], 1).reshape(x.shape[0] * 8, x.shape[1])


def unpack_cols(x, order):
    nib = torch.stack([(x >> (4 * i)) & 0xF for i in range(8)], 2)
    out = torch.empty_like(nib)
    for i, o in enumerate(order):
        out[:, :, o] = nib[:, :, i]
    return out.reshape(x.shape[0], x.shape[1] * 8)


import os, tempfile
fetch = (lambda f: os.path.join(repo, f)) if os.path.isdir(repo) else (lambda f: hf_hub_download(repo, f))
cfg_raw = json.load(open(fetch("config.json")))
qc = cfg_raw["quantization_config"]
G = qc["group_size"]
v2 = qc.get("checkpoint_format") == "gptq_v2"
f = safe_open(fetch("model.safetensors"), "pt")
sd = {}
for n in f.keys():
    if n.endswith(".qweight"):
        p = n[: -len("qweight")]
        if fmt == "gptq":
            q = unpack_rows(f.get_tensor(n))
            z = unpack_cols(f.get_tensor(p + "qzeros"), range(8)) + (0 if v2 else 1)
            s = f.get_tensor(p + "scales").float()
            gi = f.get_tensor(p + "g_idx").long()
            W = (s[gi] * (q - z[gi]).float()).T
        else:
            o = [0, 2, 4, 6, 1, 3, 5, 7]
            q = unpack_cols(f.get_tensor(n), o)
            z = unpack_cols(f.get_tensor(p + "qzeros"), o)
            s = f.get_tensor(p + "scales").float()
            gk = torch.arange(q.shape[0]) // G
            W = (s[gk] * (q - z[gk]).float()).T
        sd[p + "weight"] = W.contiguous()
    elif not n.endswith((".qzeros", ".scales", ".g_idx")):
        sd[n] = f.get_tensor(n).float()

cfg = AutoConfig.from_pretrained(repo)
del cfg.quantization_config
probe = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
# Old AutoGPTQ checkpoints store a bias for every linear even where the
# architecture has none (Llama); those must be all zeros, and are dropped.
own = {k.replace(".experts.gate_up_proj", ".experts").replace(".experts.down_proj", ".experts") for k in probe.state_dict()}
for k in [k for k in sd if k.endswith(".bias") and k not in probe.state_dict()]:
    assert torch.count_nonzero(sd[k]) == 0, f"{k} is not zero"
    del sd[k]
del probe
# Save the dequantized weights as a plain checkpoint and load it with
# from_pretrained, so transformers applies its own key conversions (e.g.
# per-expert MoE weights into its fused expert tensors).
from safetensors.torch import save_file
tmp = tempfile.mkdtemp()
cfg.save_pretrained(tmp)
save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(tmp, "model.safetensors"), metadata={"format": "pt"})
model = AutoModelForCausalLM.from_pretrained(tmp, dtype=torch.float32, attn_implementation="eager").eval()

ids_arg = sys.argv[sys.argv.index("--ids") + 1] if "--ids" in sys.argv else None
tok = None if ids_arg else AutoTokenizer.from_pretrained(repo)
if ids_arg:
    ids = [int(x) for x in ids_arg.split(",")]
elif chat:
    msgs = [{"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Explain in one sentence why the sky is blue."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
else:
    ids = tok("The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is").input_ids
print("prompt ids:", ids)
x = torch.tensor([ids])
with torch.no_grad():
    logits = model(input_ids=x).logits[0, -1]
    topv, topi = logits.topk(5)
    print("F32 top5:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  id={i:<7} {(tok.decode([i]) if tok else '')!r:<14} logit={v:.4f}")
    out = model.generate(x, max_new_tokens=16 if tok else 8, do_sample=False)
    if tok:
        print("greedy:", repr(tok.decode(out[0, len(ids):])))
    print("greedy ids:", out[0, len(ids):].tolist())
