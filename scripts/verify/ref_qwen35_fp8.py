#!/usr/bin/env python3
"""Tiny fine-grained FP8 Qwen3.5 fixture and its transformers reference.

    python scripts/verify/ref_qwen35_fp8.py --make models/testdata/qwen35/tiny models/testdata/qwen35/tiny-fp8
    python scripts/verify/ref_qwen35_fp8.py models/testdata/qwen35/tiny-fp8

--make quantizes the tiny Qwen3.5 checkpoint into the format of
Qwen/Qwen3.5-27B-FP8: every projection an F8E4M3 weight plus an F32
`weight_scale_inv` per block (here 16 x 16, so the tiny dims hold several
blocks; the real checkpoints use 128 x 128), scale = block amax / 448, and
the same modules_to_not_convert families as the real config (embedding,
LM head, linear_attn.in_proj_a / in_proj_b; conv1d is not a Linear).

Without --make: loads the fixture with transformers' own FP8 support (on a
CPU, or a GPU below sm_89, that dequantizes on load: w * weight_scale_inv
per block), in F32, and prints the top-5 last-position logits and the
greedy continuation for the 70-token id sequence of qwen3_5_test.go.
"""
import json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file

if sys.argv[1] == "--make":
    src, dst = sys.argv[2], sys.argv[3]
    block = 16
    sd = load_file(os.path.join(src, "model.safetensors"))
    cfg = json.load(open(os.path.join(src, "config.json")))
    skip_suffix = ("embed_tokens.weight", "lm_head.weight", "in_proj_a.weight", "in_proj_b.weight")
    out, not_convert = {}, []
    for name, w in sd.items():
        quantize = (w.ndim == 2 and name.endswith(".weight") and "norm" not in name
                    and not name.endswith(skip_suffix) and "conv1d" not in name)
        if not quantize:
            out[name] = w
            if name.endswith(skip_suffix):
                not_convert.append(name[: -len(".weight")])
            continue
        R, C = w.shape
        assert R % block == 0 and C % block == 0, (name, w.shape)
        wb = w.float().reshape(R // block, block, C // block, block)
        amax = wb.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
        scale = amax / 448.0
        q = (wb / scale).clamp(-448, 448).reshape(R, C).to(torch.float8_e4m3fn)
        out[name] = q
        out[name + "_scale_inv"] = scale.reshape(R // block, C // block).contiguous()
    cfg["quantization_config"] = {
        "quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [block, block],
        "modules_to_not_convert": sorted(not_convert),
    }
    os.makedirs(dst, exist_ok=True)
    save_file(out, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    for f in ("generation_config.json",):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), dst)
    print("wrote", dst, f"({sum(1 for n in out if n.endswith('_scale_inv'))} FP8 weights)")
    sys.exit(0)

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.float32, attn_implementation="eager").eval()
qkv = model.model.layers[0].linear_attn.in_proj_qkv.weight
print("model class:", type(model).__name__, "in_proj_qkv", qkv.dtype)
ids = [(37 * i + 11) % 512 for i in range(70)]
x = torch.tensor([ids])
with torch.no_grad():
    logits = model(input_ids=x).logits[0, -1].float()
    topv, topi = logits.topk(5)
    print("top5:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  id={i:<7} logit={v:.4f}")
    gen = model.generate(x, max_new_tokens=8, do_sample=False)
    print("greedy ids:", gen[0, len(ids):].tolist())
