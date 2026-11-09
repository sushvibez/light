#!/usr/bin/env python3
"""Portable transformers reference for models too large for the local card.

Runs a causal LM on a chat (or raw) prompt and writes everything a Light
test needs to verify a quantized GGUF of the same checkpoint into a JSON
file: the token ids (so the test needs no tokenizer), the top-k next-token
logits at the last position and the greedy continuation. Meant to run on
Google Colab (scripts/colab/light_reference.ipynb) for 8B–30B checkpoints;
works the same locally.

    python scripts/verify/ref_json.py meta-llama/Llama-3.1-8B-Instruct \
        --dtype bf16 --gguf 'llama-3.1-8b-instruct-q4_k_m.gguf' \
        --out models/testdata/refs/llama-3.1-8b-instruct.json

--dtype: bf16 | fp16 | f32 | 8bit | 4bit (bitsandbytes; the quantized
loaders change logits by ~0.1–0.5, raise --tolerance accordingly).
--gguf: glob of the local GGUF file name(s) the Go test should compare
(models/heavy_ref_test.go looks in ~/.cache/light/gguf or LIGHT_GGUF_DIR).
"""
import argparse
import json
import platform
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--prompt", default="What is 2+2?")
    ap.add_argument("--raw", action="store_true", help="feed the prompt as plain text, no chat template")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "f32", "8bit", "4bit"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--gguf", default="", help="glob of the GGUF file name(s) to verify against this reference")
    ap.add_argument("--tolerance", type=float, default=0, help="allowed |logit - ref| in the Go test (0: pick by quant level)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.repo, revision=a.revision)
    kw = dict(revision=a.revision, attn_implementation="eager")
    if a.dtype in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=a.dtype == "8bit", load_in_4bit=a.dtype == "4bit",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = "auto"
    else:
        kw["dtype"] = {"bf16": torch.bfloat16, "fp16": torch.float16, "f32": torch.float32}[a.dtype]
        kw["device_map"] = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.repo, **kw).eval()
    dev = next(model.parameters()).device

    if a.raw or tok.chat_template is None:
        ids = tok(a.prompt, return_tensors="pt").input_ids[0].tolist()
        chat = False
    else:
        ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True,
                                      tokenize=True, return_dict=False)
        chat = True
    x = torch.tensor([ids], device=dev)
    with torch.no_grad():
        logits = model(input_ids=x).logits[0, -1].float()
        topv, topi = logits.topk(a.topk)
        out = model.generate(x, max_new_tokens=a.max_new, do_sample=False)
    greedy = out[0, len(ids):].tolist()
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None) or cfg
    ref = {
        "repo": a.repo,
        "revision": a.revision,
        "model_type": getattr(cfg, "model_type", ""),
        "vocab_size": int(getattr(text_cfg, "vocab_size", 0)),
        "dtype": a.dtype,
        "prompt": a.prompt,
        "chat_template": chat,
        "ids": ids,
        "top": [{"id": int(i), "logit": round(float(v), 4), "text": tok.decode([i])} for v, i in zip(topv.tolist(), topi.tolist())],
        "greedy": greedy,
        "greedy_text": tok.decode(greedy, skip_special_tokens=False),
        "gguf": a.gguf,
        "tolerance": a.tolerance,
        "source": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "device": str(dev) + (" " + torch.cuda.get_device_name(0) if dev.type == "cuda" else ""),
            "python": platform.python_version(),
            "load_s": round(time.time() - t0, 1),
        },
    }
    with open(a.out, "w") as f:
        json.dump(ref, f, indent=1)
    print(f"{a.repo} ({a.dtype}) on {ref['source']['device']}: {len(ids)} prompt ids")
    for t in ref["top"]:
        print(f"  id={t['id']:<8} {t['text']!r:<14} logit={t['logit']:.4f}")
    print("greedy:", repr(ref["greedy_text"]))
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
