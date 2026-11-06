#!/usr/bin/env python3
"""Gemma 3 reference logits (transformers, eager) for the Go port.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify/ref_gemma.py [repo] [f32|bf16]

Runs the chat prompt "What is 2+2?" (Gemma template, 16 ids incl. <bos>) on
CUDA and prints the top-5 next-token logits at the last position, plus the
greedy continuation, so models/gemma_test.go can pin them.
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = sys.argv[1] if len(sys.argv) > 1 else "unsloth/gemma-3-1b-it"
dt = torch.bfloat16 if (len(sys.argv) > 2 and sys.argv[2] == "bf16") else torch.float32
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dt, attn_implementation="eager").to("cuda").eval()
# Gemma 3 tokenizer ids of the prompt; Gemma 2's vocabulary differs, so
# re-tokenize there (printed for the Go test to pin).
ids = tok.apply_chat_template([{"role": "user", "content": "What is 2+2?"}], add_generation_prompt=True, tokenize=True, return_dict=False)
print("prompt ids:", list(ids))
x = torch.tensor([ids], device="cuda")
with torch.no_grad():
    logits = model(x).logits[0, -1].float()
    topv, topi = logits.topk(5)
    print(f"dtype={dt} top5:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  id={i:<7} {tok.decode([i])!r:<14} logit={v:.4f}")
    out = model.generate(x, max_new_tokens=12, do_sample=False)
    print("greedy:", repr(tok.decode(out[0, len(ids):], skip_special_tokens=False)))
    print("greedy ids:", out[0, len(ids):].tolist())
