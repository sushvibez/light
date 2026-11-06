#!/usr/bin/env python3
"""Phi-4-mini reference logits (transformers, eager) for models/phi4_mini_test.go.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify/ref_phi4.py [repo] [bf16|f32]

Runs the chat prompt through the tokenizer's chat template on CUDA; prints
the prompt ids, the top-5 next-token logits at the last position, the
greedy continuation and its ids.
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = sys.argv[1] if len(sys.argv) > 1 else "microsoft/Phi-4-mini-instruct"
dt = torch.float32 if (len(sys.argv) > 2 and sys.argv[2] == "f32") else torch.bfloat16
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=dt, attn_implementation="eager").to("cuda").eval()
msgs = [{"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain in one sentence why the sky is blue."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
print("prompt:", repr(tok.decode(ids)))
print("prompt ids:", list(ids))
x = torch.tensor([ids], device="cuda")
with torch.no_grad():
    logits = model(input_ids=x).logits[0, -1].float()
    topv, topi = logits.topk(5)
    print(f"dtype={dt} top5:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  id={i:<7} {tok.decode([i])!r:<14} logit={v:.4f}")
    out = model.generate(x, max_new_tokens=16, do_sample=False)
    print("greedy:", repr(tok.decode(out[0, len(ids):], skip_special_tokens=False)))
    print("greedy ids:", out[0, len(ids):].tolist())
