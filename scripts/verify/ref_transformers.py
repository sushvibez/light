#!/usr/bin/env python3
"""Golden reference using HuggingFace transformers (the standard third-party impl).

Loads a real HF checkpoint and prints the top-k next-token predictions for a
prompt, so the Go forward (cmd/hfvalidate) can be compared against it.

Usage:
    python ref_transformers.py <model_dir_or_hf_id> "The capital of France is"
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_id = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    ids = tok(prompt, return_tensors="pt").input_ids
    print(f"prompt {prompt!r} -> {ids.tolist()[0]}")
    with torch.no_grad():
        logits = model(ids).logits[0, -1]  # last position
    topv, topi = logits.topk(5)
    print("REFERENCE top5 (transformers):")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  id={i:<7} {tok.decode([i])!r:<12} logit={v:.4f}")

if __name__ == "__main__":
    main()
