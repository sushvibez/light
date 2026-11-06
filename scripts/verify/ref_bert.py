#!/usr/bin/env python
"""Reference values for models/bert_bge_test.go.

Loads a BERT checkpoint with transformers.AutoModel, tokenizes the fixed
sentences below with [CLS]/[SEP] + padding to the longest, and prints:
  * token ids per sentence (so the Go WordPiece output can be cross-checked)
  * last_hidden_state[0, 0, :8]
  * CLS-pooled, L2-normalised embedding (first 8 dims) per sentence
  * the pairwise cosine matrix

Run offline against the HF cache:
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify/ref_bert.py [repo]
"""
import sys

import torch
from transformers import AutoModel, AutoTokenizer

REPO = sys.argv[1] if len(sys.argv) > 1 else "unsloth/bge-small-en-v1.5"
SENTENCES = [
    "The cat sits on the mat.",
    "A feline rests on a rug.",
    "Stock markets fell sharply today.",
    "Go is a statically typed language.",
]


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModel.from_pretrained(REPO, torch_dtype=torch.float32).eval()
    enc = tok(SENTENCES, padding=True, return_tensors="pt")
    for i, ids in enumerate(enc["input_ids"].tolist()):
        print(f"ids[{i}] = {ids}")
    with torch.no_grad():
        out = model(**enc)
    hidden = out.last_hidden_state  # (B, T, H)
    cls = hidden[:, 0, :]
    emb = torch.nn.functional.normalize(cls, p=2, dim=-1)
    fmt = lambda v: ", ".join(f"{x:.6f}" for x in v.tolist())
    print(f"hidden[0,0,:8] = [{fmt(hidden[0, 0, :8])}]")
    for i in range(len(SENTENCES)):
        print(f"emb[{i}][:8] = [{fmt(emb[i, :8])}]")
    cos = emb @ emb.T
    print("cosine matrix:")
    for i in range(len(SENTENCES)):
        print("  " + fmt(cos[i]))
    print("pooler_output[0,:8] = [" + fmt(out.pooler_output[0, :8]) + "]")


if __name__ == "__main__":
    main()
