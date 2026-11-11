#!/usr/bin/env python3
"""T5 encoder reference (transformers, CPU, F32) for models/t5_test.go.

    .venv/bin/python scripts/verify/ref_t5.py <repo> <out.json>

Encodes a fixed sentence with the repo's tokenizer and T5EncoderModel and
writes {repo, ids, d_model, hidden} (hidden = last_hidden_state[0], row-major
(T, d_model)) to out.json.
"""
import json, sys
import torch
from transformers import AutoTokenizer, T5EncoderModel

repo, out = sys.argv[1], sys.argv[2]
text = "A photograph of an astronaut riding a horse on the moon, highly detailed."
# All T5 checkpoints share one SentencePiece vocabulary; t5-small ships it as
# tokenizer.json (v1.1 repos only have spiece.model).
tok = AutoTokenizer.from_pretrained("google-t5/t5-small")
ids = tok(text).input_ids
model = T5EncoderModel.from_pretrained(repo, torch_dtype=torch.float32).eval()
with torch.no_grad():
    h = model(input_ids=torch.tensor([ids])).last_hidden_state[0]
print("ids:", ids)
print("hidden[0,:4]:", h[0, :4].tolist(), "mean abs:", h.abs().mean().item())
json.dump({"repo": repo, "text": text, "ids": ids, "d_model": h.shape[1],
           "hidden": [round(v, 6) for v in h.flatten().tolist()]}, open(out, "w"))
