#!/usr/bin/env python3
"""Energraft-EBT-Qwen3-1.7B-Base real-weight check (Colab, CPU; ~7 GB).

    python scripts/colab/energraft_check.py <work_dir> [max_new_tokens]

Downloads the v0.3 release into <work_dir>, runs its own energraft code
(EnergraftEBT.generate, seed 1234, the release's 2 energy updates) on a
short prompt, and records the Gaussian draws, every token's refined
candidate state and the greedy tokens in <work_dir>/ref.json.
models/energraft_real_test.go (LIGHT_ENERGRAFT_DIR=<work_dir>) replays the
draws with Light and compares.
"""
import json, os, sys

import torch
from huggingface_hub import snapshot_download

work = sys.argv[1]
n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 12
snapshot_download("dimweak/Energraft-EBT-Qwen3-1.7B-Base", revision="v0.3", local_dir=work)
sys.path.insert(0, work)
import energraft as E  # noqa: E402

model = E.EnergraftEBT.from_pretrained(work, device="cpu")
draws, states = [], []
_randn = torch.randn


def randn_rec(*a, **k):
    x = _randn(*a, **k)
    draws.append(x.flatten().tolist())
    return x


torch.randn = randn_rec
_infer = E.EnergraftEBT.inference_logits


def infer_rec(self, context, steps=None):
    out = _infer(self, context, steps)
    states.append(out.flatten().tolist())
    return out


E.EnergraftEBT.inference_logits = infer_rec
prompt = "The capital of France is"
ids = model.tokenizer.encode(prompt, add_special_tokens=False)
out = model.generate_ids([ids], max_new_tokens=n_new, seed=1234)[0]
text = model.tokenizer.decode(out, skip_special_tokens=True)
print("prompt ids", ids, "tokens", out, "text", repr(text), flush=True)
eos = model.backbone.config.eos_token_id
json.dump({"prompt": ids, "tokens": out, "text": text, "eos": eos if isinstance(eos, list) else [eos],
           "draws": draws, "states": states}, open(os.path.join(work, "ref.json"), "w"))
print("wrote", os.path.join(work, "ref.json"))
