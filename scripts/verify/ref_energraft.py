#!/usr/bin/env python3
"""Tiny Energraft-EBT reference for models/energraft_test.go, run through the
release's own inference code (energraft/shared_energy.py, residual_head.py,
__init__.py from dimweak/Energraft-EBT-Qwen3-1.7B-Base).

    python -I scripts/verify/ref_energraft.py <energraft_code_dir> <out_dir> <out.json>

<energraft_code_dir> holds the release's energraft package files. A seeded
tiny Qwen3 (tied embeddings, GQA, QK-norm) is saved with save_pretrained,
a seeded residual energy head as energy_head.safetensors, and the release's
energy_config.json (Gaussian candidates, 2 updates, relaxation 0.05). The
reference records, for a prompt and then for each of a few generated
tokens, the Gaussian initial state it drew and the candidate state after
every update (EnergraftEBT.inference_logits), plus the greedy tokens.
"""
import json, os, sys, types

import torch

code, out_dir, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
pkg = types.ModuleType("energraft")
pkg.__path__ = [code]
sys.modules["energraft"] = pkg
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("energraft", os.path.join(code, "__init__.py"),
                                              submodule_search_locations=[code])
E = importlib.util.module_from_spec(spec)
sys.modules["energraft"] = E
spec.loader.exec_module(E)

from safetensors.torch import save_file  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

torch.manual_seed(0)
cfg = Qwen3Config(vocab_size=160, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                  rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, eos_token_id=1,
                  bos_token_id=0, torch_dtype="float32")
bb = Qwen3ForCausalLM(cfg).float().eval()
with torch.no_grad():
    for n, p in bb.named_parameters():
        if n.endswith("norm.weight"):
            p.copy_(1 + 0.2 * torch.randn_like(p))
        elif "embed_tokens" in n:
            p.copy_(torch.randn_like(p) * 0.5)
        else:
            p.copy_(torch.randn_like(p) * 0.15)
bb.requires_grad_(False)
os.makedirs(out_dir, exist_ok=True)
bb.save_pretrained(out_dir, safe_serialization=True)
D, W = cfg.hidden_size, 32
head = {"baseline": torch.randn(D) * 0.3, "center": torch.randn(D) * 0.1,
        "first.weight": torch.randn(W, D) * 0.2, "first.bias": torch.randn(W) * 0.1,
        "last.weight": torch.randn(1, W) * 0.3}
save_file(head, os.path.join(out_dir, "energy_head.safetensors"))
econf = {"format": "energraft-ebt-v1", "energy_steps": 2, "refinement_relaxation": 0.05,
         "candidate_initialization": "gaussian", "candidate_initial_std": 1.0,
         "candidate_coordinates": "probabilities", "candidate_step_size": 1.0, "dtype": "float32",
         "energy_head_width": W}
json.dump(econf, open(os.path.join(out_dir, "energy_config.json"), "w"), indent=2)

model = E.EnergraftEBT(bb, "gaussian", 1.0, 1.0, 2, 0.05)
model.head = E.NativeResidualHead(head["baseline"], head["center"], W)
model.head.load_state_dict(head)
model.eval().requires_grad_(False)

# Record the Gaussian draws and every update's state.
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
prompt = [0, 17, 99, 3, 42, 8, 120, 77]
res = {}
for steps in (1, 2):
    draws.clear(); states.clear()
    torch.manual_seed(1234)
    outs, _ = E.batched_generate(model, [prompt], 5, True, torch.device("cpu"), 2, inference_steps=steps)
    res[str(steps)] = {"tokens": outs[0], "draws": [[round(v, 7) for v in d] for d in draws],
                       "states": [[round(v, 6) for v in s] for s in states]}
    print("steps", steps, "tokens", outs[0])
json.dump({"prompt": prompt, "vocab": cfg.vocab_size, "runs": res}, open(out_json, "w"))
print("wrote", out_json)
