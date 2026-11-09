#!/usr/bin/env python3
"""Granite reference logits (transformers, eager) for models/granite_test.go.

    .venv/bin/python scripts/verify/ref_granite.py --make-tiny <dir>
    .venv/bin/python scripts/verify/ref_granite.py <repo_or_dir> [bf16|f32] [--ids 1,2,3 | --chat] [--cpu]

--make-tiny writes a seeded tiny-random GraniteMoeHybridForCausalLM with
only attention layers and no routed experts (the granite-4.0-micro shape),
non-trivial embedding / attention / residual multipliers and logits
scaling, and perturbed norms. --chat renders the validation-bar prompt
(system + user) with the repo's chat template and decodes 16 greedy tokens.
Prints the ids, the top-5 last-position logits and the greedy continuation.
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM

if sys.argv[1] == "--make-tiny":
    out = sys.argv[2]
    cfg = GraniteMoeHybridConfig(
        vocab_size=512, hidden_size=64, intermediate_size=96, shared_intermediate_size=96,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        layer_types=["attention"] * 3, num_local_experts=0, num_experts_per_tok=0,
        max_position_embeddings=256, rope_theta=10000000.0, rms_norm_eps=1e-5,
        embedding_multiplier=12.0, attention_multiplier=0.0625, residual_multiplier=0.22, logits_scaling=10.0,
        tie_word_embeddings=True, position_embedding_type="rope", bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    torch.manual_seed(0)
    m = GraniteMoeHybridForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name:
                p.add_(0.2 * torch.randn_like(p))
    m.save_pretrained(out, safe_serialization=True)
    print("wrote", out)
    sys.exit(0)

repo = sys.argv[1]
dt = torch.float32 if (len(sys.argv) > 2 and sys.argv[2] == "f32") else torch.bfloat16
ids = None
chat = "--chat" in sys.argv
if "--ids" in sys.argv:
    ids = [int(x) for x in sys.argv[sys.argv.index("--ids") + 1].split(",")]
tok = None if ids is not None else AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=dt, attn_implementation="eager").eval()
dev = "cuda" if torch.cuda.is_available() and "--cpu" not in sys.argv else "cpu"
model = model.to(dev)
if chat:
    msgs = [{"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Explain in one sentence why the sky is blue."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
elif ids is None:
    ids = tok("The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is").input_ids
print("prompt ids:", ids)
x = torch.tensor([ids], device=dev)
with torch.no_grad():
    logits = model(input_ids=x).logits[0, -1].float()
    topv, topi = logits.topk(5)
    print(f"dtype={dt} device={dev} top5:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  id={i:<7} {(tok.decode([i]) if tok else '')!r:<14} logit={v:.4f}")
    out = model.generate(x, max_new_tokens=16 if chat else 8, do_sample=False)
    if tok:
        print("greedy:", repr(tok.decode(out[0, len(ids):], skip_special_tokens=False)))
    print("greedy ids:", out[0, len(ids):].tolist())
