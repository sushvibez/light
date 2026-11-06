#!/usr/bin/env python3
"""Falcon reference logits (transformers, eager) for models/falcon_test.go.

    .venv/bin/python scripts/verify/ref_falcon.py <repo_or_dir> [bf16|f32] [--ids 1,2,3]
    .venv/bin/python scripts/verify/ref_falcon.py --make-tiny <dir> <variant>

--make-tiny writes a seeded tiny-random FalconForCausalLM so each decoder
variant can be checked without 7B+ weights:
  mq7b   falcon-7b layout: rotary, multi_query, parallel_attn, no bias
  alibi  falcon-rw-1b layout: alibi, multi-head, sequential, bias
Without --ids the prompt is tokenised with the repo tokenizer. Prints the
ids, the top-5 last-position logits and the greedy continuation.
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, FalconConfig, FalconForCausalLM

if sys.argv[1] == "--make-tiny":
    out, variant = sys.argv[2], sys.argv[3]
    kw = dict(vocab_size=1024, hidden_size=64, num_hidden_layers=3, num_attention_heads=4, max_position_embeddings=512)
    if variant == "mq7b":
        cfg = FalconConfig(alibi=False, multi_query=True, new_decoder_architecture=False, parallel_attn=True, bias=False, **kw)
    elif variant == "alibi":
        cfg = FalconConfig(alibi=True, multi_query=False, new_decoder_architecture=False, parallel_attn=False, bias=True, **kw)
    else:
        sys.exit("variant: mq7b | alibi")
    torch.manual_seed(0)
    m = FalconForCausalLM(cfg).eval()
    m.save_pretrained(out, safe_serialization=True)
    print("wrote", out)
    sys.exit(0)

repo = sys.argv[1]
ATTN = "sdpa" if "--sdpa" in sys.argv else "eager"
dt = torch.float32 if (len(sys.argv) > 2 and sys.argv[2] == "f32") else torch.bfloat16
ids = None
if "--ids" in sys.argv:
    ids = [int(x) for x in sys.argv[sys.argv.index("--ids") + 1].split(",")]
tok = None if ids is not None else AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=dt, attn_implementation=ATTN).to("cuda").eval()
if ids is None:
    prompt = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
    ids = tok(prompt).input_ids
print("prompt ids:", ids)
x = torch.tensor([ids], device="cuda")
with torch.no_grad():
    logits = model(input_ids=x).logits[0, -1].float()
    topv, topi = logits.topk(5)
    print(f"dtype={dt} top5:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  id={i:<7} {(tok.decode([i]) if tok else '')!r:<14} logit={v:.4f}")
    out = model.generate(x, max_new_tokens=8, do_sample=False)
    if tok:
        print("greedy:", repr(tok.decode(out[0, len(ids):], skip_special_tokens=False)))
    print("greedy ids:", out[0, len(ids):].tolist())
