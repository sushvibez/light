#!/usr/bin/env python3
"""Mellum 2.1 reference logits (transformers, eager) for models/mellum_test.go.

    .venv/bin/python scripts/verify/ref_mellum.py <repo_or_dir> [bf16|f32] [--ids 1,2,3 | --chat]
    .venv/bin/python scripts/verify/ref_mellum.py --make-tiny <dir>

--chat renders the validation-bar prompt (system + user, the same messages
as ref_phi4.py) with the repo's chat template and decodes 16 greedy tokens:
    python scripts/verify/ref_mellum.py JetBrains/Mellum2.1-12B-A2.5B-Thinking bf16 --chat

--make-tiny writes a seeded tiny-random MellumForCausalLM that exercises
everything Mellum adds on top of Qwen3-MoE: sliding / full layers with a
window shorter than the test prompt, YaRN on the full layers (attention
factor derived from the factor) with a different RoPE base on the sliding
layers, and a dense MLP layer among the sparse ones.
Without --ids the prompt is tokenised with the repo tokenizer. Runs on CUDA
when available, else on the CPU. Prints the ids, the top-5 last-position
logits and the greedy continuation.
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MellumConfig, MellumForCausalLM

if sys.argv[1] == "--make-tiny":
    out = sys.argv[2]
    cfg = MellumConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_experts=8, num_experts_per_tok=2, norm_topk_prob=True,
        max_position_embeddings=256, sliding_window=6, tie_word_embeddings=False,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        mlp_layer_types=["sparse", "dense", "sparse", "sparse"],
        rope_parameters={
            "full_attention": {"rope_type": "yarn", "rope_theta": 500000.0, "factor": 4.0,
                               "original_max_position_embeddings": 64, "beta_fast": 32.0, "beta_slow": 1.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    )
    torch.manual_seed(0)
    m = MellumForCausalLM(cfg).eval()
    m.save_pretrained(out, safe_serialization=True)
    print("wrote", out)
    sys.exit(0)

repo = sys.argv[1]
dt = torch.float32 if (len(sys.argv) > 2 and sys.argv[2] == "f32") else torch.bfloat16
dev = "cuda" if torch.cuda.is_available() else "cpu"
ids = None
chat = "--chat" in sys.argv
if "--ids" in sys.argv:
    ids = [int(x) for x in sys.argv[sys.argv.index("--ids") + 1].split(",")]
tok = None if ids is not None else AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=dt, attn_implementation="eager").to(dev).eval()
if chat:
    msgs = [{"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Explain in one sentence why the sky is blue."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
elif ids is None:
    prompt = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
    ids = tok(prompt).input_ids
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
