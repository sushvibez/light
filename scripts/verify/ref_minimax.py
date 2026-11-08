#!/usr/bin/env python3
"""MiniMax-Text-01 reference logits (transformers, eager) for models/minimax_test.go.

    .venv/bin/python scripts/verify/ref_minimax.py <dir> f32 --ids 1,2,3
    .venv/bin/python scripts/verify/ref_minimax.py --make-tiny <dir>

The real model (MiniMaxAI/MiniMax-Text-01-hf, 456 B params) is far beyond
this setup, so Light's implementation is pinned to tiny-random checkpoints.
--make-tiny writes a seeded MiniMaxForCausalLM with lightning-attention
(linear) and softmax-attention layers alternating, a lightning block_size of
4 (so an 18-token prefill crosses several blocks of transformers' block
algorithm), the DeepNorm-style alpha / beta residual factors all != 1, GQA
4 / 2 heads of 16 with RoPE on the softmax layers, and 4 experts top-2 in
every layer. Norm weights are perturbed.
"""
import sys, torch
from transformers import MiniMaxConfig, MiniMaxForCausalLM

if sys.argv[1] == "--make-tiny":
    out = sys.argv[2]
    cfg = MiniMaxConfig(
        vocab_size=512, hidden_size=64, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        block_size=4, num_local_experts=4, num_experts_per_tok=2, rms_norm_eps=1e-5,
        full_attn_alpha_factor=1.3, full_attn_beta_factor=0.8, linear_attn_alpha_factor=1.2,
        linear_attn_beta_factor=0.9, mlp_alpha_factor=1.1, mlp_beta_factor=0.7,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=False, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    torch.manual_seed(0)
    m = MiniMaxForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name:
                p.add_(0.2 * torch.randn_like(p))
    m.save_pretrained(out, safe_serialization=True)
    print("wrote", out)
    sys.exit(0)

repo = sys.argv[1]
dt = torch.float32 if (len(sys.argv) > 2 and sys.argv[2] == "f32") else torch.bfloat16
ids = [int(x) for x in sys.argv[sys.argv.index("--ids") + 1].split(",")]
model = MiniMaxForCausalLM.from_pretrained(repo, dtype=dt, attn_implementation="eager").eval()
x = torch.tensor([ids])
with torch.no_grad():
    logits = model(input_ids=x).logits[0, -1].float()
    topv, topi = logits.topk(5)
    print(f"dtype={dt} top5:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  id={i:<7} logit={v:.4f}")
    out = model.generate(x, max_new_tokens=8, do_sample=False)
    print("greedy ids:", out[0, len(ids):].tolist())
