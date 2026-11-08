#!/usr/bin/env python3
"""DeepSeek-V2 reference logits (transformers, eager) for models/deepseek_v2_test.go.

    .venv/bin/python scripts/verify/ref_deepseek_v2.py <repo_or_dir> [bf16|f32] [--ids 1,2,3 | --chat] [--cpu]
    .venv/bin/python scripts/verify/ref_deepseek_v2.py --make-tiny {lite|qlora|group|v3|v3-rot} <dir>

--make-tiny writes a seeded tiny-random DeepseekV2ForCausalLM covering MLA
(compressed KV latent + RMSNorm, kv_b_proj expansion, one RoPE key shared by
all heads, value head dim != no-RoPE key dim), interleaved (complex-pair)
RoPE with YaRN and mscale_all_dim (the softmax scale gets mscale²), a dense
first layer, top-k-of-8 softmax routing without renormalisation, a
routed_scaling_factor != 1 and a shared expert. "lite" uses a plain q_proj
(DeepSeek-V2-Lite), "qlora" the q_a_proj → RMSNorm → q_b_proj chain
(DeepSeek-V2). "group" adds V2's group-limited routing (4 groups of 2
experts, 2 kept). "v3" / "v3-rot" are DeepseekV3ForCausalLM: sigmoid
router with a perturbed e_score_correction_bias, sum-of-top-2 group
selection (4 groups, 2 kept), renormalised top-3, routed scale 2.5, with
interleaved rope rows (rope_interleave true) or rotate-half ones (false).
Norm weights are perturbed away from 1.

--chat renders the validation-bar prompt (system + user) with the repo's
chat template and decodes 16 greedy tokens. Runs on CUDA when the weights
fit, else on the CPU (--cpu forces it).
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DeepseekV2Config, DeepseekV2ForCausalLM

if sys.argv[1] == "--make-tiny" and sys.argv[2].startswith("v3"):
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
    variant, out = sys.argv[2], sys.argv[3]
    cfg = DeepseekV3Config(
        vocab_size=512, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        q_lora_rank=48, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=12,
        n_routed_experts=8, num_experts_per_tok=3, n_shared_experts=1, first_k_dense_replace=1,
        n_group=4, topk_group=2, norm_topk_prob=True, routed_scaling_factor=2.5,
        rope_interleave=(variant == "v3"), max_position_embeddings=256, tie_word_embeddings=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0,
                         "original_max_position_embeddings": 32, "beta_fast": 32.0, "beta_slow": 1.0,
                         "mscale": 0.707, "mscale_all_dim": 0.707},
        bos_token_id=1, eos_token_id=2, pad_token_id=None)
    torch.manual_seed(0)
    m = DeepseekV3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name:
                p.add_(0.2 * torch.randn_like(p))
        for name, b in m.named_buffers():
            if "e_score_correction_bias" in name:
                b.copy_(0.3 * torch.randn_like(b))
    m.save_pretrained(out, safe_serialization=True)
    print("wrote", out)
    sys.exit(0)

if sys.argv[1] == "--make-tiny":
    variant, out = sys.argv[2], sys.argv[3]
    group = variant == "group"
    cfg = DeepseekV2Config(
        vocab_size=512, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        q_lora_rank=48 if variant == "qlora" else None, kv_lora_rank=32,
        qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=12,
        n_routed_experts=8, num_experts_per_tok=3, n_shared_experts=1, first_k_dense_replace=1,
        norm_topk_prob=False, routed_scaling_factor=1.5,
        topk_method="group_limited_greedy" if group else "greedy", n_group=4 if group else 1, topk_group=2 if group else 1,
        max_position_embeddings=256, tie_word_embeddings=False, rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0,
                         "original_max_position_embeddings": 32, "beta_fast": 32.0, "beta_slow": 1.0,
                         "mscale": 0.707, "mscale_all_dim": 0.707},
        bos_token_id=1, eos_token_id=2, pad_token_id=None)
    torch.manual_seed(0)
    m = DeepseekV2ForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name:
                p.add_(0.2 * torch.randn_like(p))
    m.save_pretrained(out, safe_serialization=True)
    print("wrote", out)
    sys.exit(0)

repo = sys.argv[1]
dt = torch.float32 if (len(sys.argv) > 2 and sys.argv[2] == "f32") else torch.bfloat16
chat = "--chat" in sys.argv
ids = None
if "--ids" in sys.argv:
    ids = [int(x) for x in sys.argv[sys.argv.index("--ids") + 1].split(",")]
tok = None if ids is not None else AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=dt, attn_implementation="eager").eval()
dev = "cpu"
if torch.cuda.is_available() and "--cpu" not in sys.argv:
    need = sum(p.numel() * p.element_size() for p in model.parameters())
    free, _ = torch.cuda.mem_get_info()
    if need < free - (2 << 30):
        dev = "cuda"
    else:
        print(f"weights {need / 2**30:.1f} GiB > free GPU {free / 2**30:.1f} GiB - 2: running on the CPU")
model = model.to(dev)
if chat:
    msgs = [{"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Explain in one sentence why the sky is blue."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
elif ids is None:
    ids = tok("The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is").input_ids
print("model class:", type(model).__name__)
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
