#!/usr/bin/env python3
"""Tiny GPTQ-format 4-bit checkpoints for models/int4_test.go.

    .venv/bin/python scripts/verify/make_int4_tiny.py qwen3moe <dir>

Builds a seeded tiny-random transformers model and writes it the way
AutoGPTQ / GPTQModel do ("gptq" checkpoint format, 4 bits, symmetric,
ordered groups): every attention projection and every expert's
gate / up / down projection becomes qweight I32 (K/8, N) packed along K,
qzeros I32 (K/G, N/8) packed along N holding zero − 1 (= 7 for symmetric),
scales F16 (K/G, N) and g_idx (K,); experts are written per expert
(mlp.experts.{e}.{gate,up,down}_proj, as real GPTQ MoE checkpoints).
Weights are quantized by plain round-to-nearest (GPTQ's error correction
changes the values, not the layout). The router, norms, embedding and LM
head stay float. The reference is scripts/verify/ref_int4.py on the result.
"""
import json, os, sys
import torch
from safetensors.torch import save_file

kind, out = sys.argv[1], sys.argv[2]
G = 32


def pack_gptq(w):  # w (N, K) float -> qweight, qzeros, scales, g_idx
    N, K = w.shape
    wg = w.float().reshape(N, K // G, G)
    scale = (wg.abs().amax(dim=2) / 7).clamp(min=1e-8)                 # (N, K/G)
    scale = scale.half().float()                                        # what the checkpoint stores
    q = (torch.round(wg / scale[:, :, None]) + 8).clamp(0, 15).to(torch.int64).reshape(N, K)
    qt = q.T.contiguous()                                               # (K, N)
    qweight = torch.zeros(K // 8, N, dtype=torch.int64)
    for j in range(8):
        qweight |= qt[j::8] << (4 * j)
    zeros = torch.full((K // G, N), 7, dtype=torch.int64)               # zero 8, stored as zero − 1
    qzeros = torch.zeros(K // G, N // 8, dtype=torch.int64)
    for j in range(8):
        qzeros |= zeros[:, j::8] << (4 * j)
    to_i32 = lambda x: x.to(torch.int64).apply_(lambda v: v - (1 << 32) if v >= (1 << 31) else v).to(torch.int32)
    return to_i32(qweight), to_i32(qzeros), scale.T.contiguous().half(), (torch.arange(K) // G).to(torch.int32)


if kind != "qwen3moe":
    sys.exit("kinds: qwen3moe")
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
cfg = Qwen3MoeConfig(vocab_size=512, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                     num_experts=4, num_experts_per_tok=2, norm_topk_prob=True, max_position_embeddings=256,
                     tie_word_embeddings=False, rms_norm_eps=1e-6, bos_token_id=1, eos_token_id=2)
torch.manual_seed(0)
m = Qwen3MoeForCausalLM(cfg).eval()
with torch.no_grad():
    for name, p in m.named_parameters():
        if "norm" in name:
            p.add_(0.2 * torch.randn_like(p))
sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
outsd = {}
for k, v in sd.items():
    if ".mlp.experts.gate_up_proj" in k:   # (E, 2I, H) fused in transformers 5: split per expert
        p = k[: -len("gate_up_proj")]
        I = v.shape[1] // 2
        for e in range(v.shape[0]):
            for name, w in (("gate_proj", v[e, :I]), ("up_proj", v[e, I:])):
                qw, qz, sc, gi = pack_gptq(w)
                outsd.update({f"{p}{e}.{name}.qweight": qw, f"{p}{e}.{name}.qzeros": qz,
                              f"{p}{e}.{name}.scales": sc, f"{p}{e}.{name}.g_idx": gi})
    elif ".mlp.experts.down_proj" in k:    # (E, H, I)
        p = k[: -len("down_proj")]
        for e in range(v.shape[0]):
            qw, qz, sc, gi = pack_gptq(v[e])
            outsd.update({f"{p}{e}.down_proj.qweight": qw, f"{p}{e}.down_proj.qzeros": qz,
                          f"{p}{e}.down_proj.scales": sc, f"{p}{e}.down_proj.g_idx": gi})
    elif ".self_attn." in k and k.endswith("_proj.weight"):
        p = k[: -len("weight")]
        qw, qz, sc, gi = pack_gptq(v)
        outsd.update({p + "qweight": qw, p + "qzeros": qz, p + "scales": sc, p + "g_idx": gi})
    else:
        outsd[k] = v.contiguous()
os.makedirs(out, exist_ok=True)
save_file(outsd, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
c = cfg.to_dict()
c["architectures"] = ["Qwen3MoeForCausalLM"]
c["quantization_config"] = {"quant_method": "gptq", "bits": 4, "group_size": G, "desc_act": False, "sym": True,
                            "checkpoint_format": "gptq"}
json.dump(c, open(os.path.join(out, "config.json"), "w"), indent=2)
print("wrote", out, sum(1 for k in outsd if k.endswith(".qweight")), "4-bit modules")
