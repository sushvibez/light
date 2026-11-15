#!/usr/bin/env python3
"""Tiny 4-bit Gemma 4 (31B layout) for models/gemma4_int4_test.go.

    python scripts/verify/ref_gemma4_int4.py <out_dir> <out.json>

A seeded tiny Gemma4ForCausalLM with gemma-4-31B's features (sliding and
full layers, K = V on full layers, a wider global head with partial
proportional RoPE, logit softcap), its projections from layer 1 on packed
as GPTQ 4-bit (layer 0 left in BF16, as QuantTrio's AWQ build leaves
model.layers.0). The reference logits come from the same model with the
4-bit weights dequantized (W = s·(q − z)), in F32.
"""
import json, os, sys

import torch
from safetensors.torch import save_file
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

out_dir, out_json = sys.argv[1], sys.argv[2]
G = 32
cfg = Gemma4TextConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                       num_key_value_heads=2, num_global_key_value_heads=1, head_dim=16, global_head_dim=32,
                       layer_types=["sliding_attention", "full_attention", "sliding_attention"], sliding_window=8,
                       attention_k_eq_v=True, hidden_size_per_layer_input=0, final_logit_softcapping=30.0,
                       rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
                                        "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}},
                       tie_word_embeddings=True, max_position_embeddings=256)
torch.manual_seed(0)
m = Gemma4ForCausalLM(cfg).eval()
with torch.no_grad():
    for n, p in m.named_parameters():
        if "norm" in n:
            p.copy_(0.2 * torch.randn_like(p))
        elif "layer_scalar" in n:
            p.fill_(0.9)
        else:
            p.normal_(0, 0.08)


def pack_gptq(w):  # (N, K) -> qweight, qzeros, scales, g_idx, dequantized
    N, K = w.shape
    wg = w.float().reshape(N, K // G, G)
    scale = (wg.abs().amax(dim=2) / 7).clamp(min=1e-8).half().float()
    q = (torch.round(wg / scale[:, :, None]) + 8).clamp(0, 15)
    deq = (scale[:, :, None] * (q - 8)).reshape(N, K)
    q = q.to(torch.int64).reshape(N, K).T.contiguous()
    qweight = torch.zeros(K // 8, N, dtype=torch.int64)
    for j in range(8):
        qweight |= q[j::8] << (4 * j)
    qzeros = torch.zeros(K // G, N // 8, dtype=torch.int64)
    for j in range(8):
        qzeros |= torch.full((K // G, N // 8), 7, dtype=torch.int64) << (4 * j)
    i32 = lambda x: torch.where(x >= 2 ** 31, x - 2 ** 32, x).to(torch.int32)
    return i32(qweight), i32(qzeros), scale.T.contiguous().half(), (torch.arange(K) // G).to(torch.int32), deq


sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
outsd, deq = {}, {}
for k, v in sd.items():
    layer = int(k.split(".")[2]) if k.startswith("model.layers.") else -1
    if layer >= 1 and k.endswith("_proj.weight") and (".self_attn." in k or ".mlp." in k):
        p = k[: -len("weight")]
        qw, qz, sc, gi, d = pack_gptq(v)
        outsd.update({p + "qweight": qw, p + "qzeros": qz, p + "scales": sc, p + "g_idx": gi})
        deq[k] = d
    else:
        outsd[k] = v.to(torch.bfloat16) if v.is_floating_point() and layer == 0 else v.contiguous()
os.makedirs(out_dir, exist_ok=True)
save_file(outsd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
c = cfg.to_dict()
c["architectures"] = ["Gemma4ForCausalLM"]
# Released configs (gemma-4-31B) name the full-attention head as below;
# newer transformers writes it as per_layer_config instead.
c["global_head_dim"], c["num_global_key_value_heads"] = 32, 1
c["quantization_config"] = {"quant_method": "gptq", "bits": 4, "group_size": G, "desc_act": False, "sym": True,
                            "checkpoint_format": "gptq"}
json.dump(c, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
# Reference: dequantized projections, layer 0 rounded through BF16 as stored.
with torch.no_grad():
    for k, v in m.state_dict().items():
        if k in deq:
            v.copy_(deq[k])
        elif k.startswith("model.layers.0.") and v.is_floating_point():
            v.copy_(v.to(torch.bfloat16).float())
    ids = torch.tensor([[2, 17, 99, 5, 200, 31, 64, 7, 150, 12, 3]])
    logits = m(input_ids=ids).logits[0]
json.dump({"ids": ids[0].tolist(), "logits": [round(x, 6) for x in logits.flatten().tolist()],
           "vocab": cfg.vocab_size}, open(out_json, "w"))
print("4-bit modules", sum(1 for k in outsd if k.endswith(".qweight")), "logits", tuple(logits.shape))
