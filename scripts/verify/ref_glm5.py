#!/usr/bin/env python3
"""Tiny GLM-5.3 (glm5_next) text-model reference (transformers, CPU, F32)
for models/glm5_test.go.

    python scripts/verify/ref_glm5.py <out_dir> <out.json>

A seeded tiny Glm5NextTextModel with GLM-5.3-Flash's structure - Kimi
delta-attention layers and MLA layers without RoPE (mla_use_nope), dense
then sparse (sigmoid-routed, shared-expert) MLP layers, 4-stream
manifold-constrained hyper-connections, the clamped SwiGLU - plus an LM
head, saved in the checkpoint's naming (model.language_model.*, lm_head)
with its config.json. Records the logits of every position for a prompt
shorter than index_topk (so the DSA indexer selects every visible token
and the sparse attention is the dense causal one).
"""
import json, os, sys

import torch
from safetensors.torch import save_file
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

out_dir, out_json = sys.argv[1], sys.argv[2]
L = 5
layer_types = ["linear_attention", "linear_attention", "deepseek_sparse_attention", "linear_attention",
               "deepseek_sparse_attention"]
cfg = Glm5NextTextConfig(
    vocab_size=300, hidden_size=64, intermediate_size=96, moe_intermediate_size=32, num_hidden_layers=L,
    num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=3,
    routed_scaling_factor=2.5, kv_lora_rank=24, q_lora_rank=32, qk_rope_head_dim=0, v_head_dim=16,
    qk_nope_head_dim=16, n_group=1, topk_group=1, norm_topk_prob=True, hidden_act="silu", rms_norm_eps=1e-5,
    tie_word_embeddings=False, mlp_layer_types=["dense", "sparse", "sparse", "sparse", "sparse"],
    index_topk=64, index_head_dim=16, index_n_heads=2, layer_types=layer_types,
    indexer_types=["full"] * L, swiglu_limit=1.5, linear_head_dim=16, linear_num_heads=4,
    linear_conv_kernel_dim=4, linear_lower_bound=-5.0, hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=20,
    index_kpool=4, index_kpool_always_select_tail=True, max_position_embeddings=512, pad_token_id=0,
    bos_token_id=1, eos_token_id=2)
cfg._attn_implementation = "eager"
torch.manual_seed(0)
model = Glm5NextTextModel(cfg).eval()
lm_head = torch.nn.Linear(64, 300, bias=False)
with torch.no_grad():
    for n, p in list(model.named_parameters()) + [("lm_head.weight", lm_head.weight)]:
        if n.endswith("norm.weight") or "layernorm" in n:
            p.copy_(1 + 0.2 * torch.randn_like(p))
        elif "A_log" in n:
            p.copy_(torch.log(torch.rand_like(p) * 2 + 0.5))
        elif "dt_bias" in n:
            p.copy_(torch.randn_like(p) * 0.5)
        elif n.endswith("scale"):
            p.copy_(torch.rand_like(p) + 0.5)
        elif n.endswith(".base"):
            p.copy_(torch.randn_like(p) * 0.5)
        else:
            p.normal_(0, 0.1)
    for n, b in model.named_buffers():
        if "e_score_correction_bias" in n:
            b.copy_(torch.randn_like(b) * 0.05)

sd = {"model.language_model." + k: v.detach().contiguous() for k, v in model.state_dict().items()}
sd["lm_head.weight"] = lm_head.weight.detach().contiguous()
os.makedirs(out_dir, exist_ok=True)
save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
c = cfg.to_dict()
json.dump({"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next", "text_config": c},
          open(os.path.join(out_dir, "config.json"), "w"), indent=2)

ids = torch.tensor([[5, 77, 13, 201, 9, 150, 42, 3, 260, 18, 99, 7, 61]])
with torch.no_grad():
    h = model(input_ids=ids).last_hidden_state
    logits = lm_head(h)[0]
json.dump({"ids": ids[0].tolist(), "logits": [round(x, 6) for x in logits.flatten().tolist()], "vocab": 300},
          open(out_json, "w"))
print("logits", tuple(logits.shape), "keys", len(sd))

# FP8 variant (models/glm5_test.go's FP8 case): every projection stored as
# F8E4M3 with an 8 x 8 block weight_scale_inv (DeepSeek / GLM layout,
# W = fp8 * scale_inv, ragged edge blocks included), experts written per
# expert as the released checkpoints store them; the reference runs the
# dequantized weights.
if len(sys.argv) > 4:
    fp8_dir, fp8_json = sys.argv[3], sys.argv[4]
    BS = 8

    def quant(w):
        N, K = w.shape
        nb, kb = (N + BS - 1) // BS, (K + BS - 1) // BS
        sc = torch.zeros(nb, kb)
        q = torch.zeros(N, K, dtype=torch.float8_e4m3fn)
        deq = torch.zeros(N, K)
        for i in range(nb):
            for j in range(kb):
                blk = w[i * BS:(i + 1) * BS, j * BS:(j + 1) * BS]
                s = (blk.abs().max() / 448).clamp(min=1e-12)
                qb = (blk / s).to(torch.float8_e4m3fn)
                q[i * BS:(i + 1) * BS, j * BS:(j + 1) * BS] = qb
                deq[i * BS:(i + 1) * BS, j * BS:(j + 1) * BS] = qb.float() * s
                sc[i, j] = s
        return q, sc, deq

    fsd, deq = {}, {}
    for k, v in model.state_dict().items():
        full = "model.language_model." + k
        if k.endswith("mlp.experts.gate_up_proj") or k.endswith("mlp.experts.down_proj"):
            p, kind = k.rsplit(".", 1)
            for e in range(v.shape[0]):
                if kind == "gate_up_proj":
                    I = v.shape[1] // 2
                    parts = {"gate_proj": v[e, :I], "up_proj": v[e, I:]}
                else:
                    parts = {"down_proj": v[e]}
                for name, w in parts.items():
                    q, sc, d = quant(w)
                    base = f"model.language_model.{p}.{e}.{name}"
                    fsd[base + ".weight"], fsd[base + ".weight_scale_inv"] = q, sc
                    deq.setdefault(k, {})[(e, name)] = d
        elif k.endswith("_proj.weight") and v.dim() == 2 and "indexer" not in k:
            q, sc, d = quant(v)
            fsd[full], fsd[full[: -len("weight")] + "weight_scale_inv"] = q, sc
            deq[k] = d
        else:
            fsd[full] = v.contiguous()
    fsd["lm_head.weight"] = lm_head.weight.detach().contiguous()
    os.makedirs(fp8_dir, exist_ok=True)
    save_file(fsd, os.path.join(fp8_dir, "model.safetensors"), metadata={"format": "pt"})
    json.dump({"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next", "text_config": c,
               "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [BS, BS]}},
              open(os.path.join(fp8_dir, "config.json"), "w"), indent=2)
    with torch.no_grad():
        sd = model.state_dict()
        for k, d in deq.items():
            if isinstance(d, dict):
                t = sd[k]
                for (e, name), w in d.items():
                    if name == "gate_proj":
                        t[e, : w.shape[0]] = w
                    elif name == "up_proj":
                        t[e, t.shape[1] // 2:] = w
                    else:
                        t[e] = w
            else:
                sd[k].copy_(d)
        logits = lm_head(model(input_ids=ids).last_hidden_state)[0]
    json.dump({"ids": ids[0].tolist(), "logits": [round(x, 6) for x in logits.flatten().tolist()], "vocab": 300},
              open(fp8_json, "w"))
    print("fp8 variant", tuple(logits.shape))
