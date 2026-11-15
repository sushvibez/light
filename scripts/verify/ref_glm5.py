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
