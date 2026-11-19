#!/usr/bin/env python3
"""Tiny Edge0-35B-A3B-shaped checkpoint (MLX 4-bit + edge0 LoRA) and its
reference logits, for models/qwen3_5_mlx_test.go.

    .venv/bin/python scripts/verify/make_edge0_tiny.py models/testdata/edge0-tiny models/testdata/edge0-tiny.json

Builds a seeded tiny-random Qwen3_5MoeForCausalLM (two Gated DeltaNet and
two gated full-attention layers, 8 routed experts top-2 plus the gated
shared expert) and writes it the way mlx-vlm converts
Edge0/Edge0-35B-A3B-preview: `language_model.`-prefixed names, the experts
stacked as `switch_mlp.{gate,up,down}_proj` (E, N, K), conv1d as (C, K, 1),
the zero-centred norms saved with the 1 added, metadata {"format": "mlx"},
and every Linear and the embedding affine-quantized (mlx.core.quantize,
reimplemented below from mlx/ops.cpp: group 64, 4 bits, BF16 scales /
biases) except the router and shared-expert gate, at 8 bits as config.json's
per-module overrides say. Next to it, lora_tiny.safetensors in edge0's
layout: `<module>.lora_A` (r, in) / `.lora_B` (out, r) F16 on the
attention, DeltaNet and shared-expert projections, with r / alpha / K
metadata (K: the routed width it runs at, 3 here against the config's 2).

The reference is transformers (eager, F32, CPU) on the dequantized
weights: once as is at top-2, once with the adapter merged
(W + alpha / r · B A) at top-3. The JSON holds the token ids and both
last-position logit vectors.
"""
import json, os, sys
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

out, ref_path = sys.argv[1], sys.argv[2]
G, R, ALPHA, K_LORA = 64, 4, 8.0, 3

cfg = Qwen3_5MoeTextConfig(
    vocab_size=512, hidden_size=128, num_hidden_layers=4,
    layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
    num_attention_heads=4, num_key_value_heads=2, head_dim=64,
    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
    linear_conv_kernel_dim=4, moe_intermediate_size=64, shared_expert_intermediate_size=64,
    num_experts=8, num_experts_per_tok=2, max_position_embeddings=512, tie_word_embeddings=False,
    initializer_range=0.08,
    rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                     "mrope_section": [3, 3, 2], "mrope_interleaved": True},
)
torch.manual_seed(0)
m = Qwen3_5MoeForCausalLM(cfg).eval()
with torch.no_grad():
    for name, p in m.named_parameters():
        if "experts" in name:
            p.normal_(0, 0.08)
        if "norm" in name or "A_log" in name or "dt_bias" in name:
            p.add_(0.2 * torch.randn_like(p))
sd = {k: v.detach().float().clone() for k, v in m.state_dict().items()}


def bf16(x):
    return torch.as_tensor(np.ascontiguousarray(x)).to(torch.bfloat16)


def mlx_quantize(w, bits):
    """mlx.core.quantize(w, group_size=G, bits) (affine): codes U32 packed
    low bits first, BF16 scales / biases, and the dequantized F32 weight."""
    w = w.numpy().astype(np.float32)
    N, K = w.shape
    wg = w.reshape(N, K // G, G)
    wmax, wmin = wg.max(-1, keepdims=True), wg.min(-1, keepdims=True)
    nb = np.float32(2 ** bits - 1)
    mask = np.abs(wmin) > np.abs(wmax)
    s = np.maximum((wmax - wmin) / nb, np.float32(1e-7))
    s = np.where(mask, s, -s).astype(np.float32)
    edge = np.where(mask, wmin, wmax)
    q0 = np.rint(edge / s)
    s = np.where(q0 != 0, edge / np.where(q0 != 0, q0, 1), s).astype(np.float32)
    b = np.where(q0 == 0, np.float32(0), edge).astype(np.float32)
    q = np.clip(np.rint((wg - b) / s), 0, nb).astype(np.uint32)
    per = 32 // bits
    qp = q.reshape(N, K // per, per)
    packed = np.zeros((N, K // per), np.uint32)
    for j in range(per):
        packed |= qp[:, :, j] << np.uint32(bits * j)
    sb, bb = bf16(s[..., 0]), bf16(b[..., 0])
    deq = q.astype(np.float32) * sb.float().numpy()[..., None] + bb.float().numpy()[..., None]
    return torch.from_numpy(packed), sb, bb, torch.from_numpy(deq.reshape(N, K))


mlx, quant_over, deq = {}, {}, {}


def put_quant(hf_name, w, bits=4):
    """Quantize the (N, K) HF weight hf_name ("….weight") into mlx under its MLX module name."""
    mod = "language_model." + hf_name[: -len(".weight")]
    if hf_name == "lm_head.weight":
        mod = "language_model.lm_head"
    pk, sc, bi, dq = mlx_quantize(w, bits)
    mlx[mod + ".weight"], mlx[mod + ".scales"], mlx[mod + ".biases"] = pk, sc, bi
    if bits != 4:
        quant_over[mod] = {"group_size": G, "bits": bits}
    deq[hf_name] = dq


norm_sfx = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight")
for k, v in sd.items():
    if k.endswith(".mlp.experts.gate_up_proj"):  # (E, 2I, H)
        p = "language_model." + k[: -len("experts.gate_up_proj")] + "switch_mlp."
        I = v.shape[1] // 2
        dqs = []
        for i, part in enumerate(("gate_proj", "up_proj")):
            outs = [mlx_quantize(v[e, i * I:(i + 1) * I], 4) for e in range(v.shape[0])]
            for j, sfx in enumerate(("weight", "scales", "biases")):
                mlx[p + part + "." + sfx] = torch.stack([o[j] for o in outs])
            dqs.append(torch.stack([o[3] for o in outs]))
        deq[k] = torch.cat(dqs, 1)
    elif k.endswith(".mlp.experts.down_proj"):  # (E, H, I)
        p = "language_model." + k[: -len("experts.down_proj")] + "switch_mlp.down_proj."
        outs = [mlx_quantize(v[e], 4) for e in range(v.shape[0])]
        for j, sfx in enumerate(("weight", "scales", "biases")):
            mlx[p + sfx] = torch.stack([o[j] for o in outs])
        deq[k] = torch.stack([o[3] for o in outs])
    elif k.endswith(".mlp.gate.weight") or k.endswith(".mlp.shared_expert_gate.weight"):
        put_quant(k, v, bits=8)
    elif v.dim() == 2 and k.endswith(".weight"):  # every other Linear and the embedding
        put_quant(k, v)
    elif k.endswith("conv1d.weight"):  # (C, 1, K) -> MLX (C, K, 1)
        mlx["language_model." + k] = bf16(v.transpose(1, 2).contiguous())
        deq[k] = mlx["language_model." + k].float().transpose(1, 2).contiguous()
    elif k.endswith(norm_sfx) or k == "model.norm.weight":  # saved as 1 + w
        mlx["language_model." + k] = bf16(1 + v)
        deq[k] = mlx["language_model." + k].float() - 1
    else:  # A_log, dt_bias, the gated norm
        mlx["language_model." + k] = bf16(v)
        deq[k] = mlx["language_model." + k].float()

os.makedirs(out, exist_ok=True)
save_file(mlx, os.path.join(out, "model.safetensors"), metadata={"format": "mlx"})
text = cfg.to_dict()
text.pop("transformers_version", None)
q = {"group_size": G, "bits": 4, "mode": "affine", **quant_over}
top = {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
       "text_config": text, "tie_word_embeddings": False, "quantization": q, "quantization_config": q}
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(top, f, indent=2)

# The adapter, on the same modules as the real one.
g = torch.Generator().manual_seed(1)
lora, merged = {}, dict(deq)
for k in sd:
    if not k.endswith(".weight") or sd[k].dim() != 2:
        continue
    if not any(s in k for s in (".linear_attn.in_proj", ".linear_attn.out_proj", ".self_attn.", ".mlp.shared_expert.")):
        continue
    N, Kin = sd[k].shape
    A = (0.05 * torch.randn(R, Kin, generator=g)).half()
    B = (0.05 * torch.randn(N, R, generator=g)).half()
    mod = "language_model." + k[: -len(".weight")]
    lora[mod + ".lora_A"], lora[mod + ".lora_B"] = A, B
    merged[k] = deq[k] + (ALPHA / R) * (B.float() @ A.float())
meta = {"model": "edge0-tiny", "kind": "lora", "K": str(K_LORA), "r": str(R), "alpha": str(int(ALPHA)), "format_version": "1"}
save_file(lora, os.path.join(out, "lora_tiny.safetensors"), metadata={"__metadata__": json.dumps(meta)})

ids = [(37 * i + 11) % 512 for i in range(20)]


def last_logits(state, topk):
    m.load_state_dict(state)
    for layer in m.model.layers:
        layer.mlp.gate.top_k = topk
    with torch.no_grad():
        return m(torch.tensor([ids])).logits[0, -1].tolist()


base = last_logits(deq, 2)
with_lora = last_logits(merged, K_LORA)
with open(ref_path, "w") as f:
    json.dump({"ids": ids, "base": [round(x, 6) for x in base], "lora": [round(x, 6) for x in with_lora]}, f)
for name, lg in (("base", base), ("lora", with_lora)):
    t = torch.tensor(lg)
    v, i = t.topk(5)
    print(name, list(zip(i.tolist(), [round(x, 4) for x in v.tolist()])))
