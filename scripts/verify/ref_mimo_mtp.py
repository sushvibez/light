#!/usr/bin/env python3
"""MiMo MTP draft-head reference for models/mimo_test.go (no remote code).

    .venv/bin/python scripts/verify/ref_mimo_mtp.py <dir> --ids 1,2,3,...

MiMo-7B's MiMoMTPLayers (the repo's modeling_mimo.py, read, not executed):

    e = token_layernorm(embed(t+1));  h = hidden_layernorm(h_t)
    x = input_proj(cat([h, e]))
    x = x + self_attn(input_layernorm(x));  x = x + mlp(post_attention_layernorm(x))
    logits = lm_head(final_layernorm(x))

re-expressed here in plain PyTorch (Qwen2 attention: GQA, q/k/v bias,
rotate-half RoPE, causal; SwiGLU MLP) on the main model's final normed hidden
states from transformers' Qwen2ForCausalLM (h_t = hidden_states[-1], the
LM head's input, as vLLM feeds MiMo's MTP). The pairs (h_i, ids[i+1]) for
i = 0 .. T-2 run as one causal sequence; the script prints the top-5 of the
last row (the draft for the token after ids[T-1]).
"""
import json, os, sys
import torch
from safetensors import safe_open
from transformers import Qwen2Config, Qwen2ForCausalLM

d = sys.argv[1]
ids = [int(x) for x in sys.argv[sys.argv.index("--ids") + 1].split(",")]
raw = json.load(open(os.path.join(d, "config.json")))
for k in ("model_type", "architectures", "auto_map", "num_nextn_predict_layers"):
    raw.pop(k, None)
cfg = Qwen2Config(**raw)
main = Qwen2ForCausalLM.from_pretrained(d, config=cfg, dtype=torch.float32, attn_implementation="eager").eval()
f = safe_open(os.path.join(d, "model.safetensors"), "pt")
W = lambda n: f.get_tensor("model.mtp_layers.0." + n).float()

H, nh, kvh = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
hd = getattr(cfg, "head_dim", None) or H // nh
eps = cfg.rms_norm_eps
theta = cfg.rope_parameters["rope_theta"]


def rms(x, w):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def rope(x, pos):  # x (h, T, hd), rotate-half
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
    ang = pos[:, None].float() * inv[None]
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


with torch.no_grad():
    out = main(input_ids=torch.tensor([ids]), output_hidden_states=True)
    hs = out.hidden_states[-1][0]                       # (T, H), after the final norm
    T = len(ids) - 1
    emb = main.model.embed_tokens(torch.tensor(ids[1:]))  # tokens t+1
    x = torch.cat([rms(hs[:T], W("hidden_layernorm.weight")), rms(emb, W("token_layernorm.weight"))], -1)
    x = x @ W("input_proj.weight").T
    a = rms(x, W("input_layernorm.weight"))
    q = (a @ W("self_attn.q_proj.weight").T + W("self_attn.q_proj.bias")).view(T, nh, hd).transpose(0, 1)
    k = (a @ W("self_attn.k_proj.weight").T + W("self_attn.k_proj.bias")).view(T, kvh, hd).transpose(0, 1)
    v = (a @ W("self_attn.v_proj.weight").T + W("self_attn.v_proj.bias")).view(T, kvh, hd).transpose(0, 1)
    pos = torch.arange(T)
    q, k = rope(q, pos), rope(k, pos)
    k = k.repeat_interleave(nh // kvh, 0)
    v = v.repeat_interleave(nh // kvh, 0)
    s = q @ k.transpose(1, 2) / hd ** 0.5
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    o = (s.softmax(-1) @ v).transpose(0, 1).reshape(T, nh * hd)
    x = x + o @ W("self_attn.o_proj.weight").T
    m = rms(x, W("post_attention_layernorm.weight"))
    g = torch.nn.functional.silu(m @ W("mlp.gate_proj.weight").T) * (m @ W("mlp.up_proj.weight").T)
    x = x + g @ W("mlp.down_proj.weight").T
    logits = main.lm_head(rms(x, W("final_layernorm.weight")))
    topv, topi = logits[-1].topk(5)
    print("MTP draft top5 (last row):")
    for v_, i in zip(topv.tolist(), topi.tolist()):
        print(f"  id={i:<7} logit={v_:.4f}")
