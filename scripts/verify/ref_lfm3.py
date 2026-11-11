#!/usr/bin/env python3
"""LFM3 reference for models/lfm3_test.go.

No released library implements LFM3 yet (transformers has LFM2 only and the
checkpoint ships remote code), so this is an independent plain-PyTorch
implementation of LiquidAI/LFM3-1.7B's modeling_lfm3.py for single
sequences, computed in float64:

  * conv layers: in_proj -> (B, C, x); y = C * causal_depthwise_conv7(B * x);
    out_proj
  * attention layers: GQA, per-head QK RMSNorm, RoPE in the zero-tail layout
    (full_attention: the first partial_rotary_factor * head_dim / 2
    frequencies of the full-head-dim progression, the rest zero;
    sliding_attention: plain RoPE), a sliding window on sliding layers
    (key j visible from query i when i - window < j <= i), a per-head
    sigmoid output gate from gate_proj, o_proj
  * sandwich norms, SwiGLU MLP (w1, w3, w2), embedding_norm, tied head

    .venv/bin/python scripts/verify/ref_lfm3.py --make-tiny <dir> <out.json>
    .venv/bin/python scripts/verify/ref_lfm3.py --real <out.json>

--make-tiny writes a seeded random tiny checkpoint (config.json +
model.safetensors) and its logits for 12 tokens. --real loads
LiquidAI/LFM3-1.7B (BF16 weights widened to float64), and records the
logits of a chat prompt plus 8 greedy tokens.
"""
import json, os, sys
import torch
from safetensors.torch import load_file, save_file

torch.set_default_dtype(torch.float64)


def rms(x, w, eps):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def inv_freq(cfg, kind):
    p = cfg["rope_parameters"][kind]
    hd = cfg["head_dim"]
    n = int(p.get("partial_rotary_factor", 1.0) * hd) // 2
    f = 1.0 / (p["rope_theta"] ** (torch.arange(0, 2 * n, 2, dtype=torch.float64) / hd))
    return torch.cat([f, torch.zeros(hd // 2 - n)])


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def forward(cfg, W, ids):
    D, NH, KH, HD, eps = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["norm_eps"]
    T = len(ids)
    pos = torch.arange(T, dtype=torch.float64)
    g = lambda n: W[n]
    h = g("model.embed_tokens.weight")[torch.tensor(ids)]
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"model.layers.{i}."
        x = rms(h, g(p + "operator_norm.weight"), eps)
        if kind == "conv":
            bcx = x @ g(p + "conv.in_proj.weight").T
            Bm, Cm, xx = bcx[:, :D], bcx[:, D:2 * D], bcx[:, 2 * D:]
            y = Bm * xx  # (T, D)
            w = g(p + "conv.conv.weight")[:, 0, :]  # (D, K)
            K = w.shape[1]
            yp = torch.cat([torch.zeros(K - 1, D), y], 0)
            conv = torch.stack([(yp[t:t + K] * w.T).sum(0) for t in range(T)])
            o = (Cm * conv) @ g(p + "conv.out_proj.weight").T
        else:
            a = p + "self_attn."
            q = (x @ g(a + "q_proj.weight").T).view(T, NH, HD)
            k = (x @ g(a + "k_proj.weight").T).view(T, KH, HD)
            v = (x @ g(a + "v_proj.weight").T).view(T, KH, HD)
            q, k = rms(q, g(a + "q_norm.weight"), eps), rms(k, g(a + "k_norm.weight"), eps)
            fr = pos[:, None] * inv_freq(cfg, kind)[None]
            emb = torch.cat([fr, fr], -1)
            cos, sin = emb.cos()[:, None], emb.sin()[:, None]
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
            k = k.repeat_interleave(NH // KH, 1)
            v = v.repeat_interleave(NH // KH, 1)
            s = torch.einsum("thd,shd->hts", q, k) * HD ** -0.5
            i_, j_ = torch.arange(T)[:, None], torch.arange(T)[None]
            ok = j_ <= i_
            if kind == "sliding_attention":
                ok = ok & (j_ > i_ - cfg["sliding_window"])
            s = s.masked_fill(~ok, float("-inf"))
            ctx = torch.einsum("hts,shd->thd", s.softmax(-1), v)
            gate = torch.sigmoid(x @ g(a + "gate_proj.weight").T)  # (T, NH)
            o = (ctx * gate[..., None]).reshape(T, NH * HD) @ g(a + "o_proj.weight").T
        if cfg["use_sandwich_norm"]:
            o = rms(o, g(p + "post_operator_norm.weight"), eps)
        h = h + o
        f = rms(h, g(p + "ffn_norm.weight"), eps)
        f = (torch.nn.functional.silu(f @ g(p + "feed_forward.w1.weight").T) * (f @ g(p + "feed_forward.w3.weight").T)) @ g(p + "feed_forward.w2.weight").T
        if cfg["use_sandwich_norm"]:
            f = rms(f, g(p + "post_ffn_norm.weight"), eps)
        h = h + f
    h = rms(h, g("model.embedding_norm.weight"), eps)
    return h @ g("model.embed_tokens.weight").T


r = lambda t: [round(v, 6) for v in t.flatten().tolist()]

if sys.argv[1] == "--make-tiny":
    out_dir, out_json = sys.argv[2], sys.argv[3]
    cfg = {"architectures": ["Lfm3ForCausalLM"], "model_type": "lfm3", "vocab_size": 96, "hidden_size": 32,
           "intermediate_size": 48, "num_hidden_layers": 5, "num_attention_heads": 4, "num_key_value_heads": 2,
           "head_dim": 16, "conv_L_cache": 7, "conv_bias": False, "norm_eps": 1e-5, "sliding_window": 4,
           "layer_types": ["conv", "full_attention", "conv", "sliding_attention", "conv"],
           "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1000000.0,
                                                  "rope_type": "default"},
                               "sliding_attention": {"rope_theta": 10000, "rope_type": "default"}},
           "max_position_embeddings": 4096, "tie_word_embeddings": True, "use_sandwich_norm": True,
           "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0, "dtype": "float32"}
    D, NH, KH, HD, IM, V = 32, 4, 2, 16, 48, 96
    torch.manual_seed(31)
    W = {"model.embed_tokens.weight": torch.randn(V, D) * 0.5, "model.embedding_norm.weight": 1 + 0.2 * torch.randn(D)}
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"model.layers.{i}."
        for n in ["operator_norm", "ffn_norm", "post_operator_norm", "post_ffn_norm"]:
            W[p + n + ".weight"] = 1 + 0.2 * torch.randn(D)
        W[p + "feed_forward.w1.weight"] = torch.randn(IM, D) * 0.15
        W[p + "feed_forward.w3.weight"] = torch.randn(IM, D) * 0.15
        W[p + "feed_forward.w2.weight"] = torch.randn(D, IM) * 0.15
        if kind == "conv":
            W[p + "conv.in_proj.weight"] = torch.randn(3 * D, D) * 0.2
            W[p + "conv.out_proj.weight"] = torch.randn(D, D) * 0.2
            W[p + "conv.conv.weight"] = torch.randn(D, 1, 7) * 0.3
        else:
            a = p + "self_attn."
            W[a + "q_proj.weight"] = torch.randn(NH * HD, D) * 0.2
            W[a + "k_proj.weight"] = torch.randn(KH * HD, D) * 0.2
            W[a + "v_proj.weight"] = torch.randn(KH * HD, D) * 0.2
            W[a + "o_proj.weight"] = torch.randn(D, NH * HD) * 0.2
            W[a + "gate_proj.weight"] = torch.randn(NH, D) * 0.3
            W[a + "q_norm.weight"] = 1 + 0.2 * torch.randn(HD)
            W[a + "k_norm.weight"] = 1 + 0.2 * torch.randn(HD)
    W = {k: v.float() for k, v in W.items()}  # stored F32; the reference widens to F64
    os.makedirs(out_dir, exist_ok=True)
    save_file(W, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    ids = [5, 17, 3, 88, 41, 9, 60, 23, 77, 2, 14, 50]
    with torch.no_grad():
        logits = forward(cfg, {k: v.double() for k, v in W.items()}, ids)
    print("tiny logits", tuple(logits.shape), logits.abs().max().item())
    json.dump({"ids": ids, "logits": r(logits), "vocab": V}, open(out_json, "w"))
else:
    out_json = sys.argv[2]
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    repo = "LiquidAI/LFM3-1.7B"
    cfg = json.load(open(hf_hub_download(repo, "config.json")))
    W = {k: v.double() for k, v in load_file(hf_hub_download(repo, "model.safetensors")).items()}
    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    text = "<|startoftext|><|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(text, add_special_tokens=False).ids
    with torch.no_grad():
        logits = forward(cfg, W, ids)
        last = logits[-1]
        gen, cur = [], list(ids)
        for _ in range(8):
            nxt = int(forward(cfg, W, cur)[-1].argmax())
            gen.append(nxt)
            cur.append(nxt)
    top = last.topk(5)
    print("ids", ids, "greedy", gen, repr(tok.decode(gen)), "top", top.indices.tolist(), top.values.tolist())
    json.dump({"text": text, "ids": ids, "top_ids": top.indices.tolist(), "top_logits": top.values.tolist(),
               "greedy": gen}, open(out_json, "w"))
