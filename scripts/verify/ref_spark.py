#!/usr/bin/env python3
"""Spark-X (model_type spark2_5) reference for models/spark_test.go.

The checkpoint ships remote code only, so this is an independent
plain-PyTorch port of XHToken/Spark-X2.6-4B's modeling_spark.py for single
sequences, in float64:

  * fused q_k_v_proj split into q, k, v (GQA); RoPE on the first
    partial_rotary_factor * head_dim dims of each head (frequencies over
    that rotary width, rotate_half within it), per layer type
  * sliding layers: key j visible from query i when i - window < j <= i
  * a per-head sigmoid gate g_proj(x) on the attention output, out_proj
  * pre-norm residual blocks, GELU (erf) gated MLP, final norm, tied head

    .venv/bin/python scripts/verify/ref_spark.py --make-tiny <dir> <out.json>
    .venv/bin/python scripts/verify/ref_spark.py --real <out.json>

--real keeps the BF16 weights and widens each one to float64 when used.
"""
import json, math, os, sys
import torch
from safetensors.torch import load_file, save_file

torch.set_default_dtype(torch.float64)


def rms(x, w, eps):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def rope(x, pos, theta, partial):
    hd = x.shape[-1]
    rd = int(hd * partial)
    inv = 1.0 / (theta ** (torch.arange(0, rd, 2, dtype=torch.float64) / rd))
    fr = pos[:, None] * inv[None]
    emb = torch.cat([fr, fr], -1)
    cos, sin = emb.cos()[:, None], emb.sin()[:, None]  # (T, 1, rd)
    r, p = x[..., :rd], x[..., rd:]
    return torch.cat([r * cos + rotate_half(r) * sin, p], -1)


def forward(cfg, g, ids):
    D, NH, KH, HD, eps = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"]
    T = len(ids)
    pos = torch.arange(T, dtype=torch.float64)
    h = g("model.embedding.weight")[torch.tensor(ids)]
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"model.layers.{i}."
        rp = cfg["rope_parameters"][kind]
        x = rms(h, g(p + "input_layernorm.weight"), eps)
        qkv = x @ g(p + "self_attn.q_k_v_proj.weight").T
        q = qkv[:, :NH * HD].view(T, NH, HD)
        k = qkv[:, NH * HD:(NH + KH) * HD].view(T, KH, HD)
        v = qkv[:, (NH + KH) * HD:].view(T, KH, HD)
        q = rope(q, pos, rp["rope_theta"], rp.get("partial_rotary_factor", 1.0))
        k = rope(k, pos, rp["rope_theta"], rp.get("partial_rotary_factor", 1.0))
        k = k.repeat_interleave(NH // KH, 1)
        v = v.repeat_interleave(NH // KH, 1)
        s = torch.einsum("thd,shd->hts", q, k) / math.sqrt(HD)
        i_, j_ = torch.arange(T)[:, None], torch.arange(T)[None]
        ok = j_ <= i_
        if kind == "sliding_attention":
            ok = ok & (j_ > i_ - cfg["sliding_window"])
        s = s.masked_fill(~ok, float("-inf"))
        ctx = torch.einsum("hts,shd->thd", s.softmax(-1), v)
        gate = torch.sigmoid(x @ g(p + "self_attn.g_proj.weight").T)  # (T, NH)
        o = (ctx * gate[..., None]).reshape(T, NH * HD) @ g(p + "self_attn.out_proj.weight").T
        h = h + o
        f = rms(h, g(p + "post_attention_layernorm.weight"), eps)
        f = (torch.nn.functional.gelu(f @ g(p + "mlp.gate_proj.weight").T) * (f @ g(p + "mlp.up_proj.weight").T)) @ g(p + "mlp.down_proj.weight").T
        h = h + f
    h = rms(h, g("model.norm.weight"), eps)
    return h @ g("model.embedding.weight").T


r = lambda t: [round(v, 6) for v in t.flatten().tolist()]

if sys.argv[1] == "--make-tiny":
    out_dir, out_json = sys.argv[2], sys.argv[3]
    D, NH, KH, HD, IM, V = 32, 4, 2, 16, 64, 96
    cfg = {"architectures": ["Spark2_5ForCausalLM"], "model_type": "spark2_5", "vocab_size": V, "hidden_size": D,
           "intermediate_size": IM, "num_hidden_layers": 4, "num_attention_heads": NH, "num_key_value_heads": KH,
           "head_dim": HD, "rms_norm_eps": 1e-6, "max_position_embeddings": 4096, "tie_word_embeddings": True,
           "sliding_window": 4, "hidden_act": "gelu", "attention_bias": False, "mlp_bias": False,
           "headwise_attn_output_gate": True, "gate_attn_act_mode": "sigmoid",
           "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
           "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
                               "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000}},
           "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2, "dtype": "float32"}
    torch.manual_seed(41)
    W = {"model.embedding.weight": torch.randn(V, D) * 0.5, "model.norm.weight": 1 + 0.2 * torch.randn(D)}
    for i in range(4):
        p = f"model.layers.{i}."
        W[p + "input_layernorm.weight"] = 1 + 0.2 * torch.randn(D)
        W[p + "post_attention_layernorm.weight"] = 1 + 0.2 * torch.randn(D)
        W[p + "self_attn.q_k_v_proj.weight"] = torch.randn((NH + 2 * KH) * HD, D) * 0.2
        W[p + "self_attn.g_proj.weight"] = torch.randn(NH, D) * 0.3
        W[p + "self_attn.out_proj.weight"] = torch.randn(D, NH * HD) * 0.2
        W[p + "mlp.gate_proj.weight"] = torch.randn(IM, D) * 0.15
        W[p + "mlp.up_proj.weight"] = torch.randn(IM, D) * 0.15
        W[p + "mlp.down_proj.weight"] = torch.randn(D, IM) * 0.15
    W = {k: v.float() for k, v in W.items()}
    os.makedirs(out_dir, exist_ok=True)
    save_file(W, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    ids = [5, 17, 3, 88, 41, 9, 60, 23, 77, 2, 14, 50]
    with torch.no_grad():
        logits = forward(cfg, lambda n: W[n].double(), ids)
    print("tiny logits", tuple(logits.shape), logits.abs().max().item())
    json.dump({"ids": ids, "logits": r(logits), "vocab": V}, open(out_json, "w"))
else:
    out_json = sys.argv[2]
    from huggingface_hub import hf_hub_download, snapshot_download
    from tokenizers import Tokenizer
    repo = "XHToken/Spark-X2.6-4B"
    d = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.jinja"])
    cfg = json.load(open(os.path.join(d, "config.json")))
    W = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            W.update(load_file(os.path.join(d, f)))
    tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    text = "<｜start▁of▁sentence｜><|User|>What is the capital of France?<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>"
    ids = tok.encode(text, add_special_tokens=False).ids
    g = lambda n: W[n].double()
    with torch.no_grad():
        last = forward(cfg, g, ids)[-1]
        gen, cur = [], list(ids)
        for _ in range(8):
            nxt = int(forward(cfg, g, cur)[-1].argmax())
            gen.append(nxt)
            cur.append(nxt)
    top = last.topk(5)
    print("ids", ids, "greedy", gen, repr(tok.decode(gen)), "top", top.indices.tolist(), top.values.tolist())
    json.dump({"text": text, "ids": ids, "top_ids": top.indices.tolist(), "top_logits": top.values.tolist(),
               "greedy": gen}, open(out_json, "w"))
