#!/usr/bin/env python3
"""Self-contained numpy Llama-family reference (no torch / transformers needed).

Reads config.json + model.safetensors directly and runs the forward in numpy,
printing the top-k next-token logits. Handy as a golden reference when torch is
unavailable, and for layer-by-layer debugging (set LAYERS=1 to dump per-layer
last-position norms).

Usage:
    python ref_numpy.py <model_dir> "The capital of France is" [bos_id]
"""
import sys, os, json, struct, glob
import numpy as np


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        data = f.read()
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        dt = v["dtype"]; s, e = v["data_offsets"]; raw = data[s:e]
        if dt == "BF16":
            arr = ((np.frombuffer(raw, np.uint16).astype(np.uint32)) << 16).view(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(raw, np.float32)
        elif dt == "F16":
            arr = np.frombuffer(raw, np.float16).astype(np.float32)
        else:
            continue
        out[k] = arr.reshape(v["shape"]).astype(np.float32)
    return out


def llama3_inv_freq(hd, base, scaling):
    inv = 1.0 / (base ** (np.arange(0, hd, 2) / hd))
    if not scaling:
        return inv
    factor = scaling["factor"]; lo = scaling["low_freq_factor"]; hi = scaling["high_freq_factor"]
    omax = scaling["original_max_position_embeddings"]
    low_wl, high_wl = omax / lo, omax / hi
    wl = 2 * np.pi / inv
    smooth = (omax / wl - lo) / (hi - lo)
    return np.where(wl > low_wl, inv / factor,
                    np.where(wl < high_wl, inv, (1 - smooth) * (inv / factor) + smooth * inv))


def main():
    d = sys.argv[1]
    prompt_ids = None
    cfg = json.load(open(os.path.join(d, "config.json")))
    H = cfg["hidden_size"]; L = cfg["num_hidden_layers"]; nH = cfg["num_attention_heads"]
    nKV = cfg["num_key_value_heads"]; hd = cfg.get("head_dim", H // nH)
    eps = cfg["rms_norm_eps"]; base = cfg.get("rope_theta", 10000.0)
    scaling = cfg.get("rope_scaling")

    # tokenize via the model's tokenizer.json vocab (BPE byte-level), or accept ids
    ids = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
    if ids is None:
        # crude: require explicit ids; tokenization left to the Go side / transformers
        print("pass comma-separated token ids as arg3 (e.g. from the Go harness)")
        sys.exit(1)
    S = len(ids)

    W = load_safetensors(glob.glob(os.path.join(d, "*.safetensors"))[0])
    rms = lambda x, w: x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
    inv = llama3_inv_freq(hd, base, scaling)
    ang = np.arange(S)[:, None] * inv[None, :]
    cos = np.concatenate([np.cos(ang)] * 2, -1); sin = np.concatenate([np.sin(ang)] * 2, -1)
    rot = lambda x: np.concatenate([-x[..., hd // 2:], x[..., :hd // 2]], -1)

    x = W["model.embed_tokens.weight"][ids]
    for i in range(L):
        p = f"model.layers.{i}"
        h = rms(x, W[p + ".input_layernorm.weight"])
        q = (h @ W[p + ".self_attn.q_proj.weight"].T).reshape(S, nH, hd).transpose(1, 0, 2)
        k = (h @ W[p + ".self_attn.k_proj.weight"].T).reshape(S, nKV, hd).transpose(1, 0, 2)
        v = (h @ W[p + ".self_attn.v_proj.weight"].T).reshape(S, nKV, hd).transpose(1, 0, 2)
        q = q * cos + rot(q) * sin; k = k * cos + rot(k) * sin
        k = np.repeat(k, nH // nKV, 0); v = np.repeat(v, nH // nKV, 0)
        sc = (q @ k.transpose(0, 2, 1)) / np.sqrt(hd) + np.triu(np.full((S, S), -1e9), 1)[None]
        sc -= sc.max(-1, keepdims=True); a = np.exp(sc); a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(S, H) @ W[p + ".self_attn.o_proj.weight"].T
        x = x + o
        h = rms(x, W[p + ".post_attention_layernorm.weight"])
        g = h @ W[p + ".mlp.gate_proj.weight"].T; u = h @ W[p + ".mlp.up_proj.weight"].T
        x = x + ((g / (1 + np.exp(-g))) * u) @ W[p + ".mlp.down_proj.weight"].T
        if os.environ.get("LAYERS"):
            print(f"[ref] L{i:02d} norm={np.linalg.norm(x[-1]):.4f}")
    x = rms(x, W["model.norm.weight"])
    head = W["lm_head.weight"] if "lm_head.weight" in W else W["model.embed_tokens.weight"]
    logits = x[-1] @ head.T
    top = np.argsort(-logits)[:5]
    print("REFERENCE top5 (numpy):")
    for t in top:
        print(f"  id={int(t):<7} logit={float(logits[t]):.4f}")


if __name__ == "__main__":
    main()
