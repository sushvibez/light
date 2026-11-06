#!/usr/bin/env python3
"""SDAR (block-diffusion LM, Qwen3 body) reference for models/sdar_test.go.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify/ref_sdar.py [repo] [bf16|f32] [cuda|cpu]

Loads the SDAR weights into transformers' Qwen3ForCausalLM (identical
tensor names; SDAR's modeling file only swaps the attention mask and adds
a no-store KV path) and re-implements generate.py's loop here with explicit
block-bidirectional 4-D masks and DynamicCache.crop() for the "retrieve
without store" passes:
  - mask: key j visible to query i iff block(j) <= block(i), blocks of
    `block_length` absolute positions;
  - prefill: the complete prompt blocks, stored;
  - per block: up to denoising_steps+1 passes; greedy x0 = argmax, confidence
    = softmax prob of x0 at masked positions; static schedule commits
    block_length // steps (+1 for the first block_length % steps steps)
    highest-confidence tokens per pass; a final stored pass once no mask
    remains; stop at block granularity on EOS.
Prints the prompt ids, the first denoising pass' top-5 at each masked
position of block 0, and the greedy tokens.
"""
import os, sys, json, torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache
from huggingface_hub import snapshot_download

repo = sys.argv[1] if len(sys.argv) > 1 else "JetLM/SDAR-1.7B-Chat"
dt = torch.float32 if (len(sys.argv) > 2 and sys.argv[2] == "f32") else torch.bfloat16
dev = sys.argv[3] if len(sys.argv) > 3 else "cuda"
path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
raw = json.load(open(os.path.join(path, "config.json")))
cfg = Qwen3Config(**{k: v for k, v in raw.items() if k not in ("architectures", "auto_map", "model_type", "fuse_cross_entropy", "transformers_version")})
cfg._attn_implementation = "eager"
model = Qwen3ForCausalLM.from_pretrained(path, config=cfg, dtype=dt).to(dev).eval()
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
MASK = 151669
B, STEPS, GEN = 4, 4, 24
msgs = [{"role": "user", "content": "What is 2+2?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
if not isinstance(ids, list):
    ids = ids["input_ids"]
print("ids", ids)

def block_mask(q_lo, q_hi, kv_len):
    # (1, 1, Tq, kv_len) additive mask, blocks of B absolute positions.
    qpos = torch.arange(q_lo, q_hi, device=dev)
    kpos = torch.arange(0, kv_len, device=dev)
    ok = (kpos[None, :] // B) <= (qpos[:, None] // B)
    m = torch.zeros(q_hi - q_lo, kv_len, dtype=dt, device=dev)
    m[~ok] = float("-inf")
    return m[None, None]

cache = DynamicCache()
def fwd(tokens, prior, store):
    x = torch.tensor([tokens], device=dev)
    T = len(tokens)
    pos = torch.arange(prior, prior + T, device=dev)[None]
    with torch.no_grad():
        out = model(input_ids=x, position_ids=pos, attention_mask=block_mask(prior, prior + T, prior + T),
                    past_key_values=cache, use_cache=True)
    if not store:
        cache.crop(-(len(tokens)))
    return out.logits[0].float()

def schedule(n, steps):
    base, extra = n // steps, n % steps
    return [base + (1 if s < extra else 0) for s in range(steps)]

prefill = len(ids) // B * B
prior = 0
if prefill:
    fwd(ids[:prefill], 0, True); prior = prefill
rem = ids[prefill:]
gen = []
first_pass_logged = False
while len(gen) < GEN:
    cur = (rem if prior == prefill and rem else []) + [MASK] * (B - (len(rem) if prior == prefill else 0))
    if prior != prefill:
        cur = [MASK] * B
    nmask = sum(1 for t in cur if t == MASK)
    sched = schedule(nmask, STEPS)
    for step in range(STEPS + 1):
        masked = [i for i, t in enumerate(cur) if t == MASK]
        if not masked:
            fwd(cur, prior, True)
            break
        logits = fwd(cur, prior, False)
        probs = torch.softmax(logits, dim=-1)
        x0 = probs.argmax(-1)
        conf = probs.gather(-1, x0[:, None])[:, 0]
        if not first_pass_logged:
            for i in masked:
                v, ix = logits[i].topk(5)
                print(f"block0 pass0 pos{i} top5", [(int(a), round(float(b), 4)) for a, b in zip(ix, v)])
            first_pass_logged = True
        k = min(sched[min(step, STEPS - 1)], len(masked))
        order = sorted(masked, key=lambda i: -float(conf[i]))[:k]
        for i in order:
            cur[i] = int(x0[i])
    new = cur[len(rem):] if prior == prefill and rem else cur
    gen += new
    prior += B
    rem = []
    if any(t in (151645, 151643) for t in new):
        break
print("greedy ids", gen)
print("greedy text", repr(tok.decode(gen, skip_special_tokens=False)))
