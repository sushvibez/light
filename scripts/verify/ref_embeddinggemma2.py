#!/usr/bin/env python3
"""EmbeddingGemma 2 reference embeddings (transformers, eager, text-only) for
models/embeddinggemma2_test.go.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify/ref_embeddinggemma2.py [repo] [bf16|f32]

Loads the text backbone only (vision/audio towers disabled), runs each text
unpadded, mean-pools the projected hidden states and L2-normalises (the
sentence-transformers 1_Pooling + 2_Normalize modules). Prints token ids,
the first 8 embedding values, the embedding norm before normalisation and
the cosine similarity matrix.
"""
import sys, torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

repo = sys.argv[1] if len(sys.argv) > 1 else "unsloth/embeddinggemma-2"
dt = torch.float32 if (len(sys.argv) > 2 and sys.argv[2] == "f32") else torch.bfloat16
tok = AutoTokenizer.from_pretrained(repo)
cfg = AutoConfig.from_pretrained(repo, vision_config=None, audio_config=None)
model = AutoModel.from_pretrained(repo, config=cfg, dtype=dt, attn_implementation="eager").to("cuda").eval()
lm = model.language_model
print("text params %.1f M" % (sum(p.numel() for p in lm.parameters()) / 1e6))
re = lm.rotary_emb
for lt in sorted(set(cfg.text_config.layer_types)):
    f = getattr(re, f"{lt}_inv_freq")
    print(f"{lt} inv_freq len={f.numel()} nonzero={(f != 0).sum().item()} first={f[:2].tolist()} last={f[-1].item():.3e}")

texts = [
    "task: search result | query: What causes the northern lights?",
    "title: none | text: The northern lights are caused by charged particles from the sun colliding with gases in the upper atmosphere.",
    "title: none | text: A sourdough starter needs regular feeding with flour and water to stay active.",
]
embs = []
with torch.no_grad():
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("cuda")
        print("ids:", ids[0].tolist())
        h = model(input_ids=ids).last_hidden_state[0].float()  # (T, 768)
        e = h.mean(0)
        n = e.norm()
        print(f"  norm={n.item():.4f} first8={[round(v, 4) for v in (e / n)[:8].tolist()]}")
        embs.append(e / n)
E = torch.stack(embs)
print("cosine:\n", (E @ E.T).tolist())
