#!/usr/bin/env python3
"""GLM-5.3-Flash real-weight check on a truncated model (Colab, high-mem CPU).

    python scripts/colab/glm5_check.py <work_dir> [layers]

Downloads only the shards holding the embeddings, the first `layers`
decoder layers (default 4: three Kimi delta-attention layers with dense
MLPs, then the first MLA layer with the first MoE layer) and the LM head,
writes a truncated copy of the checkpoint (config with num_hidden_layers =
layers), and records transformers' logits for a short prompt with the FP8
weights dequantized (BF16) - <work_dir>/ref.json. models/glm5_real_test.go
(LIGHT_GLM5_TRUNC=<work_dir>) loads the same shards with Light and compares.
"""
import json, os, re, sys

import torch
from huggingface_hub import hf_hub_download

REPO = "zai-org/GLM-5.3-Flash"
work = sys.argv[1]
layers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.makedirs(work, exist_ok=True)
idx = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))
wm = idx["weight_map"]
keep = {}
for k, f in wm.items():
    if k.startswith("model.visual") or "mtp" in k:
        continue
    m = re.search(r"language_model\.layers\.(\d+)\.", k)
    if m and int(m.group(1)) >= layers:
        continue
    keep[k] = f
shards = sorted(set(keep.values()))
print("shards", shards, flush=True)
for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"] + shards:
    p = hf_hub_download(REPO, f)
    dst = os.path.join(work, f)
    if not os.path.exists(dst):
        os.symlink(p, dst)
cfg = json.load(open(os.path.join(work, "config.json")))
os.remove(os.path.join(work, "config.json"))
t = cfg["text_config"]
t["num_hidden_layers"] = layers
for key in ["layer_types", "mlp_layer_types", "indexer_types"]:
    if isinstance(t.get(key), list):
        t[key] = t[key][:layers]
t["num_nextn_predict_layers"] = 0
json.dump(cfg, open(os.path.join(work, "config.json"), "w"), indent=2)
json.dump({"metadata": idx.get("metadata", {}), "weight_map": keep},
          open(os.path.join(work, "model.safetensors.index.json"), "w"))

from transformers import AutoTokenizer, FineGrainedFP8Config  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration  # noqa: E402

tok = AutoTokenizer.from_pretrained(work)
ids = tok("The capital of France is Paris. The capital of Germany is", return_tensors="pt").input_ids
model = Glm5NextForConditionalGeneration.from_pretrained(
    work, dtype=torch.bfloat16, quantization_config=FineGrainedFP8Config(dequantize=True)).eval()
with torch.no_grad():
    logits = model(input_ids=ids).logits[0].float()
last = logits[-1]
top = last.topk(10)
print("prompt ids", ids[0].tolist(), "top10", top.indices.tolist(), flush=True)
json.dump({"ids": ids[0].tolist(), "last_logits": [round(x, 5) for x in last.tolist()],
           "all_logits_l2": [round(x, 4) for x in logits.norm(dim=-1).tolist()],
           "top10": top.indices.tolist(), "layers": layers},
          open(os.path.join(work, "ref.json"), "w"))
print("wrote", os.path.join(work, "ref.json"))
