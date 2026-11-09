#!/usr/bin/env python3
"""Gemma 4 image + text reference (transformers, CPU, F32 over the BF16
weights) for models/gemma4_vision_test.go.

    .venv/bin/python scripts/verify/ref_gemma4_vl.py <repo> <dir>

Writes <dir>/image.png (synthetic 960 x 672: exactly 280 soft tokens, no
resize) and <dir>/vl.json: prompt ids, the image features after the
embedder (first 8 rows, sum and mean |x|), the last position's top-5
logits and 16 greedy tokens.
"""
import json, os, sys
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

repo, d = sys.argv[1], sys.argv[2]
os.makedirs(d, exist_ok=True)
H, W = 672, 960
y, x = np.mgrid[0:H, 0:W]
img = np.stack([(x * 255 // W), ((x // 60 + y // 60) % 2) * 170 + 50, (255 - y * 255 // H)], -1).astype(np.uint8)
Image.fromarray(img).save(os.path.join(d, "image.png"))
proc = AutoProcessor.from_pretrained(repo)
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in one sentence."}]}]
text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
inputs = proc(text=[text], images=[Image.open(os.path.join(d, "image.png")).convert("RGB")], return_tensors="pt")
model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16).eval().float()
with torch.no_grad():
    feats = model.model.get_image_features(inputs["pixel_values"].float(), inputs["image_position_ids"]).pooler_output[0]
    logits = model(**inputs).logits[0, -1]
    top = logits.topk(5)
    out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
gen = out[0, inputs["input_ids"].shape[1]:].tolist()
print("tokens", inputs["input_ids"].shape[1], "feats", tuple(feats.shape), "keys", list(inputs.keys()))
print("top5", top.indices.tolist(), "greedy", repr(proc.tokenizer.decode(gen)))
json.dump({"repo": repo, "ids": inputs["input_ids"][0].tolist(),
           "features_shape": list(feats.shape), "features_head": feats[:8].flatten().tolist(),
           "features_sum": feats.sum().item(), "features_absmean": feats.abs().mean().item(),
           "top5_ids": top.indices.tolist(), "top5_logits": top.values.tolist(),
           "greedy": gen, "greedy_text": proc.tokenizer.decode(gen)}, open(os.path.join(d, "vl.json"), "w"))
