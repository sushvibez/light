#!/usr/bin/env python3
"""Qwen3.5 image + text reference (transformers, CPU, F32) for
models/qwen35_vision_test.go.

    .venv/bin/python scripts/verify/ref_qwen35_vl.py <repo> <dir>

Uses <dir>/image.png (written by ref_qwen35_vision.py), builds the chat
prompt with the image and a question, and writes <dir>/vl.json: input ids,
grid, MRoPE position ids (get_rope_index), the last position's top-5 logits
and 16 greedy tokens. The processor's patches equal pixel_values.f32 (same
image), which the test reuses.
"""
import json, os, sys
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

repo, d = sys.argv[1], sys.argv[2]
img = Image.open(os.path.join(d, "image.png")).convert("RGB")
proc = AutoProcessor.from_pretrained(repo)
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in one sentence."}]}]
text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
inputs = proc(text=[text], images=[img], return_tensors="pt")
model = Qwen3_5ForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.float32).eval()
ids = inputs["input_ids"]
with torch.no_grad():
    pos, delta = model.model.get_rope_index(ids, inputs["mm_token_type_ids"], image_grid_thw=inputs["image_grid_thw"])
    logits = model(**inputs).logits[0, -1]
    top = logits.topk(5)
    out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
gen = out[0, ids.shape[1]:].tolist()
print("tokens", ids.shape[1], "grid", inputs["image_grid_thw"].tolist(), "delta", delta.tolist())
print("top5", top.indices.tolist(), [round(v, 4) for v in top.values.tolist()])
print("greedy", repr(proc.tokenizer.decode(gen)))
json.dump({"repo": repo, "ids": ids[0].tolist(), "grid_thw": inputs["image_grid_thw"].tolist(),
           "pixel_values_shape": list(inputs["pixel_values"].shape),
           "position_ids": pos[:, 0].tolist(), "delta": int(delta[0, 0]),
           "top5_ids": top.indices.tolist(), "top5_logits": top.values.tolist(), "greedy": gen,
           "greedy_text": proc.tokenizer.decode(gen)},
          open(os.path.join(d, "vl.json"), "w"))
