#!/usr/bin/env python3
"""Qwen3.5 vision tower reference (transformers, CPU, F32) for
models/qwen35_vision_test.go.

    .venv/bin/python scripts/verify/ref_qwen35_vision.py <repo> <out_dir>

Builds a deterministic 320 x 256 synthetic RGB image, runs the repo's
image processor and the vision tower (model.visual) in F32, and writes
image.png, pixel_values.f32 (patches, row-major), merged.f32 (the merger's
output: one row per LM token) and meta.json (shapes, grid_thw).
"""
import json, os, sys
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

repo, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
H, W = 256, 320
y, x = np.mgrid[0:H, 0:W]
img = np.stack([(x * 255 // W), (y * 255 // H), ((x // 16 + y // 16) % 2) * 200 + 30], -1).astype(np.uint8)
Image.fromarray(img).save(os.path.join(out, "image.png"))
proc = AutoProcessor.from_pretrained(repo)
inputs = proc.image_processor(images=[Image.fromarray(img)], return_tensors="pt")
pv, grid = inputs["pixel_values"].float(), inputs["image_grid_thw"]
model = Qwen3_5ForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.float32).eval()
with torch.no_grad():
    merged = model.model.visual(pv, grid_thw=grid).pooler_output
print("pixel_values", tuple(pv.shape), "grid", grid.tolist(), "merged", tuple(merged.shape), merged.abs().mean().item())
pv.numpy().astype("<f4").tofile(os.path.join(out, "pixel_values.f32"))
merged.numpy().astype("<f4").tofile(os.path.join(out, "merged.f32"))
json.dump({"repo": repo, "pixel_values_shape": list(pv.shape), "grid_thw": grid.tolist(),
           "merged_shape": list(merged.shape)}, open(os.path.join(out, "meta.json"), "w"))
