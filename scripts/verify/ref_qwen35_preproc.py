#!/usr/bin/env python3
"""Qwen3.5 image preprocessing reference for models/qwen35_vision_test.go.

    .venv/bin/python scripts/verify/ref_qwen35_preproc.py <repo> <dir>

Writes <dir>/odd.png (a synthetic 300 x 221 image the processor must
resize) and <dir>/odd.u8: the processor's patches mapped back to uint8
(v * 127.5 + 127.5, first temporal copy: C, P, P per patch), plus the grid
in <dir>/odd.json.
"""
import json, os, sys
import numpy as np
from PIL import Image
from transformers import AutoProcessor

repo, d = sys.argv[1], sys.argv[2]
H, W = 221, 300
y, x = np.mgrid[0:H, 0:W]
img = np.stack([(x * 7 + y * 3) % 256, (x * x // 50 + y) % 256, ((x // 9 + y // 7) % 2) * 220 + 15], -1).astype(np.uint8)
Image.fromarray(img).save(os.path.join(d, "odd.png"))
ip = AutoProcessor.from_pretrained(repo).image_processor
out = ip(images=[Image.open(os.path.join(d, "odd.png")).convert("RGB")], return_tensors="pt")
pv, grid = out["pixel_values"].numpy(), out["image_grid_thw"].tolist()
P, T = 16, 2
u8 = np.rint(pv.reshape(-1, 3, T, P, P)[:, :, 0] * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
u8.tofile(os.path.join(d, "odd.u8"))
json.dump({"grid_thw": grid}, open(os.path.join(d, "odd.json"), "w"))
print("grid", grid, "patches", pv.shape)
