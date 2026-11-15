#!/usr/bin/env python3
"""Tiny BiRefNet reference (idol-sdk's idol/models/birefnet.py, CPU, F32) for
models/birefnet_test.go.

    python scripts/verify/ref_birefnet.py <birefnet.py> <out_dir> <out.json>

idol-sdk's module hardcodes Swin-L and the decoder widths, so the script
execs a copy with these substitutions (each checked to apply):

  Swin    embed_dim 192 -> 16, heads [6, 12, 24, 48] -> [1, 2, 4, 8],
          window 12 -> 4 (depths stay [2, 2, 18, 2] -> [2, 2, 2, 2]:
          two blocks per stage, so every stage has a shifted window)
  ASPP    inter 256 -> 32;  decoder / input blocks inter_channels 64 -> 16
  channels (3072, 1536, 768, 384) -> (256, 128, 64, 32): twice the tiny
          Swin's widths, as mul_scl_ipt='cat' doubles them

The gdt width (16) and the _image2patches channel counts (3·2^10 ...) stay:
they do not depend on the backbone width. Weights are seeded random
(BatchNorm affine and running statistics non-trivial), stored like
speridlabs/idol's birefnet.safetensors (F16; the reference runs on the F16
values in F32). Records:

  forward     the sigmoid map for a 64 x 128 input (at 64 x 128 the Swin
              stages are 16x32, 8x16, 4x8, 2x4 and 8x16 ... 1x2 at half
              resolution, so the last ones are padded to the window)
  remove_bg   remove_background on a 37 x 50 RGB image with INPUT_SIZE set
              to 64 x 128: the uint8 alpha
"""
import json, os, re, sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

src_path, out_dir, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(src_path, encoding="utf-8").read()
subs = [
    ("embed_dim = 192", "embed_dim = 16", 1),
    ("depths = [2, 2, 18, 2]", "depths = [2, 2, 2, 2]", 1),
    ("num_heads = [6, 12, 24, 48]", "num_heads = [1, 2, 4, 8]", 1),
    ("window_size = 12", "window_size = 4", 1),
    ("inter = 256", "inter = 32", 1),
    ("inter_channels=64", "inter_channels=16", 7),
    ("(3072, 1536, 768, 384)", "(256, 128, 64, 32)", 2),
]
for old, new, n in subs:
    got = src.count(old)
    assert got == n, (old, got, n)
    src = src.replace(old, new)
mod = {"__name__": "birefnet_tiny"}
exec(compile(src, "birefnet_tiny.py", "exec"), mod)

torch.manual_seed(0)
model = mod["BiRefNet"]()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "relative_position_bias_table" in name:
            p.normal_(0, 0.5)
        elif re.search(r"\.(bn\w*|global_avg_pool\.2|gdt_convs_\d\.1)\.weight$", name) or (
                "norm" in name and p.dim() == 1 and name.endswith("weight")):
            p.copy_(1 + 0.2 * torch.randn_like(p))
        elif p.dim() == 1:
            p.normal_(0, 0.1)
        elif p.dim() == 2:
            p.normal_(0, 0.1)
        # conv weights keep PyTorch's default init
    for name, b in model.named_buffers():
        if name.endswith("running_mean"):
            b.normal_(0, 0.1)
        elif name.endswith("running_var"):
            b.uniform_(0.5, 1.5)
    model.decoder.conv_out1[0].weight.mul_(8)  # logits spread over a few units, not all near 0
model.eval()

os.makedirs(out_dir, exist_ok=True)
sd = {k: (v.half() if v.is_floating_point() else v).contiguous() for k, v in model.state_dict().items()}
path = os.path.join(out_dir, "birefnet.safetensors")
save_file(sd, path)
model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in load_file(path).items()})
model = model.float().eval()

g = torch.Generator().manual_seed(1)
x = torch.randn(1, 3, 64, 128, generator=g)
with torch.no_grad():
    y = model(x)
print("forward: mean %.4f min %.4f max %.4f" % (y.mean(), y.min(), y.max()))

rgb = (torch.rand(37, 50, 3, generator=g) * 255).to(torch.uint8).numpy()
model.INPUT_SIZE = (64, 128)
alpha = np.array(model.remove_background(Image.fromarray(rgb, "RGB")).getchannel(3))
print("remove_bg alpha: mean %.1f min %d max %d" % (alpha.mean(), alpha.min(), alpha.max()))

json.dump({
    "input": x.flatten().tolist(), "input_shape": list(x.shape),
    "forward": y.flatten().tolist(),
    "rgb": rgb.flatten().tolist(), "rgb_shape": list(rgb.shape),
    "alpha": alpha.flatten().tolist(),
}, open(out_json, "w"))
print("wrote", path, out_json)
