#!/usr/bin/env python3
"""Idol's image conditioning on tiny encoders (idol-sdk + transformers +
diffusers, CPU, F32) for diffusion/idol_test.go.

    python scripts/verify/ref_idol_pipe.py <idol-sdk checkout> <out_dir> <out.json>

Saves a seeded tiny DINOv3 ViT (dinov3/) and a tiny AutoencoderKLFlux2
(vae/, 8x downsampling so its patchified grid matches DINOv3's 16-pixel
patches), then records for a 64 x 64 image:

  preprocess  idol's preprocess_image of a synthetic RGBA object (a
              soft-edged ellipse off centre) at margin 1.2
  condition   idol-sdk ImageEncoder's feature1 (DINOv3 tokens) and
              feature2 (FLUX.2 VAE tokens) for those pixels, with the VAE
              latent sampled from a recorded eps
"""
import json, os, sys

import numpy as np
import torch
from PIL import Image

sdk, out_dir, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, sdk)
from diffusers.models import AutoencoderKLFlux2
from transformers import DINOv3ViTConfig, DINOv3ViTModel

r = lambda t: [round(float(v), 7) for v in np.asarray(t, dtype=np.float64).flatten()]
S = 64

# A synthetic RGBA object: an off-centre ellipse with a soft edge and a
# colour ramp, inside a 96 x 80 canvas (the short side is resized to S).
H, W = 80, 96
yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
d = ((xx - 58) / 22) ** 2 + ((yy - 34) / 15) ** 2
alpha = np.clip((1.15 - d) * 4, 0, 1)
rgba = np.zeros((H, W, 4), np.uint8)
rgba[..., 0] = (xx / W * 255).astype(np.uint8)
rgba[..., 1] = (yy / H * 255).astype(np.uint8)
rgba[..., 2] = 180
rgba[..., 3] = (alpha * 255).round().astype(np.uint8)
os.makedirs(out_dir, exist_ok=True)
Image.fromarray(rgba, "RGBA").save(os.path.join(out_dir, "object.png"))

from idol.preprocess import preprocess_image  # noqa: E402

prepared = preprocess_image(Image.open(os.path.join(out_dir, "object.png")), background_remover=None,
                            image_size=S, crop_margin=1.2)
pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None]

torch.manual_seed(31)
dcfg = DINOv3ViTConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=96,
                       patch_size=16, num_register_tokens=4, use_gated_mlp=True, hidden_act="silu",
                       layerscale_value=0.5, image_size=S)
dino = DINOv3ViTModel(dcfg).eval()
torch.manual_seed(32)
vae = AutoencoderKLFlux2(block_out_channels=(32, 32, 32, 32), layers_per_block=1, latent_channels=4,
                         norm_num_groups=32, sample_size=S).eval()  # Light's VAEs use 32 groups
with torch.no_grad():
    for m in (dino, vae):
        for n, p in m.named_parameters():
            if "lambda1" in n:
                p.copy_(0.5 + 0.2 * torch.randn_like(p))
            elif ("norm" in n or "bn" in n) and n.endswith("weight"):
                p.copy_(1 + 0.2 * torch.randn_like(p))
            else:
                p.normal_(0, 0.1)
    vae.bn.running_mean.copy_(0.1 * torch.randn(16))
    vae.bn.running_var.copy_(1 + 0.3 * torch.rand(16))
dino.save_pretrained(os.path.join(out_dir, "dinov3"), safe_serialization=True)
vae.save_pretrained(os.path.join(out_dir, "vae"), safe_serialization=True)

from idol.conditioning import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

with torch.no_grad():
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    tok = dino(pixel_values=(pixels - mean) / std).last_hidden_state
    f1 = torch.nn.functional.layer_norm(tok, tok.shape[-1:])
    dist = vae.encode(pixels * 2 - 1).latent_dist
    torch.manual_seed(33)
    eps = torch.randn_like(dist.mean)
    lat = dist.mean + dist.std * eps
    # idol.conditioning.ImageEncoder._flux2_tokens on the sampled latent
    b, c, h, w = lat.shape
    lat = lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    bm = vae.bn.running_mean.view(1, -1, 1, 1)
    bs = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    t2 = ((lat - bm) / bs).flatten(2).transpose(1, 2)
    f2 = torch.cat([torch.zeros(b, 5, t2.shape[2]), t2], dim=1)
json.dump({"size": S, "margin": 1.2, "pixels": r(pixels), "eps": r(eps), "eps_shape": list(eps.shape),
           "feature1": r(f1), "feature1_shape": list(f1.shape), "feature2": r(f2), "feature2_shape": list(f2.shape)},
          open(out_json, "w"))
print("pixels", tuple(pixels.shape), "f1", tuple(f1.shape), "f2", tuple(f2.shape))
