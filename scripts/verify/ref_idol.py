#!/usr/bin/env python3
"""Tiny Idol reference (speridlabs/idol-sdk, CPU, F32) for models/idol_test.go.

    python scripts/verify/ref_idol.py <idol-sdk checkout> <out_dir> <out.json>

Builds seeded tiny versions of Idol's three networks with idol-sdk's own
code - the denoiser (LatentSeqMMFlowModel), the octree decoder and the
Gaussian decoder - in F32 (the SDK's FP16 blocks are left in F32 so the
comparison is exact up to float rounding), saves them in the layout of
speridlabs/idol (config.json + denoiser / decoder / decoder_gs
.safetensors), reloads them and records:

  denoiser    one forward (latent, camera, time, DINOv3 / FLUX.2 tokens)
  sample      3 Euler steps with CFG 3 and time shift 3 from given noise
  octree      the decoder's logits for given cells, and the anchors that
              OctreeProbabilityFixedlenDecoder.sample places with every
              torch.rand / rand_like replaced by 0.5 (deterministic)
  gaussians   the Gaussian decoder's raw outputs and the activated
              Gaussian fields for given anchors
"""
import json, os, sys

import torch

sdk, out_dir, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, sdk)
import idol.models.layers as layers

layers.convert_module_to_f16 = lambda module: None  # keep the blocks in F32
from idol.models.decoder import GaussianFixedlenDecoder, OctreeProbabilityFixedlenDecoder
from idol.models.dit import LatentSeqMMFlowModel
from idol.sampler import sample_flow
from safetensors.torch import load_file, save_file
from tensordict import TensorDict

C = 128  # model channels: 2 heads of 64 (RePo's head dim is fixed at 64)
config = {
    "denoiser": {"name": "LatentSeqMMFlowModel", "args": {
        "q_token_length": 24, "in_channels": 16, "cam_channels": 5, "model_channels": C, "cond_channels": 32,
        "cond2_channels": 8, "num_blocks": 2, "num_refiner_blocks": 2, "num_heads": 2, "mlp_ratio": 2}},
    "decoder": {"name": "OctreeProbabilityFixedlenDecoder", "args": {
        "model_channels": C, "cond_channels": 16, "num_blocks": 2, "num_heads": 2, "mlp_ratio": 2}},
    "decoder_gs": {"name": "GaussianFixedlenDecoder", "args": {
        "in_channels": 3, "model_channels": C, "cond_channels": 16, "num_blocks": 2, "num_heads": 2, "mlp_ratio": 2,
        "representation_config": {
            "num_gaussians": 4,
            "lr": {"_xyz": 1.0, "_features_dc": 1.0, "_opacity": 1.0, "_scaling": 1.0, "_rotation": 0.1},
            "perturbe_size": 1.5, "3d_filter_kernel_size": 0.0009, "scaling_bias": 0.004, "opacity_bias": 0.1}}},
    "image_size": 1024,
    "max_voxel_level": 3,
    "image_encoders": {"dinov3": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
                       "flux2_vae": "black-forest-labs/FLUX.2-dev"},
}


def randomize(model, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "freqs" in name:
                p.copy_(torch.rand_like(p) * 4 + 0.5)
            elif "norm" in name and p.dim() == 1 or "gamma" in name:
                p.copy_(1 + 0.2 * torch.randn_like(p))
            else:
                p.normal_(0, 0.08)
        for name, b in model.named_buffers():
            if name == "pos_pe":
                b.copy_(torch.rand_like(b))
            elif name == "points_offset_perturbation":
                b.copy_(torch.randn_like(b) * 0.5)
            elif name == "base_offset_scale":
                b.fill_(-2.0)
    return model.eval()


den = randomize(LatentSeqMMFlowModel(**config["denoiser"]["args"]), 1)
dec = randomize(OctreeProbabilityFixedlenDecoder(**config["decoder"]["args"]), 2)
dgs = randomize(GaussianFixedlenDecoder(**config["decoder_gs"]["args"]), 3)
os.makedirs(out_dir, exist_ok=True)
json.dump(config, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
for model, name in ((den, "denoiser"), (dec, "decoder"), (dgs, "decoder_gs")):
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(out_dir, f"{name}.safetensors"))
    model.load_state_dict(load_file(os.path.join(out_dir, f"{name}.safetensors")), strict=True)
for model in (den, dec, dgs):
    model.dtype = torch.float32

r = lambda t: [round(v, 7) for v in t.detach().flatten().tolist()]
out = {}
torch.manual_seed(10)
Lc = 7
latent, camera = torch.randn(1, 24, 16), torch.randn(1, 1, 5)
f1, f2 = torch.randn(1, Lc, 32), torch.randn(1, Lc, 8)
cond = TensorDict({"feature1": f1, "feature2": f2}, batch_size=1)
with torch.no_grad():
    v = den(TensorDict({"latent": latent, "camera": camera}, batch_size=1), torch.tensor([637.5]), cond)
out["denoiser"] = {"latent": r(latent), "camera": r(camera), "t": 637.5, "feature1": r(f1), "feature2": r(f2),
                   "cond_len": Lc, "v_latent": r(v["latent"]), "v_camera": r(v["camera"])}

noise = TensorDict({"latent": torch.randn(1, 24, 16), "camera": torch.randn(1, 1, 5)}, batch_size=1)
with torch.no_grad():
    s = sample_flow(den, noise, cond=cond, neg_cond=cond.apply(torch.zeros_like), steps=3, rescale_t=3.0,
                    cfg_strength=3.0, verbose=False)
out["sample"] = {"noise_latent": r(noise["latent"]), "noise_camera": r(noise["camera"]), "steps": 3, "cfg": 3.0,
                 "shift": 3.0, "latent": r(s["latent"]), "camera": r(s["camera"])}

cond_lat = torch.randn(1, 24, 16)
cells = torch.rand(1, 5, 3)
with torch.no_grad():
    logits = dec(cells, torch.tensor([4]), cond_lat)
real_rand, real_rand_like = torch.rand, torch.rand_like
torch.rand = lambda *a, **k: torch.full(a if a and not isinstance(a[0], (tuple, list)) else a[0], 0.5,
                                        device=k.get("device"))
torch.rand_like = lambda x, **k: torch.full_like(x, 0.5, dtype=k.get("dtype", x.dtype))
with torch.no_grad():
    anchors = OctreeProbabilityFixedlenDecoder.sample(dec, cond_lat, num_points=40, level=config["max_voxel_level"])
torch.rand, torch.rand_like = real_rand, real_rand_like
out["octree"] = {"cond": r(cond_lat), "cells": r(cells), "level": 4, "logits": r(logits), "num_points": 40,
                 "anchors": r(anchors)}

pts = torch.rand(1, 6, 3)
with torch.no_grad():
    feats = dgs(pts, cond_lat)
    g = dgs.to_representation(pts, feats)[0]
out["gaussians"] = {"points": r(pts), "features": r(feats), "xyz": r(g.xyz), "features_dc": r(g.features_dc),
                    "scaling": r(g.scaling), "rotation": r(g.rotation), "opacity": r(g.opacity)}
json.dump(out, open(out_json, "w"))
print("denoiser", tuple(v["latent"].shape), "anchors", tuple(anchors.shape), "gaussians", tuple(g.xyz.shape))
