#!/usr/bin/env python3
"""Real-weight Idol reference (speridlabs/idol-sdk on CUDA) for
models.TestIdolRealMatchesSDK.

    python scripts/colab/idol_check.py <idol-sdk checkout> <image.png> <out_dir>

Runs idol-sdk's own networks on the speridlabs/idol weights with fixed
inputs and saves them with the outputs to <out_dir>/ref.safetensors:

  pixels               <image.png> preprocessed by the SDK (1, 3, 1024, 1024)
  feature1, feature2   the SDK's image tokens for them (DINOv3 in BF16, its
                       default), feature1_f32 with DINOv3 in F32
  latent0, camera0     seeded noise
  fwd_*                one denoiser forward at t = 700 (cond tokens)
  sample_*             50 Euler steps, CFG 3, shift 3 from the noise
  anchors, gs_*        the Gaussian decoder's raw outputs for seeded anchors
                       given sample_latent_f16

each in two precisions: *_f16 is the SDK as shipped (FP16 blocks, the
pipeline's sampling settings: batched CFG, TF32, cuDNN attention, without
torch.compile) and *_f32 the same networks with every block in F32 (no TF32),
the closest thing to the exact result. Light's error against *_f32 next to
the SDK's own (f16 vs f32) shows whether Light is as accurate as upstream.
Also prints the SDK's 50-step sampling time with torch.compile (warm).
With COND_ONLY=1 it only adds pixels / feature1_f32 to an existing
ref.safetensors (the image tokens are deterministic; the rest is kept).
"""
import os, sys, time

import numpy as np
import torch
from PIL import Image

sdk, image, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, sdk)
import idol.models.decoder as decoder_mod
import idol.models.dit as dit_mod
import idol.models.layers as layers
from huggingface_hub import snapshot_download
from idol.conditioning import ImageEncoder
from idol.models import GaussianFixedlenDecoder, LatentSeqMMFlowModel
from idol.pipeline import compile_for_sampling
from idol.preprocess import preprocess_image
from idol.sampler import sample_flow
from safetensors.torch import load_file, save_file
from tensordict import TensorDict

dev = torch.device("cuda")
wdir = snapshot_download("speridlabs/idol", allow_patterns=["config.json", "*.safetensors"])
import json
cfg = json.load(open(os.path.join(wdir, "config.json")))


def build(cls, args, name, f16):
    conv = layers.convert_module_to_f16
    if not f16:
        dit_mod.convert_module_to_f16 = decoder_mod.convert_module_to_f16 = lambda m: None
    try:
        m = cls(**args)
    finally:
        dit_mod.convert_module_to_f16 = decoder_mod.convert_module_to_f16 = conv
    m.load_state_dict(load_file(os.path.join(wdir, f"{name}.safetensors")), strict=True)
    if not f16:
        m.float()
        m.dtype = torch.float32
    return m.eval().requires_grad_(False).to(dev)


torch.manual_seed(0)
enc = ImageEncoder(cfg["image_encoders"]["dinov3"], "black-forest-labs/FLUX.2-klein-4B", dinov3_dtype="bf16", device=dev)
prepared = preprocess_image(Image.open(image), background_remover=None, image_size=cfg["image_size"], crop_margin=1.2)
pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None].to(dev)
with torch.no_grad():
    cond = enc(pixels)
    # DINOv3 in F32 (its checkpoint's dtype), no TF32: the exact tokens.
    from transformers import AutoModel
    enc.dinov3 = AutoModel.from_pretrained(cfg["image_encoders"]["dinov3"], dtype=torch.float32).eval().to(dev)
    torch.backends.cudnn.allow_tf32, cudnn_tf32 = False, torch.backends.cudnn.allow_tf32
    feature1_f32 = enc._dinov3_tokens(pixels).float().cpu()
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
cond = TensorDict({k: v.float() for k, v in cond.items()}, batch_size=1, device=dev)
del enc
torch.cuda.empty_cache()
os.makedirs(out_dir, exist_ok=True)
ref_path = os.path.join(out_dir, "ref.safetensors")
if os.environ.get("COND_ONLY"):
    ref = load_file(ref_path)
    d = (ref["feature1"] - cond["feature1"].cpu()).abs().max()
    print(f"feature1 vs the existing reference: max diff {d:.3g}")
    ref.update(pixels=pixels.cpu(), feature1_f32=feature1_f32)
    save_file({k: v.contiguous() for k, v in ref.items()}, ref_path)
    print("updated", ref_path)
    sys.exit()

a = cfg["denoiser"]["args"]
g = torch.Generator().manual_seed(1)
latent0 = torch.randn(1, a["q_token_length"], a["in_channels"], generator=g)
camera0 = torch.randn(1, 1, a["cam_channels"], generator=g)
anchors = torch.rand(1, 8192, 3, generator=g)
out = {"feature1": cond["feature1"].cpu(), "feature2": cond["feature2"].cpu(), "latent0": latent0, "camera0": camera0,
       "anchors": anchors, "pixels": pixels.cpu(), "feature1_f32": feature1_f32}
noise = TensorDict({"latent": latent0.to(dev), "camera": camera0.to(dev)}, batch_size=1, device=dev)
neg = cond.apply(torch.zeros_like)

for tag, f16 in [("f16", True), ("f32", False)]:
    den = build(LatentSeqMMFlowModel, a, "denoiser", f16)
    backend, tf32 = layers.ATTN_BACKEND, torch.backends.cuda.matmul.allow_tf32
    if f16:  # the pipeline's sampling settings (optimize=True), minus torch.compile
        layers.ATTN_BACKEND, torch.backends.cuda.matmul.allow_tf32 = "sdpa_cudnn", True
    else:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    with torch.no_grad():
        v = den(noise, torch.tensor([700.0], device=dev), cond)
        out[f"fwd_latent_{tag}"], out[f"fwd_camera_{tag}"] = v["latent"].float().cpu(), v["camera"].float().cpu()
        s = sample_flow(den, noise, cond=cond, neg_cond=neg, steps=50, rescale_t=3.0, cfg_strength=3.0, verbose=False,
                        batch_cfg=True)
        out[f"sample_latent_{tag}"], out[f"sample_camera_{tag}"] = s["latent"].float().cpu(), s["camera"].float().cpu()
    if f16:
        compile_for_sampling(den)
        with torch.no_grad():
            for i in range(2):  # the first compiles
                torch.cuda.synchronize()
                t0 = time.time()
                sample_flow(den, noise, cond=cond, neg_cond=neg, steps=50, rescale_t=3.0, cfg_strength=3.0,
                            verbose=False, batch_cfg=True)
                torch.cuda.synchronize()
                print(f"sdk sample (torch.compile, batched CFG, cuDNN attention) run {i}: {time.time() - t0:.1f} s", flush=True)
    layers.ATTN_BACKEND, torch.backends.cuda.matmul.allow_tf32 = backend, tf32
    del den
    torch.cuda.empty_cache()
    gs = build(GaussianFixedlenDecoder, cfg["decoder_gs"]["args"], "decoder_gs", f16)
    with torch.no_grad():
        out[f"gs_{tag}"] = gs(anchors.to(dev), out["sample_latent_f16"].to(dev)).float().cpu()
    del gs
    torch.cuda.empty_cache()
    print(tag, "done", flush=True)

for k in ["fwd_latent", "fwd_camera", "sample_latent", "sample_camera", "gs"]:
    d = (out[f"{k}_f16"] - out[f"{k}_f32"]).abs()
    print(f"sdk f16 vs f32 {k}: max {d.max():.4g} mean {d.mean():.4g} (|ref| mean {out[k + '_f32'].abs().mean():.4g})")
d = (out["feature1"] - out["feature1_f32"]).abs()
print(f"sdk dinov3 bf16 vs f32 feature1: max {d.max():.4g} mean {d.mean():.4g}")
save_file({k: v.contiguous() for k, v in out.items()}, ref_path)
print("wrote", ref_path)
