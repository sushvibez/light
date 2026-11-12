#!/usr/bin/env python3
"""diffusers reference for Light's SDXL base port (models/sdxl*.go).

Writes an npz with everything models.TestSDXLMatchesDiffusers compares:
the token ids of a prompt for both tokenizers, each text encoder's
penultimate hidden states and bigG's projected pooled output (F32, CPU), a
seeded latent, the time ids and the UNet's noise prediction at one timestep
(fp16 on the GPU — the UNet is 5 GB), plus a tiny end-to-end image at 4
DDIM steps / 256² for an eyeball check.

    .venv/bin/python scripts/verify/ref_sdxl.py --out /tmp/sdxl_ref.npz
    LIGHT_SDXL_REF=/tmp/sdxl_ref.npz LIGHT_HEAVY_TESTS=1 go test ./models -run TestSDXLMatchesDiffusers -v
"""
import argparse
import numpy as np
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

REPO = "stabilityai/stable-diffusion-xl-base-1.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a photograph of an astronaut riding a horse on the moon")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timestep", type=int, default=500)
    ap.add_argument("--size", type=int, default=512, help="image size for the UNet step (latent = size/8)")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok1 = CLIPTokenizer.from_pretrained(REPO, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(REPO, subfolder="tokenizer_2")
    te1 = CLIPTextModel.from_pretrained(REPO, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16).float().eval()
    te2 = CLIPTextModelWithProjection.from_pretrained(REPO, subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float16).float().eval()
    ids1 = tok1(a.prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    ids2 = tok2(a.prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    with torch.no_grad():
        o1 = te1(ids1, output_hidden_states=True)
        o2 = te2(ids2, output_hidden_states=True)
    h1 = o1.hidden_states[-2]  # (1, 77, 768)
    h2 = o2.hidden_states[-2]  # (1, 77, 1280)
    pooled = o2.text_embeds     # (1, 1280) projected
    ctx = torch.cat([h1, h2], dim=-1)
    print("ids1", ids1[0, :12].tolist(), "ids2", ids2[0, :12].tolist())
    print("h1", tuple(h1.shape), float(h1.abs().mean()), "h2", tuple(h2.shape), float(h2.abs().mean()), "pooled", float(pooled.abs().mean()))

    S = a.size
    time_ids = torch.tensor([[S, S, 0, 0, S, S]], dtype=torch.float32)
    latent = torch.randn(1, 4, S // 8, S // 8, generator=torch.Generator().manual_seed(1234))
    unet = UNet2DConditionModel.from_pretrained(REPO, subfolder="unet", variant="fp16", torch_dtype=torch.float16).to(dev).eval()
    with torch.no_grad():
        eps = unet(latent.to(dev, torch.float16), torch.tensor([a.timestep], device=dev), encoder_hidden_states=ctx.to(dev, torch.float16),
                   added_cond_kwargs={"text_embeds": pooled.to(dev, torch.float16), "time_ids": time_ids.to(dev, torch.float16)}).sample.float().cpu()
    print("eps", tuple(eps.shape), float(eps.abs().mean()), float(eps.std()))
    np.savez(a.out, ids1=ids1.numpy().astype(np.int64), ids2=ids2.numpy().astype(np.int64),
             h1=h1.numpy(), h2=h2.numpy(), pooled=pooled.numpy(), time_ids=time_ids.numpy(), timestep=np.array(a.timestep, dtype=np.int64),
             latent=latent.numpy(), eps=eps.numpy())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
