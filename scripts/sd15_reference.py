"""Reference SD 1.5 generation via HuggingFace Diffusers.

Run alongside `examples/37_sd15_txt2img` with the same prompt + seed +
steps, then diff the two PNGs to verify our Go implementation.

Setup:

    pip install --upgrade torch diffusers transformers safetensors accelerate pillow numpy

Usage:

    # Generate the reference image
    python scripts/sd15_reference.py --prompt "a photograph of an astronaut riding a horse" \\
        --seed 42 --steps 25 --out astro_ref.png

    # Compare two PNGs (e.g. our Go output vs the reference)
    python scripts/sd15_reference.py diff astro_go.png astro_ref.png

The reference uses the SAME schedule we hard-coded into Go:
DDIM, scaled-linear betas, β_start=0.00085, β_end=0.012,
1000 train timesteps, leading-mode inference timesteps.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from PIL import Image


def cmd_generate(args):
    import torch
    from diffusers import StableDiffusionPipeline, DDIMScheduler

    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    print(f"device: {device}  dtype: {args.dtype}  repo: {args.repo}")
    pipe = StableDiffusionPipeline.from_pretrained(args.repo, torch_dtype=torch_dtype)
    # Match our Go DDIM scheduler exactly so the numbers stay comparable.
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        beta_end=0.012,
        num_train_timesteps=1000,
        timestep_spacing="leading",
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    out = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative or None,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        height=args.height,
        width=args.width,
        generator=generator,
        output_type="np",  # (H, W, 3) float32 in [0, 1]
    )
    img = (out.images[0] * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(args.out)
    print(f"wrote {args.out}  shape={img.shape}")


def cmd_diff(args):
    a = np.asarray(Image.open(args.a)).astype(np.float32) / 255.0
    b = np.asarray(Image.open(args.b)).astype(np.float32) / 255.0
    if a.shape != b.shape:
        print(f"shape mismatch: {a.shape} vs {b.shape}", file=sys.stderr)
        sys.exit(2)
    mse = float(((a - b) ** 2).mean())
    mae = float(np.abs(a - b).mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
    print(f"shape:  {a.shape}")
    print(f"mse:    {mse:.6f}")
    print(f"mae:    {mae:.6f}  (~{mae*255:.2f}/255 per channel)")
    print(f"psnr:   {psnr:.2f} dB")
    diff = np.abs(a - b)
    if diff.max() > 0:
        scaled = (diff / diff.max() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(scaled).save("diff.png")
        print("wrote diff.png  (per-channel |Δ|, normalised to 0..255)")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=False)

    g = sub.add_parser("generate", help="run reference SD 1.5 (default if no subcmd)")
    g.add_argument("--repo", default="runwayml/stable-diffusion-v1-5")
    g.add_argument("--prompt", default="a photograph of an astronaut riding a horse")
    g.add_argument("--negative", default="")
    g.add_argument("--steps", type=int, default=25)
    g.add_argument("--cfg", type=float, default=7.5)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--width", type=int, default=512)
    g.add_argument("--height", type=int, default=512)
    g.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    g.add_argument("--cpu", action="store_true")
    g.add_argument("--out", default="astro_ref.png")

    d = sub.add_parser("diff", help="compare two PNGs (MSE / MAE / PSNR + diff.png)")
    d.add_argument("a")
    d.add_argument("b")

    # Mirror generate args at top level so `python sd15_reference.py --prompt ...`
    # works without typing `generate` first.
    p.add_argument("--repo", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--prompt", default="a photograph of an astronaut riding a horse")
    p.add_argument("--negative", default="")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", default="astro_ref.png")

    args = p.parse_args()
    if args.cmd == "diff":
        cmd_diff(args)
    else:
        cmd_generate(args)


if __name__ == "__main__":
    main()
