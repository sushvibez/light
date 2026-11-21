# Image models on one 16 GB card

Measured on the desktop RTX 5060 Ti (16 GB, PCIe 5 x8, NVMe), CUDA 13.2,
Light on `desktop/cuda`. Every pipeline loads sequentially (text encoders,
then the transformer, then the VAE) with 16-bit weight storage and F32
compute; the transformer sits on the card when it fits ("resident") and is
otherwise streamed block by block from the memory-mapped checkpoint
("streamed": the whole file is read once per step).

Reproduce: `LIGHT_HEAVY_TESTS=1 LIGHT_IMAGE_BENCH=all go test ./diffusion -run TestImageBenchCUDA -v -timeout 3h`
(the FP8 and GGUF transformers live in `~/.cache/light/models/checkpoints`,
made with `scripts/verify/make_flux_fp8_real.py`, `make_flux2_fp8_real.py`
and city96's GGUFs).

## Full generations, 1024², each model's own step count

Wall time from a cold start to the PNG, loads included.

Light, RTX 5060 Ti (16 GB), F32 compute:

| Model | Transformer | Size | Steps | Per step | Full generation |
|---|---|---|---|---|---|
| Z-Image-Turbo | BF16 | 1024² | 8 | | 1m11s |
| FLUX.2-klein-4B | BF16 | 1024² | 4 | | 1m16s |
| FLUX.2-klein-4B | FP8 | 1024² | 4 | | 41s |
| FLUX.1-schnell | BF16 streamed | 1024² | 4 | 22.0 s | 2m21s |
| FLUX.1-schnell | FP8 | 1024² | 4 | 14.3 s | 1m40s |
| FLUX.1-schnell | Q4_K_S GGUF | 1024² | 4 | 13.2 s | 1m34s |
| FLUX.1-dev | FP8 | 1024² | 28 | 14.1 s | 7m0s |
| FLUX.1-dev | Q4_K_S GGUF | 1024² | 28 | 13.1 s | 6m40s |

Per step is the mean over the steps after the first (the first includes
the transformer load). At 1024² the step is 4x the tokens of 512² and
4.5x the time: the matmuls run in F32 after widening the stored weights,
so the card's tensor cores sit idle; a BF16-compute option (Mixed16, as
Z-Image uses on the L4) is the next speed step.

Same models with diffusers (PyTorch, BF16 tensor-core compute, model CPU
offload, FP8 via layerwise casting, GGUF dequantized per matmul) and with
Candle (its quantized schnell example): measured on a Colab card next to
Light, tables in this file once the run lands (`scripts/colab/gpu_batch.sh
imagebench`, `scripts/verify/bench_diffusers.py`).

## Verification against diffusers, 512²

Same seeded noise, diffusers in BF16 with CPU offload, Light as above;
"Δ" is the mean absolute pixel difference on the [0, 1] scale (the bar is
0.03 for four 16-bit steps; it compounds with steps and with 4-bit
weights; every image shows the same scene, the Q4 dev run redraws the rocks).

| Model | Transformer | Fits how | Per step | Loads + steps | Δ | Steps |
|---|---|---|---|---|---|---|
| Z-Image-Turbo 6B | BF16 | resident | ~7 s | 53 s | 0.026 | 4 |
| FLUX.2 klein 4B | BF16 | resident | ~9 s | 57 s | 0.025 | 4 |
| FLUX.2 klein 4B | FP8 folder, 3.7 GB | resident | ~5 s | 37 s | 0.029 | 4 |
| FLUX.1 schnell 12B | BF16, 23.8 GB | streamed | 25 s | 3.3 min | 0.018 | 4 |
| FLUX.1 schnell 12B | FP8, 11.9 GB | resident | 3.1 s | 44 s | 0.024 | 4 |
| FLUX.1 schnell 12B | Q4_K_S GGUF, 6.8 GB | resident | 2.6 s | 39 s | 0.033 | 4 |
| FLUX.1 dev 12B | BF16, 23.8 GB | streamed | 33 s | 4.8 min | 0.036 | 8 |
| FLUX.1 dev 12B | FP8, 11.9 GB | resident | 3.6 s | 61 s | 0.043 | 8 |
| FLUX.1 dev 12B | Q4_K_S GGUF, 6.8 GB | resident | 3.2 s | 55 s | 0.066 | 8 |
| FLUX.2 dev 32B | BF16, 64 GB | Colab A100 40 GB: Mistral-3 24B encoder on the CPU, blocks streamed | ~80 s | 12.1 min | 0.010 | 8 |
| FLUX.2 dev 32B | FP8 folder, 30.7 GB | Colab A100 40 GB: encoder on the CPU, resident | ~35 s | 6.4 min | 0.018 | 8 |

Tiny-verified only (architecture against diffusers on random weights, no
real-weight run on this card yet): FLUX 3 image klein, Qwen-Image 2.1,
SD 1.5, SDXL base, SD3.

## Notes

- FP8 here is weight-only (float8_e4m3fn storage, widened per matmul), as
  the community `*-fp8` checkpoints and ComfyUI do; no activation
  quantisation, so the difference from BF16 is small (0.005 at four steps).
- The GGUF path keeps ggml-quantized weights on the card and dequantizes
  inside the matmul; Q4_K_S halves the FP8 size again at ~4-bit quality.
- Streamed BF16 is exact but bound by the disk and PCIe: 24 GB per step.
