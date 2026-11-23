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
so the card's tensor cores sit idle. The Mixed16 option (BF16 GEMMs with
F32 accumulation and BF16 flash attention, diffusers' own precision; the
engine's default on CUDA since 2026-11-21) changes that: schnell FP8 at
512² goes from 3.1 s to 1.0 s a step and klein FP8 keeps its image
difference (0.0295 vs 0.029), with 1024² rows measured on Colab below.

## Light, diffusers and Candle on one card (Colab L4, 24 GB)

The same prompt, 1024², each model's own step count, cold start with loads
included, one process per row, measured 2026-11-21 by
`scripts/colab/gpu_batch.sh imagebench candlebench`:

- Light: F32 compute (Mixed16 rows pending from the rerun), sequential loads.
- diffusers 0.37 / torch 2.11: BF16 tensor-core compute, model CPU
  offload; the FP8 and GGUF rows use the same files as Light (layerwise
  casting; GGUF dequantized per matmul). Their first attempt failed on the
  benchmark script (cast order, missing `gguf` package); fixed, rerun pending.
- Candle (upstream 1a49dc2, 2026-11-19, `--features cuda`): its FLUX
  example, which covers schnell with its own quantized GGUF
  (`lmz/candle-flux`, Q8) and BF16 only on a 40 GB card; second run of
  two counted (downloads and kernels warm).

| Model | Transformer | Steps | Light | Light per step | diffusers | diffusers per step | Candle |
|---|---|---|---|---|---|---|---|
| Z-Image-Turbo | BF16 | 8 | 1m49s | | 2m30s | 1.5 s | not ported |
| FLUX.2 klein 4B | BF16 | 4 | 1m13s | | 0m59s | 0.9 s | not ported |
| FLUX.2 klein 4B | FP8 | 4 | 41s | | pending | | not ported |
| FLUX.1 schnell | BF16 streamed | 4 | 3m20s | 37.8 s | | | needs 40 GB |
| FLUX.1 schnell | FP8 | 4 | 2m00s | 13.1 s | pending | | |
| FLUX.1 schnell | Q4_K_S GGUF (Light) / Q8 GGUF (Candle) | 4 | 1m53s | 13.4 s | pending | | 2m04s |
| FLUX.1 dev | FP8 | 28 | 6m45s | 13.0 s | pending | | no quantized dev |
| FLUX.1 dev | Q4_K_S GGUF | 28 | 6m44s | 13.3 s | pending | | |

What the rows say so far:

- Against Candle, Light's quantized schnell is ahead on the whole
  generation (1m53s vs 2m04s) with a smaller file (Q4_K_S, 6.8 GB vs
  Q8, 12 GB) and F32 compute; Candle's example also runs F32-accumulated
  BF16 kernels, so the per-step gap is in loading and the first step.
  Candle has no dev quantization, no FP8 path, and neither Z-Image nor
  FLUX.2.
- Against diffusers, Light loads faster (Z-Image: 1m49s vs 2m30s for
  eight steps) but computes slower per step (klein: 0m59s vs 1m13s) while
  it runs exact F32 matmuls; the Mixed16 path closes that: on the RTX
  5060 Ti at 512² schnell FP8 goes from 3.1 s to 1.0 s a step, and the
  1024² Mixed16 rows are in the pending rerun.
- Light's totals are dominated by loading: the FP8 schnell spends ~1m10s
  of 2m00s before the first step (the 12 GB file from the VM disk).

`scripts/verify/bench_diffusers.py` reproduces the diffusers rows.

## Single ops: Light, PyTorch and Candle

RTX 5060 Ti, FLUX.1's 1024² geometry (4096 image + 256 text tokens, width
3072, 24 heads of 128), mean of 10 calls after 3 warm-ups, device
synchronized; PyTorch 2.11 with TF32 off (F32 rows are true F32), Candle
4bc89dd (2026-04-16) with `--features cuda`. "Light vs" is Light's time
relative to the other (negative: Light is faster); **bold** marks Light
more than 20 % slower, the threshold for digging in. Sub-millisecond rows
move by 10 to 20 % between runs.

    LIGHT_OPBENCH=1 go test ./ops -run TestOpBenchCUDA -v
    .venv/bin/python scripts/verify/opbench_torch.py
    CANDLE_DIR=~/src/candle bash scripts/verify/opbench_candle/run.sh

| Op | Light | rate | PyTorch | Light vs PyTorch | Candle | Light vs Candle |
|---|---|---|---|---|---|---|
| GEMM F32 (4096 x 3072) · (3072 x 3072)ᵀ | 5.70 ms | 13.6 TFLOPS | 6.47 ms | -12% | 6.52 ms | -13% |
| GEMM BF16 in, F32 out (mixed) (4096 x 3072) · (3072 x 3072)ᵀ | 1.94 ms | 39.8 TFLOPS | 2.06 ms | -6% | 2.17 ms | -11% |
| GEMM BF16 in, BF16 out (4096 x 3072) · (3072 x 3072)ᵀ | 1.71 ms | 45.3 TFLOPS | 1.66 ms | +3% | 1.67 ms | +2% |
| GEMM F32 x, BF16 weight exact (W16) (4096 x 3072) · (3072 x 3072)ᵀ | 6.74 ms | 11.5 TFLOPS | — | | — | |
| FP8 weight → BF16 + GEMM (4096 x 3072) · (3072 x 3072)ᵀ | 2.04 ms | 37.9 TFLOPS | 1.74 ms | +17% | — | |
| GGUF Q8_0 weight, dequant + GEMM (4096 x 3072) · (3072 x 3072)ᵀ | 6.13 ms | 12.6 TFLOPS | — | | — | |
| GGUF Q4_0 weight, dequant + GEMM (4096 x 3072) · (3072 x 3072)ᵀ | 6.64 ms | 11.6 TFLOPS | — | | — | |
| GEMM F32 (4096 x 3072) · (12288 x 3072)ᵀ | 20.85 ms | 14.8 TFLOPS | 25.62 ms | -19% | 25.82 ms | -19% |
| GEMM BF16 in, F32 out (mixed) (4096 x 3072) · (12288 x 3072)ᵀ | 6.74 ms | 45.9 TFLOPS | 7.54 ms | -11% | 7.96 ms | -15% |
| GEMM BF16 in, BF16 out (4096 x 3072) · (12288 x 3072)ᵀ | 6.54 ms | 47.3 TFLOPS | 6.47 ms | +1% | 6.60 ms | -1% |
| GEMM F32 x, BF16 weight exact (W16) (4096 x 3072) · (12288 x 3072)ᵀ | 27.37 ms | 11.3 TFLOPS | — | | — | |
| FP8 weight → BF16 + GEMM (4096 x 3072) · (12288 x 3072)ᵀ | 7.39 ms | 41.8 TFLOPS | 7.00 ms | +6% | — | |
| Attention F32 (1, 24, 4352, 128) | 82.60 ms | 2.8 TFLOPS | 35.84 ms | **+130%** | 55.35 ms | **+49%** |
| Attention BF16 (1, 24, 4352, 128) | 6.47 ms | 36.0 TFLOPS | 5.71 ms | +13% | 25.41 ms | -75% |
| LayerNorm F32 (1, 4352, 3072) | 0.28 ms | 388 GB/s | 0.29 ms | -3% | 0.40 ms | -30% |
| RMSNorm F32 (1, 4352, 24, 128) | 0.46 ms | 234 GB/s | 0.33 ms | **+39%** | 0.30 ms | **+53%** |
| GELU(tanh) F32 (1, 4352, 12288) | 1.41 ms | 303 GB/s | 1.13 ms | **+25%** | 1.36 ms | +4% |
| SiLU F32 (1, 4352, 3072) | 0.69 ms | 155 GB/s | 0.29 ms | **+138%** | 0.35 ms | **+97%** |
| Add F32 (1, 4352, 3072) | 0.28 ms | 575 GB/s | 0.28 ms | +0% | 0.34 ms | -18% |
| Softmax F32 (24, 1024, 1024) | 0.53 ms | 379 GB/s | 0.53 ms | +0% | 6.48 ms | -92% |
| Conv2d 3x3 F32 (1, 128, 512, 512) → 128 | 11.36 ms | 6.8 TFLOPS | 11.07 ms | +3% | 25.74 ms | -56% |
| Conv2d 3x3 BF16 (1, 128, 512, 512) → 128 | 3.15 ms | 24.5 TFLOPS | 2.36 ms | **+33%** | 17.55 ms | -82% |

Reading it:

- The GEMMs, where a diffusion step spends most of its time, are at
  parity or ahead: F32 12 to 19 % faster than both (cuBLAS with Light's
  layout), BF16 tensor-core GEMMs within 3 %.
- **F32 attention is the outlier**: Light's F32 flash kernel runs at 2.8
  TFLOPS, 2.3x slower than PyTorch's and 1.5x slower than Candle's naive
  matmul-softmax-matmul. It is what made the exact-F32 pipeline rows slow
  at 1024² (two attentions per block). The BF16 flash kernel is fine (36
  TFLOPS, 13 % behind PyTorch, 4x ahead of Candle, which has no flash
  attention in its FLUX path).
- To dig into (over 20 %): F32 attention, RMSNorm (+39 %), BF16 conv
  (+33 %), GELU (+25 %); SiLU swung between runs and needs a longer
  measurement before it counts.
- Candle's softmax (12x slower) and convolutions (2.3x F32, 5.6x BF16)
  are where it loses most against both.

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
