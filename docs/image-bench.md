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

## Same precision: Light Mixed16 against diffusers (Colab A100, 40 GB)

Measured 2026-11-23 (`gpu_batch.sh imagebench`, Light at 8040107, before
the F32 attention change below). 1024², cold start, loads included.
"Light exact" computes in F32; "Light Mixed16" computes the transformer
in BF16 on tensor cores, the precision diffusers runs at, so the last two
columns are the like-for-like pair.

| Model | Transformer | Steps | Light exact | per step | Light Mixed16 | per step | diffusers BF16 | per step |
|---|---|---|---|---|---|---|---|---|
| Z-Image-Turbo | BF16 | 8 | (always Mixed16) | | 1m38s | | 2m32s | 0.4 s |
| FLUX.2 klein 4B | BF16 | 4 | 1m00s | | 30s | | 38s | 0.2 s |
| FLUX.2 klein 4B | FP8 | 4 | 28s | | 17s | | script failed | |
| FLUX.1 schnell | FP8 | 4 | 1m33s | 6.7 s | 45s | 1.1 s | script failed | |
| FLUX.1 schnell | Q4_K_S GGUF | 4 | 59s | 6.7 s | 58s | 6.7 s | script failed | |
| FLUX.1 dev | FP8 | 28 | 3m46s | 6.7 s | 1m11s | 1.1 s | script failed | |
| FLUX.1 dev | Q4_K_S GGUF | 28 | 3m38s | 6.7 s | 3m37s | 6.7 s | script failed | |
| FLUX.1 schnell | BF16 streamed | 4 | 1m58s | 11.9 s | 1m37s | 6.2 s | | |

- Mixed16 is worth 6x per step on FLUX.1 FP8 (6.7 s to 1.1 s): dev at
  28 steps in 1m11s from a cold start.
- Whole generations at the same precision: Light is ahead of diffusers on
  klein (30 s against 38 s) and Z-Image (1m38s against 2m32s), on loading;
  Light's per-step time for those two pipelines was not reported in this
  run (no step callback then). `Flux2Options.OnStep` and
  `ZImageOptions.OnStep` exist now and the benchmark uses them; the
  1024² figures need a rerun on Colab, the 512² ones are in the
  verification table below.
- The GGUF rows did not move under Mixed16 in this run: quantized
  weights dequantized to F32 and ran the F32 GEMM (6.7 s a step against
  1.1 s for FP8). Fixed since (`ops.MatMulQMixed`: dequantize, round to
  BF16, tensor-core GEMM with F32 accumulation): the quantized matmul at
  FLUX's shape goes from 6.7 ms to 2.2 ms, and schnell Q4_K_S on the RTX
  5060 Ti at 512² from 2.6 s to 1.1 s a step with the same image
  difference (0.034 against 0.033); the 1024² row follows with the next
  Colab run.
- diffusers' FP8 and GGUF rows still fail in `bench_diffusers.py` (the
  layerwise cast does not reach the offloaded transformer; the GGUF
  single-file load errors), so those cells are open, not a diffusers
  limit.

## Single ops: Light, PyTorch and Candle

RTX 5060 Ti, FLUX.1's 1024² geometry (4096 image + 256 text tokens, width
3072, 24 heads of 128). Each number is the median of 21 individually
synchronized calls, best of three runs, for Light and PyTorch 2.11 (TF32
off: F32 rows are true F32); Candle 4bc89dd (2026-04-16, `--features
cuda`) is the mean of 10 calls. "Light vs" is Light's time relative to
the other (negative: Light is faster); **bold** marks Light more than
20 % slower, the threshold for digging in.

    LIGHT_OPBENCH=1 go test ./ops -run TestOpBenchCUDA -v
    .venv/bin/python scripts/verify/opbench_torch.py
    CANDLE_DIR=~/src/candle bash scripts/verify/opbench_candle/run.sh

| Op | Light | PyTorch | Light vs PyTorch | Candle | Light vs Candle |
|---|---|---|---|---|---|
| GEMM F32 (4096 x 3072) · (3072 x 3072)ᵀ | 5.92 ms | 6.49 ms | -9% | 6.52 ms | -9% |
| GEMM BF16 in, F32 out (mixed) (4096 x 3072) · (3072 x 3072)ᵀ | 1.96 ms | 2.00 ms | -2% | 2.17 ms | -10% |
| GEMM BF16 in, BF16 out (4096 x 3072) · (3072 x 3072)ᵀ | 1.65 ms | 1.65 ms | +0% | 1.67 ms | -1% |
| GEMM F32 x, BF16 weight exact (W16) (4096 x 3072) · (3072 x 3072)ᵀ | 6.85 ms | — | | — | |
| FP8 weight → BF16 + GEMM (4096 x 3072) · (3072 x 3072)ᵀ | 2.06 ms | 1.75 ms | +18% | — | |
| GGUF Q8_0 weight, dequant + GEMM (4096 x 3072) · (3072 x 3072)ᵀ | 6.72 ms | — | | — | |
| GGUF Q4_0 weight, dequant + GEMM (4096 x 3072) · (3072 x 3072)ᵀ | 6.73 ms | — | | — | |
| GEMM F32 (4096 x 3072) · (12288 x 3072)ᵀ | 20.79 ms | 25.50 ms | -18% | 25.82 ms | -19% |
| GEMM BF16 in, F32 out (mixed) (4096 x 3072) · (12288 x 3072)ᵀ | 6.71 ms | 7.47 ms | -10% | 7.96 ms | -16% |
| GEMM BF16 in, BF16 out (4096 x 3072) · (12288 x 3072)ᵀ | 6.57 ms | 6.51 ms | +1% | 6.60 ms | -0% |
| GEMM F32 x, BF16 weight exact (W16) (4096 x 3072) · (12288 x 3072)ᵀ | 26.97 ms | — | | — | |
| FP8 weight → BF16 + GEMM (4096 x 3072) · (12288 x 3072)ᵀ | 7.24 ms | 6.95 ms | +4% | — | |
| Attention F32 (1, 24, 4352, 128) | 33.26 ms | 34.42 ms | -3% | 55.35 ms | -40% |
| Attention BF16 (1, 24, 4352, 128) | 5.38 ms | 5.57 ms | -3% | 25.41 ms | -79% |
| LayerNorm F32 (1, 4352, 3072) | 0.29 ms | 0.29 ms | +0% | 0.40 ms | -28% |
| RMSNorm F32 (1, 4352, 24, 128) | 0.28 ms | 0.34 ms | -18% | 0.30 ms | -7% |
| GELU(tanh) F32 (1, 4352, 12288) | 1.13 ms | 1.13 ms | +0% | 1.36 ms | -17% |
| SiLU F32 (1, 4352, 3072) | 0.28 ms | 0.29 ms | -3% | 0.35 ms | -20% |
| Add F32 (1, 4352, 3072) | 0.28 ms | 0.29 ms | -3% | 0.34 ms | -18% |
| Softmax F32 (24, 1024, 1024) | 0.56 ms | 0.53 ms | +6% | 6.48 ms | -91% |
| Conv2d 3x3 F32 (1, 128, 512, 512) → 128 | 11.26 ms | 11.01 ms | +2% | 25.74 ms | -56% |
| Conv2d 3x3 BF16 (1, 128, 512, 512) → 128 | 2.56 ms | 2.40 ms | +7% | 17.55 ms | -85% |

Reading it:

- The GEMMs, where a diffusion step spends most of its time, are at
  parity or ahead: F32 9 to 19 % faster than both (cuBLAS with Light's
  layout), BF16 tensor-core GEMMs within 2 %.
- F32 attention was the outlier at first measurement (82.6 ms: the
  streaming flash kernel at 2.8 TFLOPS, 2.3x slower than PyTorch). It now
  runs as two cuBLAS GEMMs per head around an in-place exp, with the
  softmax division moved to the small output and queries in chunks of at
  most 512 MB of scores (`ops.AttentionGEMM`): 2.5x faster, ahead of
  PyTorch and Candle, and never slower than the flash kernel down to 64
  tokens. One GEMM per head rather than cuBLAS's strided-batched call is
  worth 1.5x at this size.
- RMSNorm over 128-wide head rows was +39 to +45 %: a 256-thread block
  per row left half the threads idle and paid two barriers a row. Rows up
  to 512 wide now take a warp each, eight rows a block, shuffles only:
  0.46 ms to 0.29-0.35 ms, level with PyTorch.
- The swings first seen on Add, GELU and Softmax (Add at 0.28, 0.47 and
  0.76 ms) were the allocator, not the kernels: an output returns to the
  caching pool only at a garbage collection, so a cold pool misses, and a
  miss is a cudaMalloc (about 80 µs for a 53 MB block, a quarter of such
  an op), plus a forced collection (0.4 to 0.9 ms) for every GB of fresh
  memory. With a warm pool the same calls show no mallocs and match
  PyTorch to 3 % (`TestAllocJitterCUDA`; `storage.CUDAPoolStats` now
  counts mallocs and their time). In a pipeline the pool is warm after
  the first step; the benchmark warms it before timing.
- LayerNorm and SiLU read +24 % and +29 % in the first table, and that
  was the benchmark again, not the kernels: the timed calls ran without
  a collection between them, so depending on what earlier rows had left
  in the pool, more than half of the 21 calls could miss it and pay a
  cudaMalloc, which lands on the median. The benchmark now collects
  outside the timed calls every third one and reports any mallocs inside
  them; both ops then measure 0.28-0.29 ms on every run, level with
  PyTorch, with no kernel change.
- BF16 convolution was +29 % (3.06 ms against 2.38 ms) for two host-side
  reasons, neither the algorithm (cuDNN's heuristic picks the same
  implicit-precomputed GEMM PyTorch gets; direct and plain implicit GEMM
  are 3x slower here). The 128 MB workspace was a cudaMalloc and a
  cudaFree per call, and cudaFree waits for the device: it now comes
  from the caching pool and goes straight back (0.3 ms). And the four
  descriptors, the algorithm query and the workspace-size query ran on
  every call, 0.28 ms before the device started (0.62 ms for F32): they
  are now cached per shape (0.2 ms). 2.56 ms, 7 % behind PyTorch; F32
  11.26 ms, 2 % behind. Every VAE convolution gets the same saving.
- BF16 attention was 17 % behind PyTorch (6.50 ms, 35.8 TFLOPS). The
  flash kernel at head dim 128 ran 8 warps of 16 query rows over
  single-buffered 64-key tiles. It now runs 4 warps of 32 rows, so each
  key / value fragment read from shared memory feeds two tensor-core
  products, over double-buffered 32-key tiles, the next tile loading
  under the current one's products: 5.38 ms, 43.3 TFLOPS, 3 % ahead of
  PyTorch and 4.7x ahead of Candle, which has no flash attention in its
  FLUX path. 0.63 to 0.52 ms at 1280 tokens. Either change alone was
  worth 1 to 4 %.
- Nothing in the table is now more than 7 % behind PyTorch.
- Candle's softmax (12x slower) and convolutions (2.3x F32, 5.6x BF16)
  are where it loses most against both.

## Verification against diffusers, 512²

Same seeded noise, diffusers in BF16 with CPU offload, Light as above;
"Δ" is the mean absolute pixel difference on the [0, 1] scale (the bar is
0.03 for four 16-bit steps; it compounds with steps and with 4-bit
weights; every image shows the same scene, the Q4 dev run redraws the rocks).

| Model | Transformer | Fits how | Per step | Loads + steps | Δ | Steps |
|---|---|---|---|---|---|---|
| Z-Image-Turbo 6B | BF16 | resident | 0.50 s | 53 s | 0.026 | 4 |
| FLUX.2 klein 4B | BF16 | resident | 1.31 s | 57 s | 0.025 | 4 |
| FLUX.2 klein 4B | FP8 folder, 3.7 GB | resident | 1.27 s | 37 s | 0.029 | 4 |
| FLUX.1 schnell 12B | BF16, 23.8 GB | streamed | 25 s | 3.3 min | 0.018 | 4 |
| FLUX.1 schnell 12B | FP8, 11.9 GB | resident | 3.1 s | 44 s | 0.024 | 4 |
| FLUX.1 schnell 12B | Q4_K_S GGUF, 6.8 GB | resident | 2.6 s | 39 s | 0.033 | 4 |
| FLUX.1 dev 12B | BF16, 23.8 GB | streamed | 33 s | 4.8 min | 0.036 | 8 |
| FLUX.1 dev 12B | FP8, 11.9 GB | resident | 3.6 s | 61 s | 0.043 | 8 |
| FLUX.1 dev 12B | Q4_K_S GGUF, 6.8 GB | resident | 3.2 s | 55 s | 0.066 | 8 |
| FLUX.2 dev 32B | BF16, 64 GB | Colab A100 40 GB: Mistral-3 24B encoder on the CPU, blocks streamed | ~80 s | 12.1 min | 0.010 | 8 |
| FLUX.2 dev 32B | FP8 folder, 30.7 GB | Colab A100 40 GB: encoder on the CPU, resident | ~35 s | 6.4 min | 0.018 | 8 |

The Z-Image and klein per-step figures are measured through the step
callback, each step ended by a device synchronisation, mean over the
steps after the first (they were estimates from the total before, which
counted the loads: about 7, 9 and 5 s). With Mixed16, klein is 0.44 s a
step from the BF16 repo and 0.49 s from the FP8 folder, against 1.31 s
and 1.27 s at F32 compute. Z-Image has no Mixed16 option: its 0.50 s is
its only path. The totals are dominated by loading and move with the
page cache (klein FP8 measured 37 s to 96 s across runs).

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
