# Light — Roadmap

Open work only, in priority order. What already works is in
[`STATUS.md`](STATUS.md); finished items move there (with the evidence in
the commit message) when they meet the validation bar:
top-5 next-token logits vs the reference on a ≥15-token chat prompt with real
weights, greedy agreement, CPU = CUDA.

The inference-first roadmap of 2026-10-27 (items 1–17: SD 1.5, Llama-3,
Phi-3, sliding windows, EmbeddingGemma 2, Gemma 2/3, MoE, Muse Glimmer,
Gated DeltaNet / Qwen3.5, DeepSeek-V2/V3, GLM-4, Nemotron, MiMo, MiniMax,
GPTQ / AWQ) is complete as of 2026-11-08 except for the items below.

---

## 1. Remaining inference roadmap

1. **SDXL base** — same primitives as SD 1.5 (CLIP, UNet2DCondition, VAE,
   DDIM), with dual text encoders and a bigger UNet.
2. **Flux on real weights / SD3** — the Flux pipeline (`diffusion.LoadFlux`) is
   verified end to end on a tiny checkpoint; FLUX.1-schnell (12B) still needs
   a real run (Colab, or w16 on a 32 GB host) and a CLI entry point. SD3
   (MMDiT with CLIP-L + CLIP-G + T5) reuses most of it.
3. **INT8 W8A8 GEMM** — `cublasGemmEx(CUBLAS_COMPUTE_32I)` for
   SmoothQuant-style int8 weights × int8 activations; today int8 is
   per-column weights × F32 activations only (`ops.MatMulInt8`).

## 2. Model coverage gaps

### Vision towers and multimodal input

- **Gemma 3 SigLIP** tower (image input for gemma-3-4b and up; the text path
  skips it today).
- Qwen3.5 images work end to end (2026-11-09); next: images in the
  interactive shell, video frames, Qwen3-VL deepstack (Qwen3-VL proper,
  WeVisOne-VL, the Qwen-Image text encoder).
- Gemma 4 vision / audio towers; Muse Glimmer's 1.8B ViT; EmbeddingGemma 2
  vision / audio towers (text encoders only today).
- CLIP vision tower (SD only has the text tower), SigLIP, DINOv2, Segment
  Anything, Depth Anything v2.

### Large-MoE validation

- **Mixtral** — HF `mixtral` model_type is not routed yet (`block_sparse_moe.*`
  names onto `nn.SparseMoE`); the GGUF MoE metadata path for mixtral
  converters exists but has not been run on a real file.
- **Qwen3.6-35B-A3B** (and Qwen3-30B-A3B: Q4_K_M ≈18 GB is above a 16 GB
  card; run on the A100, or Q3/Q2 / CPU offload).
- Gemma 4 26B-A4B MoE blocks (not implemented).

### Other open model items

- **Pending GPU runs** (code done and CPU-verified; waiting for the Colab
  batch): GPTQ act-order (TinyLlama) on CUDA; DeepSeek V2 group-limited and
  V3 sigmoid routing on CUDA; MiniMax chunked lightning attention on CUDA;
  MiMo MTP speculative decoding speed and acceptance rate
  (`TestGenerateMiMoSpeculativeBF16`); a full engine generation with the
  Muse Glimmer 30B GGUF; 4-bit MoE experts on CUDA.
- **Qwen3 dense**: standalone checkpoint (Qwen3-0.6B / 1.7B / 4B) logit check.
- **GLM `glm`** (THUDM/glm-4-9b-chat-hf) on real weights (glm4 is verified,
  `glm` only on the tiny model).
- **Qwen3.8-27B** checkpoint (same architecture as Qwen3.5) and the MTP head
  of Qwen3.5 / 3.8 (`mtp.*` is skipped today) for speculative decoding.
- **DeepSeek-V3** MTP layer (not loaded); full-size V2 / V3 are out of reach
  for a real-weight run.
- Muse Glimmer DFlash block-diffusion drafter.
- Phi-4 (14B): needs a 4-bit path (GPTQ / AWQ or a phi3 GGUF loader).
- GPT-2 large / xl: run once on real weights.
- Nemotron-Mini-4B-Instruct / Minitron-4B ship only `pytorch_model.bin`: wire
  `light/pth` into the Nemotron loader.
- Bigger sizes on bigger GPUs: Gemma 3 12B / 27B, Gemma 4 E4B / 12B / 31B
  (quantized path), Whisper beyond tiny.
- Full-sequence diffusion LMs (LLaDA-8B, Dream-7B): SDAR's mask machinery with
  block = whole sequence, plus a quantized path to fit 16 GB.
- **TTS**: Kokoro-82M (StyleTTS2 decoder, needs a phonemizer) or Parler-TTS
  (Candle has it); EnCodec / SNAC codecs first, since both emit codec tokens.
- Encoders: nomic-bert / modernbert (next on the encoder path), RoBERTa (BERT
  with byte-level BPE + position offset), T5 / Flan-T5 (relative position
  bias), Nemotron-3.5-Embed-1B (after a Ministral-3 backbone).
- Other Candle modules not ported: phi (1 / 1.5 / 2), gemma v1, starcoder2,
  mpt, olmo / olmo2, stable_lm, lfm2, SmolLM3, chatglm / codegeex4,
  mamba / mamba2 / rwkv / recurrent_gemma (SSM scans), distilbert, jina_bert,
  xlm_roberta, marian, blip / llava / moondream / paligemma / pixtral,
  wuerstchen.
- **OpenMythos compatibility** (Phase 11, added 2026-04-20): run
  [OpenMythos](https://github.com/kyegomez/OpenMythos) models on this backend.
- Torchvision catalogue (not started):
  - Classifiers: SqueezeNet (1.0, 1.1), DenseNet (121/161/169/201), GoogLeNet,
    Inception v3 (auxiliary classifiers), ResNeXt (50, 101; grouped conv ready),
    Wide ResNet (50, 101), MnasNet (0.5/1.0/1.3), MobileNet v3 (Small / Large;
    needs SE + h-swish), ShuffleNet v2 (channel shuffle), EfficientNet B0–B7
    and v2 S/M/L (SE, Fused-MBConv), RegNet X/Y, ConvNeXt; Swin Transformer
    (window attention), MaxViT.
  - Detection: YOLOv8, YOLOv11 / v12 (anchor-free, decoupled head), YOLOX
    (SimOTA); Faster R-CNN and Mask R-CNN (need RoI Align), RetinaNet (focal
    loss + FPN), SSD300 / SSDLite320, FCOS, DETR (ViT blocks ready);
    pretrained YOLOv5n weights.
  - Segmentation: FCN (backbone ready), DeepLab v3 (ASPP), LRASPP, SAM / SAM 2.
  - Video: R3D-18 / MC3-18 / R(2+1)D-18 (Conv3d forward exists), MViT v1/v2,
    S3D, Swin3D. Optical flow: RAFT.
  - Quantized torchvision models (GoogLeNet, Inception v3, MobileNet v2/v3,
    ResNet 18/50, ShuffleNet v2): need INT8 tensor ops (§1.4).
  - Missing prerequisites: Squeeze-and-Excitation block, window attention,
    RoI Align / RoI Pool.
  - Llama-2 7B (same loader as Llama-3; needs the checkpoint).

## 3. User-queued targets (2026-10-28, in the order given)

Each starts with a config / architecture inspection, then the usual
validation bar.

1. **LTX-2.5** — video DiT + video VAE.
2. **`black-forest-labs/flux-3-image-klein-comfy`** — FLUX image model,
   ComfyUI checkpoint layout (after §1.2).
3. **`XHToken/Spark-X2.6-4B`**
4. **LFM3** (Liquid AI)
5. **`HelpingAI/PixelGen`**

## 4. Backends

### WebGPU

- Opt-in load-time Q8_0 (`-dtype q8_0`) for models whose 16-bit weights
  still do not fit shared memory.
- BF16 / F16 arithmetic (16-bit weights are stored and widened inside the
  matmul since 2026-11-09; activations are F32; no F64 either).
- Load-time memory: the CPU loads the checkpoint's 16-bit weights directly,
  but Gemma-4-E2B still peaks at ~16 GB working set on the iGPU.
- MoE kernels (`moe_gemm`, routing).
- Missing kernels: `conv1d`, `gather`, `cumsum`, compare ops, `clamp`,
  `where`, `flash_attention`.
- Remaining per-token cost after dispatch batching.
- **Perf pass:** the Q8_0 GEMV decodes 35% slower than the 16-bit one on the
  Intel iGPU (granite 2.9 vs 4.4 tok/s) although it reads about half the
  bytes; `q8_0` should beat `w16` once it is vectorised.

### CUDA

- **CUDA graphs / per-token overhead** (desktop) for small models (launch overhead on
  the decode hot path); multi-stream CUDA.
- **Faster GGUF decode kernel**: `matmul_q` v1 (warp per column, byte loads)
  decodes Qwen2-0.5B at ~17 tok/s vs 31 tok/s F32 cuBLAS — vectorised block
  loads, MMVQ-style int8 dot with Q8 activations, and an MMQ tile path for
  prefill.
- WMMA tensor-core tiles for the MoE grouped GEMM.
- Paged and varlen attention.
- F64 einsum on the device (falls back to the host).

### CPU

- Remaining prefill time is spread over small ops (casts, copies,
  element-wise); Q5_K and the other GGUF formats without a CPU row decoder
  still dequantize per call; Qwen3.5's delta rule on the CPU is unoptimised.

### Other

- IQ / TQ GGUF types (Candle lacks them too).
- `Unique` op.
- Benchmark harness vs Candle / vLLM / llama.cpp, and per-op wall time +
  peak host / GPU memory (requested 2026-10-27; `light bench` covers whole
  models per host).
- Regenerate the CUDA column of the kernel matrix (STATUS.md §3) on a CUDA
  build.
- Metal backend and an ONNX importer (deferred).

## 5. Training / autograd

Deferred until inference parity (user decision 2026-10-27); these block LLM
fine-tuning.

- Autograd for RoPE, GroupNorm, FlashAttention, Conv1d / Conv3d, Cast, and
  the embedding gather on CUDA.
- `no_grad` scope; hooks; in-place safety via the version counter.
- `nn.ModuleList` / `nn.ModuleDict`.
- LoRA modules + a fine-tune-Qwen example.
- Mixed-precision training (BF16 weights, F32 master copy, loss scaling /
  grad scaler).
- DataLoader.
- Gradient clipping, Adafactor, checkpoint save / load helper (AdamW +
  cosine polish).
- Training recipes: ResNet-50 ImageNet-style (synthetic 1k-class smoke test
  first), MobileNet v2 + grouped-conv backward gradcheck, ViT-B/16 fine-tune
  on a small classification set, YOLOv5n on a small detection benchmark
  (needs box loss + assigners).

---

## Candidates: open-weight releases Jun–Oct 2026

Web survey of 2026-10-27, filtered for a single 16 GB GPU. Items marked
*[unverified]* rest on secondary sources; check the model card before
implementing. Already done from this survey: Qwen3.5 architecture (covers
Qwen3.6 / 3.8), Muse Glimmer 30B, Mellum 2.1, EmbeddingGemma 2, Qwen3-ASR.

| Model | Month | Size | What it needs from us | License | Fits 16 GB |
|-------|-------|------|-----------------------|---------|------------|
| **Qwen3.8-27B** (`Qwen/Qwen3.8-27B`, +FP8) | Aug | 27B dense (+1B ViT) | Same hybrid stack as Qwen3.5 (implemented); checkpoint validation, MRoPE + vision tower for images, MTP head | Apache-2.0 | Q4 only |
| **Jais 3** (`inception42/Jais-3-*`) | Oct | 0.9B / 3.7B / 7B / 32B, 36B-A4B MoE | custom `k2_horizon` arch, `trust_remote_code` *[arch unverified]* | Apache-2.0 | 0.9B–7B BF16 |
| **Apertus 1.5** (`swiss-ai/Apertus-v1.5-8B`) | Jul | 8B / 70B | xIELU activation, 262K ctx *[1.5 details unverified]* | Apache-2.0 (gated) | 8B BF16 |
| **Nemotron 3.5 Embed-1B / Rerank-8B** (`nvidia/Nemotron-3.5-*`) | Oct | 1B / 8B | Bidirectional Ministral-3 tower (+ Pixtral vision encoder), mean pool + MLP head | OpenMDW-1.1 | yes |
| **WeVisOne-VL 2B/4B** (`tencent/WeVisOne-VL-2B`) | Oct | 2.4B / 4.8B | Qwen3-VL base + box-guided query decoder + mask head (detection, grounding, OCR, segmentation); needs the Qwen3-VL tower | Apache-2.0 | yes |
| **Qwen-Image-2.1** (`Qwen/Qwen-Image-2.1`) | Sep | 7B DiT | 32 single-stream DiT layers, Qwen3-VL-8B text encoder, 64-ch VAE *[encoder/VAE from secondaries]* | Qwen Research (non-commercial) | FP8 |
| Krea 2 / Ideogram 4.0 | Jun | 12B / 9.3B DiT | single-stream GQA DiT, Qwen3-VL encoder, Qwen-Image VAE | restrictive | FP8 Turbo ~12 GB |
| Breeze TTS 2 (`BreezeBlue/Breeze-TTS-2`) | Aug | 3B | arch undisclosed | non-commercial | yes |

Too large for this box, for awareness: Kolibri-1 (Aleph Alpha, 78B-A3.5B),
GLM-5.3-Flash (320B-A18B), DeepSeek V4.1-Flash (552B), Kimi K3, Mistral
Large 4, Muse Spark Open (816B). No Phi-5, SmolLM4, OLMo 4, Gemma 5, new
Granite, YOLO27 weights, SAM 4 or open Whisper successor were found in the
window (Gemma 4 and Granite 4.1 were April; YOLO26 January).
