# Light — Status

What works today, as of **2026-11-09**. Open work lives in
[`ROADMAP.md`](ROADMAP.md). The evidence behind each status is in the commit
that landed it (`git log`); the work log kept until 2026-11-09 is archived in
[`docs/history/checkpoints-2026.md`](docs/history/checkpoints-2026.md).

This file merges the former `MODELS.md` (model catalogue), `FEATURES.md`
(feature checklist and kernel matrix) and `docs/PARITY-CANDLE.md`
(comparison with Candle).

**Validation bar (since the 2026-10-27 RoPE finding):** a model counts as
verified only when its top-5 next-token logits match the reference
implementation (transformers via `scripts/verify/ref_*.py`, or the upstream
code) on a ≥15-token chat-template prompt with real weights, greedy
continuations agree, and the CPU and CUDA paths agree with each other.
Codecs and encoders are checked on codes, embeddings and decoded signals.
Checkpoints too large for the 16 GB reference card run as gated heavy tests
(`LIGHT_HEAVY_TESTS=1`) on a Colab A100 80 GB / L4.

**Legend.** Model tables: `✅ Verified <date>` = meets the bar above;
`Done — pretrained` = HF / torchvision weights load and outputs match the
reference; `Done — inference + GPU` = random-init forward verified on CUDA,
parameter count matches the reference; `Training` = end-to-end training
example green; `Architecture-ready` = loader path exists, no real-weight
check yet (the check is on the roadmap). Feature lists: `[x]` complete and
tested (CPU + CUDA parity where applicable). Partial or open items are in
ROADMAP.md.

Contents: [1. Models](#1-models) · [2. Features](#2-features) ·
[3. Kernel coverage](#3-kernel-coverage) · [4. Light vs Candle](#4-light-vs-candle) ·
[5. Performance snapshots](#5-performance-snapshots)

---

## 1. Models

Each model lives in `models/` with a one-call loader and a `Forward` method;
LLMs implement `models.CausalLM` and are routed by `model_type` through
`LoadCausalLMFromHF` / `LoadChatModelFromHF` (safetensors) and
`LoadChatModelGGUF` (dispatch on `general.architecture`), and the engine
catalog lists them. Notes are one-line summaries; the numbers behind each
status are in the commits (and, up to 2026-11-09, the
[archived work log](docs/history/checkpoints-2026.md)).

### 1.1 Language models

| Model | File | Status | Notes |
|-------|------|--------|-------|
| **GPT-2 (small / medium)** | `models/gpt2.go` | Done — pretrained | 124M; greedy matches PyTorch; CPU, CUDA and WebGPU identical. `examples/25_gpt2_pretrained` |
| GPT-2 large / xl | `models/gpt2.go` | Architecture-ready | Same loader; not run |
| **Qwen2 / Qwen2.5** | `models/llama.go` | Done — pretrained | Generic Llama-arch decoder. F32 top-5 = transformers; GGUF, GPTQ and AWQ verified; CPU, CUDA and WebGPU agree |
| **Qwen3 (dense)** | `models/llama.go` (`QKNorm`) | Done as a decoder body | Verified inside Qwen3-ASR and SDAR; standalone logits not yet checked |
| **Qwen3-MoE** | `models/llama.go` (`LlamaConfig.NumExperts…`, `llamaBlock.MoE`), `qwen_load.go` (`mapHFMoE`), `nn/moe.go` | ✅ Verified 2026-10-28 (architecture) | Routed FFN, both HF expert layouts, GGUF and 4-bit experts; tiny-random matches transformers on CPU and CUDA |
| **Qwen3.5 / 3.6 / 3.8 text (Gated DeltaNet hybrid)** | `models/qwen3_5.go`, `qwen3_5_load.go`, `qwen3_5_gguf.go`; `ops.GatedDeltaRule`, `kernels/cuda/src/gated_delta.cu` | ✅ Verified 2026-11-08 (0.8B, 9B, 27B BF16 / FP8 / GGUF) | Gated DeltaNet + gated attention. 0.8B / 9B / 27B (BF16, FP8, GGUF) match transformers; runs on CPU, CUDA and WebGPU. Image input (vision tower, MRoPE, Go preprocessing): 0.8B image chat = transformers, 16 / 16 greedy |
| **Llama-3 / 3.1 / 3.2** | `models/llama3_load.go` | Done — pretrained | llama3 RoPE scaling; Llama-3.2-1B top-5 = transformers; GGUF and GPTQ act-order verified |
| **Mistral 7B** | `models/llama3_load.go` | ✅ Verified 2026-11-09 (BF16) | Mistral-7B-Instruct-v0.3 BF16: top-5 ids = transformers, 16 / 16 greedy |
| **Granite 3.x / Granite 4.0 (attention)** | `models/granite_load.go` | ✅ Verified 2026-11-09 (granite-4.0-micro BF16) | Llama decoder + Granite multipliers; granite-4.0-micro 16 / 16 greedy. Mamba-hybrid layers not supported |
| **Ministral 3 (text)** | `models/ministral_load.go` | ✅ Verified 2026-11-09 (3B, FP8 → BF16) | YaRN + Llama 4 query scaling, FP8 → BF16 at load; Ministral-3-3B 16 / 16 greedy. Image input (Pixtral tower + projector): image chat = transformers, 16 / 16 greedy |
| Mixtral | `models/llama3_load.go` | Architecture-ready (SWA done) | Sliding window verified; HF `mixtral` not routed yet (ROADMAP) |
| **Phi-3 / Phi-3.5** | `models/llama3_load.go` (`LoadPhi3FromHF`) | Done — pretrained | Fused QKV split + LongRoPE; Phi-3.5-mini BF16 top-5 = transformers |
| **Phi-4-mini** | `models/llama.go` (`PartialRotary`), `llama3_load.go` | ✅ Verified 2026-10-29 | Partial rotary + LongRoPE, 200k vocab; BF16 top-5 and 16-token greedy = transformers |
| Phi-4 (14B) | — | Architecture-ready | Same template; needs 4-bit to fit 16 GB |
| **Gemma 3 (text)** | `models/gemma.go`, `gemma_load.go` | ✅ Verified 2026-10-28 | 1B F32 top-5 = transformers to 4 dp; 4B multimodal checkpoint loads text-only |
| **Gemma 2** | `models/gemma.go`, `gemma_load.go` | ✅ Verified 2026-10-29 | Softcaps + alternating local / global; 2B matches transformers to 4 dp |
| **Gemma 4 (text)** | `models/gemma4.go`, `gemma4_load.go` | ✅ Verified 2026-10-29 | Per-layer embeddings, KV sharing, proportional RoPE; E2B matches transformers; runs on CPU, CUDA and WebGPU. Image input: E2B image chat = transformers, 16 / 16 greedy |
| **Muse Glimmer 30B (text)** | `models/muse_glimmer_load.go`, `muse_glimmer_gguf.go` (on the `Gemma3` skeleton) | ✅ Verified 2026-11-08 (30B GGUF Q4) | Gemma 3 skeleton + Muse switches (NoPE globals, output gate, q scale); 30B GGUF Q4 top-5 = reference |
| **Mellum 2.1 12B-A2.5B** | `models/llama3_load.go` (`model_type` `mellum`, Qwen3-MoE path) | ✅ Verified 2026-11-07 (12B BF16, A100) | Qwen3-MoE + YaRN + per-layer-type RoPE; 12B BF16 matches transformers (A100) |
| **DeepSeek-V2 / V3** | `models/deepseek_v2.go`, `deepseek_v2_load.go`, `deepseek_route.go` | ✅ Verified 2026-11-08 (V2-Lite-Chat, A100); V3 routing tiny-verified | MLA (latent KV cache) + DeepSeekMoE; V2-Lite-Chat matches transformers (A100); V3 routing tiny-verified |
| **GLM-4 (glm, glm4)** | `models/glm_load.go` | ✅ Verified 2026-11-08 (GLM-4-9B-0414, A100) | Interleaved partial RoPE, post-norms; GLM-4-9B-0414 top-5 ids = transformers |
| **Nemotron** | `models/nemotron_load.go` | ✅ Verified 2026-11-08 (Nemotron-4-Mini 4B, A100) | LayerNorm1P + ReLU² MLP; Nemotron-4-Mini 4B top-5 ids = transformers, 16 / 16 greedy |
| **MiMo (Xiaomi MiMo-7B)** | `models/llama3_load.go` (`mimo` → Qwen2 loader), `models/mimo_mtp.go`, `models/spec_decode.go` | ✅ Verified 2026-11-08 (MiMo-7B-RL, A100) | Qwen2 + MTP head; MiMo-7B-RL matches transformers; MTP speculative decoding |
| **MiniMax-Text-01** | `models/minimax.go`, `minimax_load.go` | ✅ Tiny-verified 2026-11-08 | Lightning attention + MoE; tiny-random matches transformers |
| **Spark-X 2.5 / 2.6** | `models/spark_load.go` (Llama: `HeadwiseGate`, `GELUMLP`, `SlidingFullRotary`) | ✅ Verified 2026-11-11 (2.6-4B BF16) | Fused QKV, per-head sigmoid output gate, GELU MLP, sliding + partial-RoPE full layers. Remote code only: checked against an F64 PyTorch port (tiny 1.1e-5 CPU / WebGPU; 4B top-5 logits exact to 1e-4, 8 / 8 greedy). Real weights on WebGPU untested (iGPU memory) |
| **LFM3** | `models/lfm3.go`, `lfm3_load.go` | ✅ Verified 2026-11-11 (1.7B BF16) | Short-conv + full / sliding attention hybrid. No library implements it yet: checked against an F64 PyTorch port of the model code (tiny 1.7e-5; 1.7B top-5 logits 1e-4, 8 / 8 greedy); CPU = WebGPU |
| **Falcon** | `models/falcon.go`, `falcon_load.go` | ✅ Verified 2026-10-29 | Every decoder variant (ALiBi, MQA, new arch); falcon-rw-1b / 7b match transformers |
| **SDAR-1.7B-Chat (block-diffusion LM)** | `models/sdar.go` (`Llama.ForwardDiffusion`, `GenerateDiffusion`, `SDAROpts`), `models/llama.go` (block-bidirectional mask, no-store KV pass), `engine/engine.go` (diffusion branch) | ✅ Verified 2026-10-28 | Block-diffusion decoding on a Qwen3 body; first pass = transformers on CUDA BF16 |
| **MercyLM** | `models/mercy.go` | Done — pretrained | 15.4M GPT-style LM; `examples/38_mercy_chat` |
| **T5 encoder (T5 / v1.1 / Flan)** | `models/t5.go` | ✅ Verified 2026-11-09 (t5-v1_1-small) | Hidden states within 5.3e-6 of transformers on CPU and WebGPU; Flux / SD3 text encoder |
| **BERT** | `models/bert.go`, `models/bert_load.go` | Done — pretrained | bge-small-en-v1.5 embeddings match PyTorch to 7.8e-7 |
| **EmbeddingGemma 2** | `models/embeddinggemma2.go`, `embeddinggemma2_load.go` | ✅ Verified 2026-10-29 | Bidirectional Gemma 4 blocks; embeddings match transformers |

### 1.2 Speech and audio

| Model | File | Status | Notes |
|-------|------|--------|-------|
| **Whisper (tiny…large-v3)** | `models/whisper.go`, `whisper_load.go`, `audio/` | ✅ Verified 2026-10-28 | Mel, encoder and step-0 logits = transformers (tiny); greedy identical; `/v1/audio/transcriptions` |
| **Qwen3-ASR (1.7B / 0.6B)** | `models/qwen3_asr.go`, `qwen3_asr_load.go` | ✅ Verified 2026-10-28 | Audio encoder + Qwen3 decoder; values and greedy = transformers on 1.7B |
| **Mimi (codec)** | `models/mimi.go`, `mimi_load.go` | ✅ Verified 2026-10-29 | SEANet + transformer + split RVQ; codes and decoded audio match the reference |
| **Moshi (speech-to-speech LM)** | `models/moshi.go`, `moshi_load.go`, `moshi_converse.go`, `tokenizer/spm_model.go` | ✅ Verified 2026-11-06; real time on an L4 2026-11-08 | Temporal transformer + depformer over Mimi codes; Q8 GGUF real time on an L4 (58 ms / 80 ms frame) |

### 1.3 Diffusion

| Model | File | Status | Notes |
|-------|------|--------|-------|
| **Stable Diffusion 1.5** | `models/sd*.go` | Done — pretrained | CLIP + UNet + VAE + DDIM; text-to-image and img2img match diffusers within 2e-4 (CPU, LIGHT_HEAVY_TESTS). `light image`; BF16 on RTX 5060 Ti |
| **SDXL base** | `models/sdxl.go`, `sdxl_load.go` (UNet / CLIP generalised in `sd_unet.go`, `sd_clip.go`) | ✅ Verified 2026-11-09 | CLIP-L + OpenCLIP bigG (penultimate states, projected pooled), 3-level UNet with 1/2/10 transformer depth, text-time add-embedding, SDXL VAE; both encoders match diffusers to 1e-4, UNet step within 1.1 % (BF16 vs fp16); 1024² in 34 s / 30 DDIM steps on the RTX 5060 Ti (10.3 GB). `LoadSDXLFromHF`; `examples/43_sdxl_txt2img`; `scripts/verify/ref_sdxl.py` |
| **Flux (FLUX.1 schnell / dev)** | `models/flux.go`, `diffusion/flux.go` | ✅ Tiny-verified 2026-11-09 | MMDiT + T5 + CLIP pooled + flow-match Euler + 16-ch VAE; diffusers' FluxPipeline within 3.3e-6, FluxImg2ImgPipeline within 6.4e-6 (CPU, WebGPU). Real 12B weights not run yet |
| **FLUX.2 [dev]** | `diffusion/flux2.go` (Flux2Pipeline) | ✅ Tiny-verified 2026-11-11 | Mistral-Small-3 encoder with BFL's system message, guidance embedding; tiny pipeline within 7.2e-6 (CPU, WebGPU). Real weights: 64 GB DiT + 48 GB encoder, H100 / G4 |
| **FLUX.2 klein** | `models/flux2.go`, `diffusion/flux2.go` | ✅ Tiny-verified 2026-11-09 | Qwen3 encoder + FLUX.2 DiT; text-to-image within 7.4e-6, multi-reference editing within 8.9e-6 (CPU, WebGPU) |
| **Qwen-Image 2.1** | `models/qwenimage21.go`, `qwenimage_vae.go`, `diffusion/qwenimage21.go` | ✅ Tiny-verified 2026-11-12 | Block-causal single-stream DiT, Qwen3-VL pre-norm text states, Wan-2.2-style RGBA VAE, exponential shift + terminal stretch; text-to-image within 5.4e-6 (CPU, WebGPU). Editing and the prefix KV cache not wired |
| **FLUX 3 Image Klein** | `models/flux3.go`, `diffusion/flux3.go` | ✅ Tiny-verified 2026-11-11 | Per-mode + joint DiT, flow-map sampler, Qwen3-VL text (stacked layers, padded window), FLUX.2 VAE; transformer within 2.8e-6 (incl. references), text-to-image within 2.6e-6, editing (1 reference padded, several cropped) within 3e-6 (CPU, WebGPU). Real weights not run yet |
| **Z-Image (Turbo)** | `models/zimage.go`, `diffusion/zimage.go` | ✅ Tiny-verified 2026-11-09 | Qwen3 encoder + S3-DiT; ZImagePipeline within 4.6e-5, img2img within 6.9e-5 (CPU, WebGPU) |

### 1.4 Vision (classification and detection)

| Model | File | Status | Notes |
|-------|------|--------|-------|
| LeNet-5 | `models/lenet.go` | Training | MNIST 98.20% test acc in `examples/22_train_mnist` |
| AlexNet | `models/alexnet.go` | Training | 5.4M params; 100% eval on `examples/21_train_alexnet` |
| VGG (11/13/16/19) | `models/vgg.go` | Done — inference + GPU | 138.36M params (VGG-16) on RTX 5060 Ti, 166ms forward; `examples/28_vgg16` |
| **ResNet (CIFAR variants 20/32/44/56)** | `models/resnet.go` | Training | `examples/23_train_resnet_cifar10` — ResNet-20 → 82.84% on CIFAR-10 (no augmentation) |
| **ResNet9 (DAWNBench / fastai)** | `models/resnet9.go` | Training | 6.57M params; `examples/34_resnet9_cifar10` — ResNet9 + RandomCrop(pad=4) + RandomHorizontalFlip + per-channel normalize + AdamW + CosineLR. Targets 90+% test accuracy in 8 epochs |
| **ResNet (18/34/50/101/152)** | `models/resnet_imagenet.go` | Done — pretrained + train/infer split | torchvision-compatible (BasicBlock + Bottleneck); ResNet-50 = 25.56M params, 150ms; `examples/26_resnet50/{train,infer}` for from-scratch CIFAR-10, `examples/32_resnet_pretrained` for HF `microsoft/resnet-50` weights |
| **MobileNetV2** | `models/mobilenetv2.go` | Done — pretrained + train/infer split | 3.50M params (×1.0); inverted residuals + ReLU6; `examples/27_mobilenetv2/{train,infer}` for from-scratch CIFAR-10, `examples/33_mobilenet_pretrained` for HF `google/mobilenet_v2_1.0_224` weights |
| **ViT-B/16, ViT-L/16, ViT-H/14** | `models/vit.go` | Done — pretrained | ViT-B/16 = 86.57M params, 152ms; patch embed via Conv2d; CLS token + learned position; `examples/29_vit_b16`, `examples/31_vit_pretrained` (HF `google/vit-base-patch16-224`) |
| **YOLOv5 (n/s/m/l/x)** | `models/yolov5.go` | Done — inference + GPU | YOLOv5n = 1.87M params, 141ms forward at 640×640; CSPDarknet + PANet + Detect; `examples/30_yolov5n` |

Not-yet-ported torchvision / detection / segmentation / video architectures
are listed in ROADMAP.md.

---

## 2. Features

### Core types

- [x] `dtype.DType` enum + `Size()` / `IsFloat()` / `IsInt()` / `IsSigned()` / `IsQuantized()` / `BlockElems()` / `BlockBytes()` / `ByteSize()` / `String()` / `Valid()`
- [x] `dtype.Numeric` constraint and `dtype.Of[T]()`
- [x] `dtype.Half` / `dtype.BFloat16` software conversion (round-trip, subnormals, NaN/Inf; CPU BF16 rounds to nearest even, bit-identical to CUDA and torch since 2026-10-29)
- [x] `dtype.F8E4M3` + `dtype.Float8` conversion (fn encoding: RNE, saturate to ±448, NaN 0x7F; every code round-trips) — 2026-10-28
- [x] `shape.Shape` (immutable, negative indexing)
- [x] `shape.Strides` + contiguity check + `Permute` + `ContigWithDim0Broadcast`
- [x] `shape.Broadcast` / `BroadcastN` / `BroadcastStrides`
- [x] `device.Device` interface: `device.CPU()`, `device.CUDA(ordinal)`, `device.WebGPU()` (wgpu-native loaded at run time, no cgo), `device.Auto()` (CUDA → WebGPU → CPU), `IsCPU` / `IsCUDA` / `IsWebGPU`, `MustCUDA`, `CUDAAvailable`, `CUDAMemInfo`
- [x] `storage.Storage` CPU + `runtime.AddCleanup` via `storage.Track[T]`
- [x] `storage.Storage` CUDA (`cudaMalloc`/`cudaFree`, `cudaMemset` zero-init, refcount-driven cleanup, caching allocator)
- [x] `storage.Storage` WebGPU (`storage.NewWebGPU`, wgpu buffers; the adapter's own buffer / binding limits requested at device init; live-byte accounting `wgpudev.AllocatedBytes`)
- [x] `tensor.Tensor` constructors: `Zeros`, `Ones`, `Full[T]`, `New[T]`, `Arange`, `Rand`, `RandN`
- [x] Views: `Reshape` (with -1 infer), `Permute`, `Transpose`, `Contiguous`, `Narrow`, `Slice`, `Expand`, `AsStrided`
- [x] `tensor.To(dev)` across all three device kinds; per-kind `Contiguous` / `Copy` hooks (`tensor.RegisterContig`, `tensor.RegisterCopy`)
- [x] View sharing via atomic refcount (`storage.Track` / `Release`)
- [x] `tensor.ToSlice[T]` / `MustSlice[T]` / `ToHalfSlice` / `ToBF16Slice`
- [x] Autograd fields on `Tensor` (`requiresGrad`, `grad`, `gradFn`, `outputNr`, `version`)

### Linear algebra

- [x] MatMul 2-D (CPU F32/F64/F16/BF16/I32/I64; CUDA cuBLAS Sgemm/Dgemm + `GemmEx` for F16/BF16 with the transposed-weight fast path and plain layout; auto-contiguifies strided operands) — F16/BF16 CUDA 2026-10-28
- [x] CPU F32 matmul: AVX2/FMA Go-assembly `Dot` / `Axpy` and a 4 × 3 register-tiled `simd.MatTB` (112 GFLOP/s on one core), parallel over output columns (`internal/parallel`), pure-Go fallback off amd64 — 2026-11-08
- [x] Batched MatMul (CPU F32/F64/F16/BF16, parallel over batches; CUDA `cublasGemmStridedBatched(Ex)` F32/F16/BF16)
- [x] Mixed-precision MatMul (`matmul_mixed`: F32×{BF16,F16}→F32, BF16×BF16→BF16, F16×F16→F16; F32 accumulate via `cublasGemmEx`; CPU reference kernel)
- [x] **MoE grouped GEMM** (`ops.MoEGemm`, Candle `moe_gemm` contract: stacked (E, N, K) expert weights, expert-sorted (token, slot) pairs, optional routing weights, per-token or per-pair input rows; CUDA tiled kernel over all experts in one launch, CPU reference; F32/BF16/F16) + `ops.MoERoute` (top-k, `norm_topk_prob`, expert sort on device) + `ops.MoERouteFromIDs` (explicit selections, custom routers) + `ops.MoECombine` + `nn.SparseMoE` (router → gate/up grouped GEMM → act ⊙ up → down grouped GEMM, routing-weighted sum; optional `Router` hook and `RouterF32`) — 2026-10-28, matches the per-expert index_select/index_add loop to 1e-5
- [x] **Quantized MoE grouped GEMM** (`ops.MoEGemmQ`: F32 activations × GGUF Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K expert stacks (E, N, K), fused dequant CUDA kernels in `moe_gguf.cu`, dequant + MoEGemm reference elsewhere; `nn.SparseMoE` runs quantized experts; GGUF `qwen3moe` loader fuses `ffn_gate_exps`/`ffn_up_exps` at the byte level) + `quant.Quantize` encoders for Q4_0/Q5_0/Q8_0 — 2026-10-28
- [x] **4-bit MoE experts** (`ops.MoEInt4`: only the experts the routing uses are dequantized into compact stacks in the shared weight scratch, then the grouped MoEGemm) — 2026-11-08
- [x] **FP8 MatMul** (`ops.MatMulF8`: E4M3 × E4M3 with per-tensor scales → F32/BF16/F16; CUDA via cuBLASLt TN GEMM on tensor cores, K % 16 == 0, skipped with an explanation below compute capability 8.9; CPU reference; `ops.QuantizeF8` amax/448 quantiser; `nn.Linear` runs F8 weights with `WScale`) — 2026-10-28
- [x] **FP8 block-scaled weights** (`ops.DequantizeF8Block`, `ops.MatMulF8Block`; fused CUDA `fp8_block_gemv` for M ≤ 8 and one-pass `fp8_block_dequant` for prefill, decoding E4M3 without FP8 hardware (sm_80); `nn.Linear.WBlockScale`; weights stay 1 byte / param, expansion into a reused scratch buffer) — 2026-11-07
- [x] Int8 weight MatMul (`ops.MatMulInt8`, per-column scales; CPU + CUDA F32 activations)
- [x] Einsum (`ops.Einsum`, general CPU; CUDA F32 fully on-device: diagonals via `tensor.AsStrided`, N-ary pairwise contraction, batched matmul; F64 on CUDA falls back to host)

### Shape / layout ops

- [x] Cast (all dtype pairs on CPU; CUDA every dtype ↔ F32 kernel, other pairs hop through F32; WebGPU F32/U32/I32/I64 and F32 ↔ BF16 / F16 with round-to-nearest-even)
- [x] Concat / Stack (all dtypes, all three devices)
- [x] Split / Chunk (`ops/split.go`)
- [x] IndexSelect, Gather, ScatterAdd, Scatter (set), IndexAdd, SliceScatter, SliceAssign (CPU + CUDA; gather/scatter/scatter_add/index_add at F32/F64/I32/I64/BF16/F16, CUDA accumulations atomic) — index ops widened 2026-10-28
- [x] Pad, Flip, Roll (CPU + CUDA F32)
- [x] Cumsum / Cumprod (CPU + CUDA F32)
- [x] Tril / Triu, mask ops, Where (CPU + CUDA)
- [x] Sort / TopK (CPU all; CUDA bitonic kernels F32/F64/I32/I64, bit-identical tie order; TopK = sort + narrow on CUDA)
- [x] Linspace / Meshgrid / Eye (`tensor/constructors.go`)
- [x] Unfold, PixelShuffle / PixelUnshuffle, ReplicationPad2d (device-agnostic compositions)

### Normalization & activations

- [x] Softmax / LogSoftmax (fused, block reduction; CUDA F32/F64 + BF16 softmax; CPU BF16 via F32)
- [x] LayerNorm, RMSNorm (fused; CUDA F32 + BF16; CPU BF16 via F32 with one rounding)
- [x] GroupNorm (CUDA F32 + BF16; CPU F32/F64) — SD 1.5
- [x] InstanceNorm2d (`ops.InstanceNorm`, `nn.InstanceNorm2d`)
- [x] BatchNorm2d train (native CUDA stats + apply + backward) and eval (composed)
- [x] Dropout (train-mode inverted, CPU + CUDA)
- [x] GELU tanh-approx (WebGPU clamps the tanh argument to ±15: Intel's D3D12 `tanh` returns NaN for large inputs) + exact-erf `ops.GeluErf` (BERT, Falcon; CUDA F32/F64/BF16/F16), GeGLU fused (CUDA), ReLU6 (composed), Softplus (torch threshold 20; CUDA F32)
- [x] Candle-parity op batch (2026-10-28, CPU + CUDA kernels with parity tests): `Sign`, `Erf`, `Recip`, `Sqr`, `Mish`, `HardSwish`, `HardSigmoid`, `Clamp`, `ELU`, `LeakyReLU`, `PowScalar`, `Affine`, `RoundTo`, `Maximum`/`Minimum` (broadcast), compare/where on F64/I32/I64/U8/BF16 CUDA, `RoPEInterleaved` (GPT-J), `RoPETHD`, `Unfold`, `PixelShuffle`/`PixelUnshuffle`, `ReplicationPad2d`, `ConvTranspose1d/2d` (stride/padding/output_padding/dilation/groups), `UpsampleBilinear2d`

### Convolution & pooling

- [x] Conv1d (CPU + CUDA F32 forward; no backward), causal depthwise Conv1d with carried state (`ops.CausalConv1dDepthwise`, CPU + CUDA F32)
- [x] Conv2d (cuDNN on CUDA incl. BF16; forward + backward; grouped / depthwise)
- [x] Conv3d (CPU + CUDA F32 forward)
- [x] ConvTranspose1d / 2d (`ops.ConvTranspose2d`, `nn.ConvTranspose2d`; CUDA `conv_transpose.cu`, 2026-10-28; matches `F.conv_transpose2d`)
- [x] MaxPool2d / AvgPool2d (forward + backward, CPU + CUDA F32)
- [x] AdaptiveAvgPool2d (composed on CPU/CUDA; native WebGPU)
- [x] UpsampleNearest (forward; CUDA F32 + BF16), UpsampleBilinear2d (align_corners both ways, up and down)

### Attention & sequence mixers

- [x] RoPE (`ops.RoPE`, rotate-half / HF convention, pinned by `TestRoPEMatchesHFRotateHalf`), interleaved GPT-J pairing (`ops.RoPEInterleaved`), frequency scaling (`ops.RoPEWithInvFreq`, `models.RopeScaling`: `llama3`, `longrope` with attention factor, `yarn` with mscale, linear), partial rotary, per-layer-type RoPE; CUDA F32 + BF16; WebGPU F32
- [x] SDPA (`nn.SDPA`) with causal, sliding-window and ALiBi masks
- [x] FlashAttention (`ops.FlashAttention`; CPU online softmax; CUDA tiled D∈{32,64,128} + generic ≤256). 16× faster than SDPA at S=1024
- [x] SelfAttention / MultiHeadAttention / CrossAttention (autograd-aware, batched)
- [x] GroupedQueryAttention (in `models.Llama`), KVCache (prefill + decode, `Truncate` for speculative decoding, per-layer linear-attention state)
- [x] Sliding-window attention inside `models.Llama` (`LlamaConfig.SlidingWindow` + per-layer `SlidingLayers`, rotating KV cache; matches transformers `use_sliding_window` on Qwen2-0.5B to 5e-3; HF `sliding_window`/`layer_types`/`max_window_layers` and GGUF `attention.sliding_window` parsed); NoPE layers (Muse Glimmer)
- [x] Ring KV cache + split-K single-query decode attention (`ops.KVRingWrite`, `ops.AttnDecode`; CPU + CUDA F32) — 2026-11-08
- [x] Multi-head latent attention (DeepSeek MLA, latent-only KV cache) — 2026-11-08
- [x] Gated delta rule (`ops.GatedDeltaRule`: recurrent + chunked forms, carried state, in-kernel q/k L2 norm; CPU, CUDA recurrent + chunked prefill kernel) — 2026-11-06 / 11-07
- [x] Lightning attention (MiniMax, chunked block algorithm from existing ops) — 2026-11-08
- [x] TransformerBlock (pre-norm)
- [x] BF16 inference path (weights, activations, KV cache, scores, LM head; norms stay F32; BF16 Linear + bias rounds once)
- [x] Speculative decoding (`models.SpecGreedy` with an MTP draft head; output equals greedy) — 2026-11-08
- [x] Block-diffusion decoding (`Llama.ForwardDiffusion`, `GenerateDiffusion`)
- [x] `audio`: WAV decoder, resampler, `audio.LogMel` (Whisper), `audio.LogMelFrames` (any length) + `audio.MelFilterBank` (slaney, matches transformers' `mel_filter_bank`)
- [x] `models.CausalLM` interface; `Llama.Embed` + `Llama.ForwardEmbeds` for multimodal front ends

### Autograd

- [x] `Function` / `Context` / `FunctionNode` (PyTorch-style dynamic DAG), reverse-mode `Backward`, `GradCheck`
- [x] Functions: Add, Sub, Mul, Div, Neg, MatMul (batched), ReLU, ReLU6, Sigmoid, Tanh, SiLU, GELU, Sum, Mean, AxisSum, AxisMean, Reshape, Transpose, Permute, Contiguous, IndexSelect, LayerNorm, RMSNorm, Softmax, LogSoftmax, CrossEntropyLoss, Conv2d, Conv2dGrouped, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, BatchNorm2d, Dropout
- [x] Also differentiable: `Exp`, `Log`, `Abs`, `Sqr`, `ELU`, `LeakyReLU`, `Mish`, `HardSwish`, `HardSigmoid`, `Clamp`, `Maximum`, `Minimum` (grad to the winner), `PReLU`, `Narrow`, `Concat`; losses `MSELoss`, `NLLLoss`, `BCEWithLogitsLoss`, `HuberLoss` (all grad-checked)
- [x] `nn.LayerNorm` / `nn.RMSNorm` route through autograd only when a gradient is wanted (fixed 2026-10-27; had silently broken training through norms)

### NN modules

- [x] `nn.Module`, `nn.Sequential`, `nn.Linear` (+ `Tied`, `Uninit`, `WithDType`, placeholders; F32/F64/BF16/F16/F8E4M3 (per-tensor or block-scaled)/GGUF/int4 weights), `nn.Embedding` (incl. quantized tables, `NewEmbeddingFromWeight`)
- [x] `nn.ReLU` / `SiLU` / `Sigmoid` / `Tanh`, ELU, LeakyReLU, Mish, HardSwish, HardSigmoid, GELU, ReLU6, PReLU, SwiGLU modules (all with gradients)
- [x] `nn.LayerNorm`, `nn.RMSNorm`, `nn.GroupNorm`, `nn.InstanceNorm2d`, `nn.BatchNorm2d`, `nn.Dropout`
- [x] `nn.Conv2d` / `GroupedConv2d` / `DepthwiseConv2d` / `ConvTranspose2d` / `MaxPool2d` / `AvgPool2d` / `AdaptiveAvgPool2d` / `Flatten`
- [x] `nn.SelfAttention`, `nn.MultiHeadAttention`, `nn.CrossAttention`, `nn.SDPA`, `nn.TransformerBlock`
- [x] `nn.LSTM` / `nn.GRU` (torch layout, match torch to 1e-5 on CPU + CUDA, trainable)
- [x] `nn.SparseMoE` (dense, GGUF-quantized and 4-bit experts; lazy construction for GGUF loads)
- [x] `nn.VarMap` (Candle VarMap): named tensor set, `Save` to safetensors/npz, `Load` in place from safetensors/npz/pth with shape checks, `Prefixed` views; `nn.ReadCheckpoint` dispatches on extension — 2026-10-29

### Loaders

- [x] safetensors single-file + **sharded** (`model.safetensors.index.json`; reads HF Python cache layout `refs/<rev>` + `snapshots/<sha>/`), read + write, direct-to-device, `F8_E4M3`
- [x] GGUF (read + minimal write; all 12 GGML dtypes); llama / qwen2 / qwen3 / mistral / qwen3moe via `LoadLlamaGGUF`, `qwen35`, `muse-glimmer`, Moshi; `LoadChatModelGGUF` dispatch on `general.architecture`
- [x] NumPy `.npy` / `.npz` read + write (`light/npy`: all NumPy numeric dtypes, Fortran order, stored or deflated members, lazy `Npz.Get`) — 2026-10-29
- [x] PyTorch `.pth` / `.pt` / `.bin` (`light/pth`: pickle VM for protocols 2–5, `_rebuild_tensor_v2` / `_rebuild_parameter` / `_rebuild_from_type_v2`, strided and offset views gathered contiguous, state dicts, checkpoint dicts by key, pickled `nn.Module` trees flattened to dotted names) — 2026-10-29
- [x] GPTQ (gptq / gptq_v2, ordered groups and act-order `desc_act`) and AWQ (GEMM) 4-bit checkpoints, converted once at load to Light's `quant.Int4` layout; all-zero exporter biases dropped — 2026-11-08
- [x] Fine-grained FP8 checkpoints (`quantization_config.quant_method: fp8`, 128 × 128 `weight_scale_inv`) — 2026-11-07
- [x] HF Hub downloader (`hub.Download`, `hub.DownloadModelSafetensors`, `hub.ListFiles`, `DownloadWithProgress` / `DownloadTo`; `.part` resume with Range; local checkpoint directories; `HF_TOKEN` / `HF_ENDPOINT`)
- [x] MNIST, CIFAR-10 (`light/data`) + augmentation (`RandomCropPad`, `RandomHorizontalFlip`, `NormalizeCIFAR10`)

### Tokenizers

- [x] BPE byte-level (GPT-2 / Llama-3 / Qwen), special tokens from `added_tokens` (+ `lstrip` / `rstrip`); `Split` pre-tokenizers with `Removed` + `invert` (GPT-4o / Phi-4)
- [x] BPE SentencePiece-style (Llama-2 / Mistral / Phi-3: Prepend `▁` normalizer, rune-level merges, `<0xNN>` byte fallback) — auto-detected from `tokenizer.json`; ids identical to HF on Phi-3.5
- [x] CLIP tokenizer (SD 1.5)
- [x] WordPiece + BERT normalizer + `[CLS]`/`[SEP]` post-processor (exact match vs HF on bge-small)
- [x] Unigram / SentencePiece (Viterbi, byte fallback); raw SentencePiece `.model` protobuf (`tokenizer.LoadSentencePieceModel`, verified vs the sentencepiece package)
- [x] `tokenizer.Load` / `tokenizer.Parse` dispatch on `tokenizer.json` model type
- [x] **GGUF-metadata tokenizer** (`tokenizer.LoadGGUF` / `FromGGUFMeta`): `llama` (SentencePiece: score-ranked merges, `▁` prefix, byte fallback) and `gpt2` (byte-level BPE with llama.cpp's per-family pre-tokenizer regex chains for `tokenizer.ggml.pre` = default / llama-bpe / qwen2 / qwen35 / llama4 / gpt-2 / deepseek-llm / deepseek-coder / falcon / starcoder-family / chatglm / poro …, `ignore_merges`), control + user-defined tokens as specials, BOS/EOS/EOT/add_bos; identical ids to llama.cpp on all 46 cases of its 10 vocab fixtures — 2026-10-29
- [x] `models.BOSTokenID` from `tokenizer.json` post_processor; `models.EOSTokenIDs` from generation_config / config
- [x] **Chat templates (Jinja2 subset)**: pure-Go Jinja engine (`internal/jinja`: whitespace control incl. trim_blocks/lstrip_blocks, if/for/set/namespace/macro/filter/with, loop.*, break/continue, Python-semantics values, str/dict/list methods, builtin filters/tests, transformers' `tojson`) + `chattemplate` (HF `chat_template.jinja` / `tokenizer_config.json` / GGUF `tokenizer.chat_template`, `raise_exception`, `strftime_now`, special tokens). The engine renders prompts with the model's own template and falls back to the hand-ported family template. Byte-identical to transformers 5.21 `apply_chat_template` on 242 renders of 11 templates (Qwen2.5, Qwen3, Llama-3.2, Phi-3.5, Phi-4-mini, Gemma 2/3, Mistral v0.3, SmolLM2/3, DeepSeek-R1-Distill) and to jinja2 3.1.6 on 98 engine cases (`scripts/verify/ref_chat_templates.py`, `ref_jinja_cases.py`) — 2026-11-06

### Quantization

- [x] Dequant Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q8_1 / Q2_K–Q8_K (CPU + CUDA parity; WebGPU Q4_0 / Q5_0 / Q8_0 / Q4_K / Q6_K)
- [x] Quantize Q4_0 / Q5_0 / Q8_0 (`quant.Quantize`, ggml reference rounding; round-trip tested) — 2026-10-28
- [x] Int8 per-column weight quant (`quant.QuantizeInt8PerColumn`) + int8 × F32 matmul
- [x] **INT8 W8A8 GEMM** (`ops.MatMulW8A8`, 2026-11-09): activations quantised per row to int8 on the device, int8 × int8 → int32 through `cublasGemmEx(CUBLAS_COMPUTE_32I)` (tensor-core IMMA), row × column scale epilogue; Go reference on other devices (bit-exact int32 path); 88–93 TOPS at M ≥ 512 vs 41–44 TFLOPS for BF16 cuBLAS on the RTX 5060 Ti
- [x] **GGUF quantized inference**: `ops.MatMulQ` (fused dequant-matmul CUDA kernels for Q4_0 / Q5_0 / Q8_0 / Q4_K / Q5_K / Q6_K; dequant+matmul fallback for the rest), `nn.Linear` accepts quantized weights, `models.LoadLlamaGGUF` (Q/K un-permute for llama arch, `rope_freqs` factors). Qwen2-0.5B Q8_0 / Q4_K_M and Llama-3.2-1B Q4_K_M match transformers greedy; logits within quant noise
- [x] **Quantized CPU kernels** (2026-11-08): GGUF `matmul_q` for Q4_0 / Q5_0 / Q8_0 / Q4_K / Q6_K and GPTQ / AWQ `int4_gemv` keep weights quantized; AVX2 row decoders (`kernels/cpu/qdecode_amd64.s`, bit-identical to the Go decoders, 5–9× faster per row) feed 48-row tiles into `simd.MatTB`; GGUF tied heads stay quantized (`MatMulQFusedOn`)
- [x] Quantized embedding lookup: `IndexSelect` on every GGUF block format and the narrow fixed-width dtypes (byte-row gather, CPU + CUDA + WebGPU); `nn.Embedding` with a quantized table dequantizes only the gathered rows; `LoadLlamaGGUF` keeps `token_embd` quantized (tied head via `ops.MatMulQFused`) — 2026-10-29
- [x] Tokenizer from GGUF metadata (`tokenizer.LoadGGUF`, 2026-10-29)
- [x] **GPTQ / AWQ 4-bit** (`quant.Int4` canonical layout, `quant.FromGPTQ` incl. act-order `Perm`, `quant.FromAWQ`; `ops.DequantizeInt4`, `ops.MatMulInt4` with fused CUDA `int4_gemv` for ≤ 8 rows and dequant-into-scratch + GEMM above; CPU and WebGPU kernels; 4-bit MoE experts) — 2026-11-08
- [x] FP8 E4M3 tensor-core GEMM (cuBLASLt) + safetensors `F8_E4M3` load/save — 2026-10-28; blockwise (DeepSeek / Qwen `weight_scale_inv`) scales with fused kernels — 2026-11-07

### Backends

- [x] **CUDA**: default build; PTX modules built with NVRTC 13.0 (`scripts/nvrtc_ptx.py`, no nvcc or host compiler needed; loads on CUDA 13.0 drivers) or `scripts/build_ptx.sh` (compute_80), embedded and JIT-compiled by the driver; cuBLAS / cuBLASLt / cuDNN bindings; driver-context binding per thread (`cudart.ensureContext`) and module-load retry; caching allocator; weight-sized temporaries reuse one scratch buffer per device / dtype. `-tags nocuda` opts out; `CGO_ENABLED=0` picks the CPU-only build automatically
- [x] CUDA graph decode (2026-11-12): the engine captures a model's single-token step once (`models.DecodeGraph`; position, KV append slot and attention length read on the device via `ops.KVAppend` / `ops.DecodeAttention`) and replays it per token, one upload and one launch; Llama-family and Qwen3.5 opt in, others decode eagerly; `LIGHT_CUDA_GRAPHS=0` disables
- [x] **CPU**: AVX2/FMA SIMD (CPUID + XGETBV detection), multithreaded matmuls, quantized kernels (see above); BF16 rmsnorm / layernorm / rope / softmax via F32 — 2026-11-08
- [x] **WebGPU** in every build without cgo (`-tags nowebgpu` opts out): `internal/wgpudev` (WGSL shaders in `kernels/webgpu`) over `internal/gpu` / `internal/ffi` (vendored from gfx; run-time library loading via Windows syscall / purego); wgpu-native loaded from `$LIGHT_WGPU_LIB`, next to the executable, `~/.cache/light/wgpu/` (`scripts/fetch-wgpu.sh`) or the system path; cross-builds for Linux / macOS without cgo — 2026-11-08
- [x] WebGPU LLM inference (2026-11-08): `rope` / `rope_i` / `rope_scaled`, `int4_dequant` / `int4_gemv` (GPTQ / AWQ), GGUF `matmul_q` fused GEMV / `dequant` / quantized gather for Q4_0 / Q5_0 / Q8_0 / Q4_K / Q6_K; Llama-family checkpoints in F32, GPTQ / AWQ and GGUF match the CPU (Qwen2.5-0.5B, TinyLlama GPTQ act-order) on Intel integrated graphics over D3D12; `-device webgpu` in the engine and CLI
- [x] **16-bit weight storage under F32 compute (`w16`)** on the CPU and WebGPU: `ops.MatMulW16` (weight widened exactly inside the matmul), weights allocated at 16 bits during load (`nn.WithWeightStorage16`), WebGPU embeddings over the buffer cap kept in host memory; the CLI default on both (lossless for 16-bit checkpoints) — 2026-11-09
- [x] WebGPU dispatch batching: one open encoder / compute pass, submit every 256 dispatches (`$LIGHT_WGPU_BATCH`) or before host-ordered work; per-dispatch uniforms in a 1 MB ring uploaded once per submit — 2026-11-08

### Engine, CLI and API

- [x] Chat engine (`engine/`): catalog of HF-cache + GGUF models (chat, STT, embedding, speech, pytorch kinds), chat templates (the model's own Jinja `chat_template`, else built-in ChatML, Llama-3, Phi-3, Phi-4, Mistral, Gemma), streaming generation, diffusion-LM streaming, MTP speculative decoding for greedy requests
- [x] OpenAI-compatible API: `/v1/models`, `/v1/chat/completions`, `/v1/audio/transcriptions` (Whisper, Qwen3-ASR), `/v1/embeddings` (EmbeddingGemma 2), speech-to-speech `/v1/audio/conversations` (Moshi); speech slot releases the chat model
- [x] **`light` CLI** (`cmd/light`, replaces `cmd/light-serve`; 2026-11-08): interactive shell with `/` commands (`/model`, `/list`, `/pull`, `/info`, `/rm`, `/clear`, `/system`, `/set temp|top-p|top-k|max-tokens|seed|device|dtype`, `/verbose`, `/serve [addr|stop]`, `/unload`, `/bench`, `/perf`, `/help`, `/exit`; unique prefixes, `"""` multi-line, Ctrl-C stops a reply); subcommands `list` (`-json`), `pull <org/repo> [file ...]`, `run <model> [prompt]` (shell or one-shot / stdin), `info`, `rm`, `serve`, `bench`, `perf`. Device defaults to CUDA when built in, else the CPU (F32 for safetensors ≤ 6 GB); GOGC 25
- [x] Per-host performance table (`~/.cache/light/perf.json`, `$LIGHT_PERF_FILE`): per host (CPU, RAM, OS, GPU names per device kind) and model × device: dtype, load time, host RAM and GPU memory growth, prompt length, prefill and decode tok/s; filled by every load / reply and by `light bench` — 2026-11-08
- [x] Light Studio (Wails v3 + React desktop app) — moved to its own repository, `light-studio`, on 2026-11-08 (history kept; builds against a sibling checkout of this repo)

### Training

- [x] SGD (momentum + weight decay), AdamW, CosineLR (+ warmup)
- [x] Real recipes: MNIST 98.2%, CIFAR-10 ResNet-20 82.8%, ResNet9 90%+

### Infra

- [x] Op dispatch registry (`internal/registry`), `scripts/opmatrix` coverage dump (§3)
- [x] Reference harness: `scripts/verify/` (numpy, transformers, per-model `ref_*.py`, tiny-random fixture makers, GGUF fixtures); Colab runner (`scripts/colab`) for heavy tests on A100 / L4 (`// colab-download:` limits per-test downloads)
- [x] Tools: `scripts/hfget`, `scripts/gguflist`, `scripts/ggufdump`

### Examples

01–25 (tensor basics → autograd training → GPT-2 pretrained), plus:

- [x] `26_resnet50`, `27_mobilenetv2`, `28_vgg16`, `29_vit_b16`, `30_yolov5n` — random-init forward (+ train/infer splits for 26–29)
- [x] `31_vit_pretrained`, `32_resnet_pretrained`, `33_mobilenet_pretrained` — HF weights
- [x] `34_resnet9_cifar10` — augmented training recipe
- [x] `35_qwen_inference` — generic causal LM (`-repo`, `-gguf`, `-bf16`, `-bos`, `-rope-scaling`, `-logits`); Qwen2, Llama-3.2, Phi-3.5
- [x] `36_qwen_chat` — multi-turn with KV reuse
- [x] `37_sd15_txt2img` — Stable Diffusion 1.5
- [x] `38_mercy_chat` — MercyLM
- [x] `40_bert_embeddings` — sentence embeddings, cosine matrix
- [x] `41_whisper`, `42_qwen3_asr` — speech-to-text
- [x] `25_gpt2_pretrained -device webgpu` — GPT-2 on WebGPU

---

## 3. Kernel coverage

**Generated, not hand-edited:** run `go run ./scripts/opmatrix -md` and paste
the table here after adding kernels.

Current table (113 registered ops): the CPU and WebGPU columns were regenerated on 2026-11-08 (`CGO_ENABLED=0 go run ./scripts/opmatrix -md` on Windows, where the CUDA kernels are not linked); the CUDA column is the last CUDA-build run (2026-10-28) plus the CUDA registrations added since, read from `kernels/cuda/*.go` (`gated_delta_rule`, `causal_conv1d_depthwise`, `softplus`, `int4_*`, `kv_ring_write`, `attn_decode`, `fp8_block_*`, BF16/F16 `gelu_erf`). Rerun the generator on a CUDA build to refresh that column. Rows list the dtypes with a registered kernel per device.

`adaptive_avgpool2d` and `batchnorm_eval` are composed from other ops on
CPU/CUDA and only appear as native kernels on WebGPU.

**CUDA parity:** every op runs on the device. `sort` has bitonic kernels (F32/F64/I32/I64); `topk` lowers to sort + narrow; `einsum` F32 runs fully on CUDA (diagonals via `AsStrided`, N operands pairwise, pairs via batched matmul; F64 einsum still falls back to the host); `lpnorm` lowers to abs/pow/sum. The rows without a CUDA kernel entry are these lowered ops (and the WebGPU-only `dequant`), not gaps. `moe_gemm_q`, `geglu` and the `fp8_block_*` ops are CUDA-only kernels; on other devices `ops` takes dequantize + reference paths. `matmul_q` has CPU (AVX2), CUDA and WebGPU kernels.

**WebGPU:** F32 activations only (no F16/BF16/F64 arithmetic; BF16 / F16 weights are stored and widened inside `matmul_w16`); no `conv1d`, `gather`, `cumsum`, compare, `groupnorm`, `where`, `flash_attention` or MoE kernels. LLM inference works for Llama-family, Gemma 3 / 4, Qwen3.5 (Gated DeltaNet), Phi-4-mini, Granite and Ministral checkpoints with 16-bit or F32 weights, GPTQ / AWQ 4-bit and GGUF (Q4_0 / Q5_0 / Q8_0 / Q4_K / Q6_K).

| Op | CPU | CUDA | WebGPU |
|----|-----|------|--------|
| `abs` | F16 BF16 F32 F64 U8 I64 I8 I32 | F32 F64 | F32 |
| `adaptive_avgpool2d` | — | — | F32 |
| `add` | F16 BF16 F32 F64 U8 I64 I8 I32 | F16 BF16 F32 F64 I64 I32 | F32 |
| `affine` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `all` | U8 | U8 | — |
| `any` | U8 | U8 | — |
| `argmax` | F32 F64 I64 I32 | F32 | — |
| `argmin` | F32 F64 I64 I32 | F32 | — |
| `attn_decode` | F32 | F32 | — |
| `avgpool2d` | F32 F64 | F32 | F32 |
| `batchnorm_eval` | — | — | F32 |
| `cast` | F16 BF16 F32 F64 U8 U32 I64 I16 F8E4M3 U64 U16 I8 I32 | F16 BF16 F32 F64 U8 U32 I64 I8 I16 F8E4M3 U64 I32 U16 | F32 U32 I64 I32 |
| `causal_conv1d_depthwise` | F32 | F32 | — |
| `ceil` | F16 BF16 F32 F64 | F32 F64 | F32 |
| `clamp` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `concat` | F16 BF16 F32 F64 U8 U32 I64 U16 I32 I8 U64 I16 | F16 BF16 F32 F64 U8 U32 I64 I32 I8 U16 I16 U64 | F32 F64 U32 I64 U64 I32 |
| `conv1d` | F32 F64 | F32 | — |
| `conv2d` | F32 F64 | BF16 F32 | F32 |
| `conv3d` | F32 F64 | F32 | — |
| `conv_transpose2d` | F32 F64 | BF16 F32 | — |
| `cos` | F16 BF16 F32 F64 | F32 F64 | F32 |
| `cumprod` | F32 F64 I64 I32 | F32 | — |
| `cumsum` | F32 F64 I64 I32 | F32 | — |
| `dequant` | — | — | Q4_0 Q4_K Q5_0 Q6_K Q8_0 |
| `div` | F16 BF16 F32 F64 U8 I64 I8 I32 | F16 BF16 F32 F64 | F32 |
| `dropout` | F32 F64 | F32 | — |
| `einsum` | F32 F64 | — | — |
| `elu` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `eq` | F16 BF16 F32 F64 U8 U32 I64 I32 I8 I16 U16 U64 | BF16 F32 F64 U8 I64 I32 | — |
| `erf` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `exp` | F16 BF16 F32 F64 | F16 BF16 F32 F64 | F32 |
| `flash_attention` | F32 F64 | F32 | — |
| `flip` | F32 F64 U8 I64 I32 I8 | F32 | — |
| `floor` | F16 BF16 F32 F64 | F32 F64 | F32 |
| `fp8_block_dequant` | — | F32 BF16 | — |
| `fp8_block_gemv` | — | F32 BF16 | — |
| `gated_delta_rule` | F32 | F32 | — |
| `gather` | F16 BF16 F32 F64 I64 I32 | F16 BF16 F32 F64 I64 I32 | — |
| `ge` | F16 BF16 F32 F64 U8 U32 I64 U64 U16 I32 I16 I8 | BF16 F32 F64 U8 I64 I32 | — |
| `geglu` | — | BF16 F32 | — |
| `gelu` | F16 BF16 F32 F64 | BF16 F32 F64 | F32 |
| `gelu_erf` | F16 BF16 F32 F64 | F16 BF16 F32 F64 | — |
| `groupnorm` | F32 F64 | BF16 F32 | — |
| `gt` | F16 BF16 F32 F64 U8 U32 I64 U16 I16 I32 U64 I8 | BF16 F32 F64 U8 I64 I32 | — |
| `hard_sigmoid` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `hard_swish` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `index_add` | F16 BF16 F32 F64 I64 I32 | F16 BF16 F32 F64 I64 I32 | — |
| `index_select` | F16 BF16 F32 F64 U8 U32 I64 Q8_0 I8 Q3_K I16 F8E4M3 Q6_K Q4_K Q2_K Q8_K U16 Q5_K Q5_1 Q5_0 Q4_1 I32 Q4_0 U64 Q8_1 | F16 BF16 F32 U8 U32 I8 Q5_0 Q8_1 Q4_K Q5_K U64 Q2_K Q5_1 Q4_0 Q6_K Q3_K F8E4M3 I16 Q4_1 U16 Q8_K Q8_0 | F32 F64 U32 I64 Q6_K Q4_0 I32 Q5_0 Q8_0 Q4_K U64 |
| `int4_dequant` | BF16 F32 | F32 BF16 | F32 |
| `int4_gemv` | F32 | F32 BF16 | F32 |
| `kv_ring_write` | F32 | F32 | — |
| `l2norm` | F32 F64 | F32 F64 | — |
| `layernorm` | BF16 F32 F64 | BF16 F32 | F32 |
| `le` | F16 BF16 F32 F64 U8 U32 I64 I8 I16 I32 U16 U64 | BF16 F32 F64 U8 I64 I32 | — |
| `leaky_relu` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `log` | F16 BF16 F32 F64 | F16 BF16 F32 F64 | F32 |
| `logical_and` | U8 | U8 | — |
| `logical_not` | U8 | U8 | — |
| `logical_or` | U8 | U8 | — |
| `logsoftmax` | F32 F64 | F32 | F32 |
| `lpnorm` | F32 F64 | — | — |
| `lt` | F16 BF16 F32 F64 U8 U32 I64 I16 I32 U16 U64 I8 | BF16 F32 F64 U8 I64 I32 | — |
| `matmul` | F16 BF16 F32 F64 I64 I32 | F16 BF16 F32 F64 | F32 |
| `matmul_batched` | F16 BF16 F32 F64 | F16 BF16 F32 | F32 |
| `matmul_f8` | F16 BF16 F32 | F16 BF16 F32 | — |
| `matmul_int8` | F32 | F32 | — |
| `matmul_mixed` | F16 BF16 F32 | F16 BF16 F32 | — |
| `matmul_q` | Q6_K Q4_K Q5_0 Q4_0 Q8_0 | Q5_0 Q8_0 Q5_K Q4_0 Q4_K Q6_K | Q5_0 Q4_K Q6_K Q4_0 Q8_0 |
| `max` | F32 F64 I64 I32 | F32 F64 | F32 |
| `maximum` | F16 BF16 F32 F64 U8 I64 I8 I32 | BF16 F32 F64 I64 I32 | — |
| `maxpool2d` | F32 F64 | F32 | F32 |
| `mean` | F32 F64 | F32 F64 | F32 |
| `min` | F32 F64 I64 I32 | F32 F64 | F32 |
| `minimum` | F16 BF16 F32 F64 U8 I64 I32 I8 | BF16 F32 F64 I64 I32 | — |
| `mish` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `moe_gemm` | F16 BF16 F32 | F16 BF16 F32 | — |
| `moe_gemm_q` | — | Q6_K Q4_K Q5_0 Q4_0 Q8_0 Q5_K | — |
| `mul` | F16 BF16 F32 F64 U8 I64 I8 I32 | F16 BF16 F32 F64 | F32 |
| `ne` | F16 BF16 F32 F64 U8 U32 I64 I16 U16 U64 I32 I8 | BF16 F32 F64 U8 I64 I32 | — |
| `neg` | F16 BF16 F32 F64 U8 I64 I32 I8 | F16 BF16 F32 F64 | F32 |
| `pad` | F32 F64 U8 I64 I32 I8 | F32 | — |
| `pow` | F16 BF16 F32 F64 U8 I64 I32 I8 | F32 F64 | F32 |
| `powf` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `prod` | F32 F64 I64 I32 | F32 F64 | F32 |
| `recip` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `relu` | F16 BF16 F32 F64 | F16 BF16 F32 F64 | F32 |
| `rmsnorm` | BF16 F32 F64 | BF16 F32 | F32 |
| `roll` | F32 F64 U8 I64 I8 I32 | F32 | — |
| `rope` | BF16 F32 F64 | BF16 F32 | F32 |
| `rope_i` | BF16 F32 F64 | BF16 F32 F64 | F32 |
| `rope_scaled` | F32 F64 | BF16 F32 | F32 |
| `round` | F16 BF16 F32 F64 | F32 F64 | F32 |
| `scatter` | F16 BF16 F32 F64 I64 I32 | F16 BF16 F32 F64 I64 I32 | — |
| `scatter_add` | F16 BF16 F32 F64 I64 I32 | F16 BF16 F32 F64 I64 I32 | — |
| `sigmoid` | F16 BF16 F32 F64 | F16 BF16 F32 F64 | F32 |
| `sign` | F16 BF16 F32 F64 U8 I64 I32 I8 | BF16 F32 F64 I64 I32 | — |
| `silu` | F16 BF16 F32 F64 | F16 BF16 F32 F64 | F32 |
| `sin` | F16 BF16 F32 F64 | F32 F64 | F32 |
| `softmax` | BF16 F32 F64 | BF16 F32 F64 | F32 |
| `softplus` | F16 BF16 F32 F64 | F32 | — |
| `sort` | F32 F64 I64 I32 | F32 F64 I64 I32 | — |
| `sqr` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `sqrt` | F16 BF16 F32 F64 | F16 BF16 F32 F64 | F32 |
| `sub` | F16 BF16 F32 F64 U8 I64 I8 I32 | F16 BF16 F32 F64 | F32 |
| `sum` | F16 BF16 F32 F64 I64 I32 | F32 F64 | F32 |
| `tan` | F16 BF16 F32 F64 | F32 F64 | F32 |
| `tanh` | F16 BF16 F32 F64 | F16 BF16 F32 F64 | F32 |
| `topk` | F32 F64 I64 I32 | — | — |
| `tril` | F32 F64 | F32 | — |
| `triu` | F32 F64 | F32 | — |
| `upsample_bilinear2d` | F32 F64 | BF16 F32 | — |
| `upsample_nearest` | F32 F64 | BF16 F32 | — |
| `where` | F16 BF16 F32 F64 U8 I64 I8 I32 | BF16 F32 F64 U8 I64 I32 | — |

---

## 4. Light vs Candle

Reference: Hugging Face Candle 0.11.0 (crates.io 2026-06-26) plus `main` as of
2026-10-14 (surveyed 2026-10-27/28). Candle is still the HF Rust framework; it
has not been renamed or superseded. Light's side reflects 2026-11-08.

Legend: ✅ have · ⚠️ partial · ❌ missing · — not applicable.

### 4.1 Headline

| Area | Candle 0.11 | Light | Gap |
|------|-------------|-------|-----|
| Backends | CPU, CUDA (cuBLAS, optional cuDNN, NCCL multi-GPU, cuTile JIT), Metal, WASM-CPU. WGPU is an open PR, not merged | CPU (AVX2/FMA, multithreaded), CUDA (cuBLAS, cuBLASLt, cuDNN), WebGPU (merged, no cgo, F32 activations, LLMs incl. GPTQ / AWQ / GGUF) | Metal ❌; WebGPU is ours only. NCCL (multi-GPU collectives) is irrelevant on one GPU. cuTile is a Python/Rust tile DSL with no Go frontend: not applicable; our equivalent is NVRTC/nvcc → PTX |
| DTypes | U8 U32 I16 I32 I64 BF16 F16 F32 F64 F8E4M3 (+ storage-only F6/F4/F8E8M0) | F16 BF16 F32 F64 F8E4M3 U8 I8 I16 U16 U32 I32 U64 I64 (CPU); CUDA compute F32/F64 broad, BF16 + F16 GEMM/elementwise, F8E4M3 cast + GEMM | none: F8E4M3 landed 2026-10-28 with a cuBLASLt FP8 GEMM, which Candle lacks |
| Core ops | ~45 op families, all three backends | 113 registered ops (§3); the few without a CUDA kernel are lowered (einsum F64, lpnorm, topk) | small |
| Fused attention | FA2 + FA3 (CUDA), Metal SDPA, CPU flash-attn, paged | FlashAttention CUDA tiled (D≤256) + CPU; split-K decode attention over ring caches | ⚠️ no paged, no varlen |
| Linear attention / scans | (mamba, rwkv modules) | Gated delta rule (CPU + CUDA chunked), lightning attention | different model families covered |
| **Quantized inference** | GGUF Q4_0…Q8_K matmul on CPU/CUDA/Metal, MMVQ/MMQ fast kernels, quantized MoE, quantized embeddings | GGUF dequant for all 12 types; fused dequant-matmul Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K on CUDA (v1 kernels), AVX2 CPU kernels, WebGPU GEMV; quantized MoE and embeddings; GPTQ / AWQ 4-bit; FP8 per-tensor and blockwise; int8 per-column | ⚠️ CUDA kernel speed (MMVQ/MMQ); Light ahead on GPTQ / AWQ / FP8 |
| MoE | fused `moe_gemm` (f16/bf16/GGUF, CUDA) | ✅ `ops.MoEGemm` F32/BF16/F16 + `ops.MoEGemmQ` GGUF + `ops.MoEInt4`, `MoERoute` / `MoERouteFromIDs`, `nn.SparseMoE` with custom routers (DeepSeek V2 / V3) | WMMA tensor-core tiles are a perf item |
| Models | ~110 modules (see §4.4) | see §1 | large, by design |
| Audio | whisper, encodec, mimi, dac, snac, metavoice, parler_tts, csm | Whisper + Qwen3-ASR (STT), Mimi codec (streaming), Moshi speech-to-speech LM (real time on an L4) | TTS gap (parler/metavoice/csm) |
| Training | tape autograd, SGD (no momentum), AdamW, 5 losses; no schedulers; 2 examples | dynamic DAG autograd, SGD+momentum, AdamW, CosineLR, 10+ training examples, 90% CIFAR recipe | Light ahead (but LLM fine-tune blocked, see §4.6) |
| ONNX | candle-onnx, ~80 ops | ❌ | ecosystem |
| Tokenizers | `tokenizers` crate (everything) + GGUF-metadata tokenizer | BPE byte-level, BPE SentencePiece, WordPiece, Unigram, raw SentencePiece `.model`, CLIP, GGUF-metadata tokenizer (llama.cpp-exact), Jinja chat templates (byte-identical to transformers) | none |
| Product surface | examples | engine + OpenAI API + `light` CLI (separate desktop app `light-studio`) | Light-only |

### 4.2 Core ops, per family

| Family | Candle | Light | Notes |
|--------|--------|-------|-------|
| Unary: exp log sin cos tanh neg sqrt abs floor ceil round relu silu gelu gelu_erf sigmoid | ✅ | ✅ | |
| recip, sqr, sign, erf, elu, powf (scalar), clamp, round_to, affine | ✅ | ✅ (2026-10-28: fused CPU + CUDA kernels F32/F64/BF16; sign also I32/I64; `RoundTo` composed) | plus mish, hard_swish, hard_sigmoid, leaky_relu, softplus |
| Binary add sub mul div + broadcast | ✅ | ✅ | |
| maximum / minimum (elementwise) | ✅ | ✅ (2026-10-28, broadcast; F32/F64/BF16/I32/I64; autograd routes grad to the winner) | |
| Compare eq ne lt le gt ge → U8 | ✅ | ✅ (CUDA F32/F64/I32/I64/U8/BF16; where likewise) | widened 2026-10-28 |
| Reduce sum min max argmin argmax (+ mean var logsumexp norm) | ✅ | ✅ (var, logsumexp composed) | |
| where, index_select, gather, scatter_add | ✅ | ✅ | |
| scatter / scatter_set / index_add / slice_scatter / slice_assign | ✅ | ✅ (2026-10-28; gather/scatter/scatter_add/index_add on CPU + CUDA for F32/F64/I32/I64/BF16/F16) | |
| matmul, batched, mv, dot | ✅ | ✅ F32/F64/F16/BF16 2-D (plain + transposed-weight) and batched on CUDA via cuBLAS GemmEx; `MatMulMixed` F32×{BF16,F16}; `MatMulF8` E4M3 via cuBLASLt (K%16==0); AVX2 tiled CPU | Light ahead on FP8 |
| conv1d, conv2d (groups, dilation), conv_transpose1d/2d | ✅ | ✅ conv1d/2d/3d; conv_transpose1d/2d (stride, padding, output_padding, dilation, groups; CPU F32/F64, CUDA F32/BF16) | matches `F.conv_transpose2d` |
| avg/max pool2d, upsample_nearest 1d/2d, upsample_bilinear2d, interpolate | ✅ | ✅ pools + nearest 2d + bilinear 2d (align_corners both ways, up and down) | |
| cat, narrow, chunk, pad, flip, repeat, unfold, tril/triu/eye, meshgrid | ✅ | ✅ | |
| sort / argsort (last dim) | ✅ all backends | ✅ CPU + CUDA bitonic (F32/F64/I32/I64, stable tie order) | |
| cumsum | matmul trick | ✅ kernel | Light ahead |
| topk | ❌ (sampling only) | ✅ CPU + CUDA (sort + narrow) | Light ahead |
| einsum | ❌ | ✅ CPU; CUDA F32 fully on-device (diagonals, N-ary) | Light ahead |
| rope (rotate-half), rope_i (interleaved), rope_thd | ✅ ×3 | ✅ ×3 (+ scaled tables: llama3, longrope, yarn, linear; partial rotary) | |
| fused rms_norm / layer_norm | ✅ (CPU F16/BF16/F32, CUDA, Metal) | ✅ (CUDA F32/BF16; CPU F32/F64, BF16 via F32) | |
| softmax / log_softmax | ✅ | ✅ | |
| custom op API | CustomOp1/2/3 + inplace | registry key + kernel func | equivalent |

### 4.3 candle-nn

| Module | Candle | Light |
|--------|--------|-------|
| Linear, Conv1d/2d, Embedding, LayerNorm, RmsNorm, GroupNorm, BatchNorm, Dropout, Sequential | ✅ | ✅ |
| ConvTranspose1d/2d | ✅ | ✅ `nn.ConvTranspose2d` (inference) + `ops.ConvTranspose1d/2d` |
| LSTM, GRU | ✅ | ✅ `nn.LSTM` / `nn.GRU` (PyTorch weight layout, differentiable unroll; match torch to 1e-5 on CPU + CUDA) |
| PReLU, activation enum (14 variants) | ✅ | ✅ ReLU/ReLU6/SiLU/Sigmoid/Tanh/GELU/ELU/LeakyReLU/Mish/HardSwish/HardSigmoid/PReLU/SwiGLU modules, all with gradients |
| KV caches: KvCache, RotatingKvCache (sliding window), Scattered (batched fixed ctx), Concat, Interleaved | ✅ | ✅ rotating/sliding window in `models.KVCache` (per-layer window; concat-grown), ring caches for single-frame decode, latent caches (MLA), recurrent state slots; no Scattered/Interleaved |
| rotary_emb, sdpa (Metal), flash_attn (CPU), varlen | ✅ | ✅ RoPE, SDPA masks, FlashAttention (no varlen) |
| moe_gemm | ✅ | ✅ dense (`ops.MoEGemm`), GGUF (`ops.MoEGemmQ`, same six formats as `matmul_q`), 4-bit (`ops.MoEInt4`) |
| VarBuilder (safetensors mmap, sharded, npz, pth, varmap save/load) | ✅ | ✅ safetensors + sharded, `npy` (npy/npz read+write), `pth` (torch.save zip), `nn.VarMap` save/load (2026-10-29) |
| Optimizers: SGD, AdamW | ✅ | ✅ + momentum + CosineLR |
| Losses: nll, cross_entropy, mse, bce_with_logits, huber | ✅ | ✅ all five (autograd, grad-checked) |
| pixel_shuffle/unshuffle, replication_pad2d | ✅ | ✅ (`ops.PixelShuffle/PixelUnshuffle/ReplicationPad2d`) |

### 4.4 Models

Candle's `candle-transformers/src/models` (Oct 2026) by group; details per model in §1.

| Group | Light ✅ | Partial ⚠️ | Missing ❌ |
|-------|---------|------------|-----------|
| LLMs (dense) | llama, mistral, phi3 (+ Phi-4-mini), gemma / 2 / 3 / 4, qwen2, qwen3, deepseek2, falcon, glm4, granite | — | phi-1/2 (mixformer), gemma v1, chatglm, codegeex4, starcoder2, mpt, persimmon, stable_lm, olmo, helium, lfm2, smol, llama2_c |
| LLMs (MoE / recurrent) | qwen3_moe | mixtral, qwen2_moe (loaders not wired), granitemoehybrid (no Mamba layers) | mamba / mamba2, rwkv v5–v7, based, recurrent_gemma |
| GGUF | llama / qwen2 / qwen3 / mistral / qwen3moe, qwen35, muse-glimmer, moshi | — | the other Candle quantized modules |
| Encoders | bert, embedding_gemma2, t5 (encoder) | — | distilbert, jina / nomic / modern bert, deberta, xlm-roberta, stella, nvembed, colpali, marian |
| Vision | resnet, vgg, vit (+ mobilenetv2, yolov5) | — | dinov2, beit, eva2, convnext, efficientnet, mobileone, fastvit, hiera, repvgg, segformer, SAM, depth-anything, trocr |
| Multimodal | — | clip (SD text tower) | siglip, blip, llava, moondream, paligemma, pixtral, qwen3_vl, paddleocr_vl, voxtral |
| Diffusion | flux | stable_diffusion (1.5 only) | SD 2.1 / XL, mmdit (SD3), wuerstchen, z_image |
| Audio | whisper, mimi, moshi (+ qwen3_asr) | — | encodec, dac, snac, metavoice, parler_tts, csm |

Light-only: Qwen3.5, LFM3, Spark-X, Muse Glimmer, Mellum, Nemotron, MiMo (+ MTP), MiniMax-Text-01, SDAR, Qwen3-ASR, EmbeddingGemma 2, Ministral 3.

### 4.5 Quantization and precision

| | Candle | Light |
|--|--------|-------|
| GGUF read / write | ✅ / ✅ | ✅ / minimal |
| GGML dtypes | F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1 Q2K–Q8K (no IQ/TQ) | same set (no IQ/TQ) |
| Quantized matmul CPU | f32 + f16, NEON/AVX-512 repacked kernels | ✅ AVX2 row decoders + tiled SIMD matmul for Q4_0 / Q5_0 / Q8_0 / Q4_K / Q6_K (2026-11-08); dequant + matmul for the rest |
| Quantized matmul CUDA | dequant, MMVQ (gemv), MMQ (gemm) for Q4_0…Q8_0, Q2K–Q6K; bf16/f16/f32 activations | ⚠️ fused dequant-matmul for Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K (F32 activations), v1 kernel; others dequant+cuBLAS |
| GGUF model loader | 19 quantized model modules | ✅ `LoadLlamaGGUF` (llama / qwen2 / qwen3 / mistral / qwen3moe), `LoadQwen35GGUF`, `LoadMuseGlimmerGGUF`, `LoadMoshiGGUF` |
| Quantized embedding, quantized MoE | ✅ | ✅ both (GGUF table stays in block format; only gathered rows are dequantized; tied LM head reuses the quantized storage) |
| Tokenizer from GGUF metadata | ✅ | ✅ `tokenizer.LoadGGUF` (matches llama.cpp ids on its vocab fixtures) |
| safetensors int8 / fp8 / GPTQ / AWQ | ❌ | ✅ int8 per-column; `F8_E4M3` per-tensor and 128 × 128 blockwise scales with fused kernels; GPTQ (incl. act-order) and AWQ 4-bit incl. MoE experts (2026-11-08) |
| BF16 LLM path | ✅ | ✅ |
| F8E4M3 | dtype + cast | ✅ dtype + cast (CPU/CUDA, RNE + saturate) + FP8 GEMM (cuBLASLt, F32/BF16/F16 out) + `QuantizeF8` |

### 4.6 Training

Light is ahead on everything except breadth of backward ops. Candle: tape autograd; backward for conv1d/2d, conv_transpose2d, pools, upsample_nearest, index ops, cat/narrow, reductions; no backward for fused norms, rope, sdpa, flash-attn, quantized matmul, MoE; SGD without momentum; no schedulers; two training examples. Light: dynamic DAG, 30+ differentiable functions incl. fused LayerNorm/RMSNorm, BatchNorm2d train mode, AdamW, momentum SGD, CosineLR, ten training examples. The gaps that block LLM fine-tuning are listed in ROADMAP.md §5 (deferred until inference parity, user decision 2026-10-27).

### 4.7 What Light has that Candle lacks

Merged WebGPU backend (no cgo, LLMs incl. quantized) · einsum · cumsum/topk kernels · Conv3d · InstanceNorm · momentum SGD + LR scheduler + real training recipes · BatchNorm train-mode CUDA kernels · FP8 E4M3 GEMM on tensor cores (cuBLASLt) and blockwise FP8 · GPTQ / AWQ 4-bit · int8 per-column safetensors quant · Gated DeltaNet (Qwen3.5) · block-diffusion LM (SDAR) · Jinja chat templates · engine + OpenAI API + CLI · op×device×dtype matrix generator · verification harness vs transformers with a stated bar.

---

## 5. Performance snapshots

### RTX 5060 Ti 16 GB (reference box)

| Model | Variant | Params | Forward | Example |
|-------|---------|--------|---------|---------|
| ResNet-50 | 1×3×224×224 | 25.56 M | 150 ms | `examples/26_resnet50` |
| MobileNetV2 ×1.0 | 1×3×224×224 | 3.50 M | 800 ms | `examples/27_mobilenetv2` |
| VGG-16 | 1×3×224×224 | 138.36 M | 166 ms | `examples/28_vgg16` |
| ViT-B/16 | 1×3×224×224 | 86.57 M | 152 ms | `examples/29_vit_b16` |
| YOLOv5n | 1×3×640×640 | 1.87 M | 141 ms | `examples/30_yolov5n` |
| GPT-2 small | 1×16 (BPE) | 124.44 M | 137 ms | `examples/24_gpt2_forward`, `examples/25_gpt2_pretrained` |
| Qwen2-0.5B-Instruct | 1×26 prompt + decode | 494.03 M | 187 ms prefill + ~100 ms/tok | `examples/35_qwen_inference` (KV cache, chat template) |
| Qwen2.5-3B-Instruct (F32) | 1×31 prompt + decode | 3 085.94 M | 599 ms prefill + 24 ms/tok | `examples/35_qwen_inference -repo Qwen/Qwen2.5-3B-Instruct` (sharded safetensors, 12.7 GB GPU) |
| Qwen2.5-3B-Instruct (BF16) | 1×31 prompt + decode | 3 085.94 M | 707 ms prefill + 16 ms/tok | `examples/35_qwen_inference -repo Qwen/Qwen2.5-3B-Instruct -bf16` (6.9 GB GPU, tensor-core path) |
| Llama-3.2-1B-Instruct (F32) | prompt + decode | 1 235.81 M | ~29 tok/s | `examples/35_qwen_inference -repo meta-llama/Llama-3.2-1B-Instruct` |
| Llama-3.2-1B-Instruct (BF16) | prompt + decode | 1 235.81 M | ~34 tok/s | same with `-bf16` |
| bge-small-en-v1.5 (BERT) | 10 sentences | 33.4 M | — | `examples/40_bert_embeddings`; max abs err 7.8e-7 vs PyTorch |
| Stable Diffusion 1.5 (BF16) | 512×512, DDIM | ~1 B | — | `examples/37_sd15_txt2img` |
| GPT-2 small on WebGPU | 1×16 | 124.44 M | ~2× slower than cuBLAS | `examples/25_gpt2_pretrained -device webgpu` |
| Phi-3.5-mini-instruct (BF16) | 10-token chat + decode | 3 821.08 M | 164 ms prefill + 15 tok/s | `examples/35_qwen_inference -repo microsoft/Phi-3.5-mini-instruct -bf16` (7.6 GB GPU) |
| Qwen2-0.5B-Instruct **GGUF Q8_0 / Q4_K_M** | 15-token chat + decode | 494.03 M | 235 ms prefill + 17 tok/s | `examples/35_qwen_inference -gguf ~/.cache/light/gguf/qwen2-0.5b-instruct-q4_k_m.gguf` (0.4 GB + F32 embedding); logits within 0.05 (Q8_0) / 0.4 (Q4_K_M) of F32 |
| Llama-3.2-1B-Instruct **GGUF Q4_K_M** | 17-token chat + decode | 1 235.81 M | 501 ms prefill + 14 tok/s | `-gguf ~/.cache/light/gguf/llama-3.2-1b-instruct-q4_k_m.gguf -repo meta-llama/Llama-3.2-1B-Instruct`; same top-3 as F32, greedy identical |

### Laptop: Core Ultra 7 356H CPU and Intel integrated graphics (2026-11-08)

Qwen2.5-0.5B-Instruct, `light bench` (71-token prompt) unless noted.

| Path | Prefill tok/s | Decode tok/s | Notes |
|------|---------------|--------------|-------|
| CPU F32 | 166 | 22.2 | was 2 / 2.2 before the AVX2 + threading pass; 1922 MB working set |
| CPU Q4_K_M GGUF | 155 | 21.0 | 497 MB |
| CPU GPTQ-Int4 | 163 | 27.4 | 748 MB |
| CPU AWQ | 161 | 29.2 | |
| CPU TinyLlama-1.1B GPTQ act-order | 73 | 14.5 | |
| WebGPU F32 | 128 | 11.7 | decode after dispatch batching (32 tokens); was 9.6 |
| WebGPU Q4_K_M GGUF | 83 | 14.8–15.6 | decode after batching (32 tokens) |
| WebGPU GPTQ-Int4 | 118 | 17.8–19.9 | decode after batching (32 tokens) |

### The ten-model list (2026-11-09)

The user-requested list on all three backends. Laptop: CLI defaults (`w16`), "Why is the sky blue? One sentence.", greedy, decode tok/s on a loaded machine (rough; same reply on CPU and WebGPU for every model). CUDA: the desktop's `light bench` (RTX 5060 Ti 16 GB, BF16, 32 tokens; `CHECKPOINTS-desktop.md` D.3).

| Model | CPU | WebGPU (Intel iGPU) | CUDA BF16 prefill / decode |
|-------|-----|---------------------|----------------------------|
| Qwen3.5-0.8B | 8.9 | 7.6 | 211 / 60 |
| Qwen3.5-2B | 4.4 | 8.5 | 200 / 56 |
| Qwen2.5-Coder-3B-Instruct | 2.0 | 6.4 | 228 / 37 |
| gemma-3-1b-it | 8.6 | 12.9 | 229 / 59 |
| gemma-4-E2B-it | 3.5 | 7.4 (2–7, noisy) | 106 / 39 |
| Llama-3.2-3B-Instruct | 2.7 | 6.1 | 368 / 46 |
| DeepSeek-R1-Distill-Qwen-1.5B | 5.9 | 4.7 | 166 / 50 |
| Phi-4-mini-instruct | 2.2 | 2.8 | 215 / 35 |
| granite-4.0-micro | 2.1 | 4.0 | 258 / 25 |
| Ministral-3-3B-Instruct-2512 | 1.5 | 3.7 | 987 (567-token prompt) / 35 |

CUDA decode after the per-token overhead pass (2026-11-09, `desktop/cuda`: uninitialised op outputs, no repeat_kv, in-place kᵀ, device-side argmax / top-k sampling, allocation-pressure GC): Qwen3.5-0.8B 120, Qwen3.5-2B 86, Coder-3B 54, gemma-3-1b 93, gemma-4-E2B 57, Llama-3.2-3B 60, DeepSeek-R1-1.5B 87, Phi-4-mini 47, granite-4.0-micro 41, Ministral-3-3B 49 tok/s; Llama-3.2-1B 144.

GGUF decode on CUDA (2026-11-09, `desktop/cuda`: lane-cooperative M = 1 kernels for Q4_K / Q5_K / Q6_K, dequant + cuBLAS for prompts of ≥ 8 rows, prefill timed after a device sync): Qwen2-0.5B Q4_K_M 145 tok/s (was 49), Q8_0 153 (61), Llama-3.2-1B Q4_K_M 222 (14; its BF16 runs 147); prefill 1.1–1.8k tok/s.

### Colab L4 / A100

| Workload | Result |
|----------|--------|
| Moshi speech-to-speech (Moshiko Q8 + Mimi, L4) | 58 ms per 80 ms frame through the engine (1.4× real time); model loop 61.9 ms / frame (LM 34.2, depformer 10.7, Mimi 9.8 + 7.2) |
| Qwen3.5-9B BF16 heavy test (A100) | 32 s for load + prefill + 16 greedy steps |
| Qwen3.5-27B-FP8, DeepSeek-V2-Lite, GLM-4-9B (A100) | 47–48 s each for load + prefill + 16 decode steps |
| Qwen3.5-27B / Muse Glimmer 30B GGUF Q4 (A100) | 35–42 s each for load + prefill + 16 decode steps |
