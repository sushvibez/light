# GoLight ⇄ Candle — Parity, Coverage & Roadmap

> Single source of truth for GoLight's feature parity with HuggingFace Candle, the backend/dtype
> coverage matrix, and the live parity roadmap. (Merged from the former `candle-parity.md`,
> `feature-coverage.md`, and `parity-roadmap.md`.)
>
> - Candle: https://github.com/huggingface/candle (Rust ML framework by HuggingFace)
> - Light: `workspace/accelerate/light` (module `light`, Go 1.26; CPU + WebGPU + CUDA)
> - **Goal:** full parity with Candle across dtypes and devices (CPU, CUDA, WebGPU).
> - Last updated: 2026-06-18.

**Legend (per-backend columns):** ✅ native on-device kernel · 🔄 works via CPU round-trip (device-preserving fallback, no native kernel) · ⚠️ partial · ❌ not available.
**light-webgpu is f32-only** — WGSL cannot express f64/i64; a ✅ there means the f32 path.

---

## 1. Progress dashboard

| Phase | Scope | Weight | Progress |
|-------|-------|:------:|:--------:|
| **1. Dtype + tensor ops** | INT8 (inference), f16, bf16, missing tensor ops | 25% | **90%** 🟢 |
| **2. NN modules** | grouped/depthwise conv, ConvTranspose, GroupNorm, SDPA/flash, SWA, ALiBi… | 15% | **100%** ✅ |
| **3. Model architectures** | Mistral, Gemma, T5, BERT, Whisper, Mamba, SD/VAE… | 25% | **45%** 🟡 |
| **4. Backends** | cuDNN, Metal, MKL | 15% | 0% ⬜ |
| **5. Quantization** | Q4, Q8, GGUF | 12% | 0% ⬜ |
| **6. Ecosystem** | HF Datasets, HF Hub download | 8% | 0% ⬜ |

### **Overall parity-gap closure: ~44%**

> Weights are relative effort toward closing the Candle gap (the codebase already has a large parity
> baseline — this tracks the *remaining* enumerated work). ⬜ not started · 🟡 in progress · ✅ done.

---

## 2. Roadmap

Agreed order: **dtype + tensor ops → NN modules → model archs → cuDNN/Metal/MKL → Q4/Q8/GGUF**, plus HF Datasets/Hub.

### Phase 1 — Dtype + tensor ops (25% weight, ~90% done)

**1a. Missing activations — ✅ DONE (verified on CPU + WebGPU + CUDA hardware).**
ELU, LeakyReLU (scalar opcodes 4/5); Mish, HardSigmoid, HardSwish (unary opcodes 11–13).
Wired across `storage/ops.go`, `tensor/ops.go`, `nn/activation.go`, `gpu/shaders/{unary,scalar}.wgsl`, `cuda/kernels.cu`. Verified on RTX 5060 Ti: CPU tests + `TestGPUParityActivations` + `TestCUDAParityActivations` (`-tags cuda`).

**1b. Missing tensor ops — 🟡 nearly done (only runtime dtype cast remains).**
- [x] LogSoftmax, SoftmaxDim (softmax along any dim)
- [x] Max/MinDim, ArgMaxDim/ArgMinDim, ArgMin (global)
- [x] General N-D `Permute`
- [x] Chunk, Split, Pad, Flip, Roll, Cumsum, TopK, Sort, Lp `Norm`
- [x] Full NumPy broadcasting + public `BroadcastTo`/`Expand` (fixed a real CUDA `k_binop_t` broadcast bug)
- Impl: `tensor/ops_shape.go`; tests: `tensor/ops_shape_test.go`.
- [ ] **Runtime dtype cast (`tensor.To(dtype)`)** — cross-dtype, needs a non-generic bridge.

**1d/1e. Native f16 + bf16 compute — 🟡 in progress.**
Design (chosen): **device-native half** — public compute API stays f32; tensors gain a 16-bit *device* representation; CUDA/WebGPU kernels run natively in f16/bf16 (via the existing `GotorchDtype` dispatch + `__half`/`__nv_bfloat16`); convert at host boundaries. CPU promotes to f32. No change to `dtype.Numeric`; no broken operators. Targets fitting 7B+ models in 16GB VRAM + faster GPU inference.
- [x] **Increment 1 — Go dtype foundation:** `dtype.F16` value type (mirrors `BF16`), `FloatToHalf` (round-to-nearest-even, exact on known values), `F16SliceToFloat32`/`Float32SliceToF16`. Tests in `dtype/half_f16_test.go` (pass). (`BF16` already existed.)
- [x] **Increment 2 — CUDA native half matmul:** added `GOTORCH_F16`/`GOTORCH_BF16`; `cuda_matmul_half` + `cuda_batched_matmul_half` via `cublasGemmEx`/`cublasGemmStridedBatchedEx` (16-bit storage, **f32 accumulate**); Go `HalfMatMulF32` entry point. **Verified on RTX 5060 Ti** (`cuda/half_matmul_test.go`): f16 and bf16 match f32 CPU reference within tolerance. (Elementwise half kernels still TODO.)
- [x] **Increment 3 — device-buffer integration:** `cuda.HalfBuffer` implements `storage.TypedBuffer[float32]` (2 bytes/elem on device, f32 host API). Optional `storage.HalfConverter` interface (no WebGPU churn); CUDA `MatMul`/`BatchedMatMul` detect half operands (incl. mixed f32×half → promote) → native kernel; public `tensor.ToBF16`/`ToF16`; `Linear.ToHalf` + `Llama.Half`. **Verified on RTX 5060 Ti**: `tensor/half_cuda_test.go` + `models/llama_half_cuda_test.go` (Llama weights → bf16, projection rel-err-norm 0.0065, weight VRAM halved). Halves weight VRAM (n×2 vs n×4 bytes).
- [x] **Increment 3b — half elementwise on CUDA:** native `__half`/`__nv_bfloat16` kernels for binop (add/sub/mul/div), unary (all activations), and scalar ops (read 16-bit → compute f32 → write 16-bit); `allocTyped` returns `HalfBuffer` for half dtypes. Half tensors now support matmul + elementwise + unary + scalar. Verified (`tensor/half_cuda_test.go TestHalfElementwise`).
- [x] **Increment 4 — WebGPU f16 (minimal-viable): implemented, runtime-blocked by binding.** Full implementation landed: `GPUHalfBuffer` (f16 storage + f32 host API via staging readback), f16 WGSL matmul+elementwise shaders (`enable f16`), `WebGPUBackend.ToHalf` + BinOp/MatMul f16 routing, generalized `tensor.toHalf` (device-agnostic), and a `SupportsHalf(dt)` capability check. **Blocker:** the bundled wgpu-native/naga (cogentcore v0.23) **cannot parse `enable f16;`** even though the adapter advertises the shader-f16 feature — a device probe detects this (`HasF16=false`) so `ToF16` degrades gracefully (falls to CUDA / clear error) and `TestWebGPUF16` skips. **Will activate automatically with the wgpu-native v29 fork** (newer naga). WebGPU is f16-only (no bf16 in WGSL) and still subject to the 256 MB/buffer cap.
- [x] **Half norm/softmax CUDA kernels** — native `k_rmsnorm_half`/`k_layernorm_half`/`k_softmax_half` (half x + f32 weight → f32 compute → half out) + plugin routing. Verified (`cuda/half_norm_test.go`). **Not yet reached by the model forward** — `nn/norm.go` only has WebGPU+CPU paths (no CUDA branch), so wiring half norms into the Llama forward still needs: a CUDA branch in `nn.RMSNorm`/`LayerNorm.Forward`, gate-flips for Softmax/Cat/Permute4D to allow half, and RoPE mixed half×f32 handling.
- [x] **Full bf16 Llama forward (on-device, verified)** — half kernels for movement (permute4d/cat/transpose/repeat/gather/copy/expand as 16-bit copy), softmax, norms; mixed bf16×f32 elementwise auto-coerces in `CUDABackend.BinOp` (RoPE/residuals); 2D Contiguous + dispatch gates accept half. A small Llama runs end-to-end in bf16 on CUDA, logits match f32 at **rel-err-norm 0.0049**, and intermediates verified to **stay CUDA+bf16 (no CPU round-trips)**. Tests: `models/llama_bf16_forward_cuda_test.go`.

**1c. INT8 (weight-only, inference) — ✅ DONE (verified end-to-end).**
- [x] `dtype.Int8`; `dtype.QuantizeInt8PerColumn` (symmetric per-output-channel: `scale[n]=max|w[:,n]|/127`); CUDA `cuda_matmul_int8` (int8 weight × f32 activation, f32 accumulate, per-channel dequant). Verified vs f32 at rel-err 0.0051.
- [x] Integration: `cuda.Int8Buffer` (int8 data + per-channel scales, f32 host API) implementing `storage.TypedBuffer[float32]`; `storage.Int8Quantizer` interface; CUDA `MatMul` detects int8 weights; `tensor.ToInt8`, `Linear.ToInt8`, `Llama.QuantizeInt8`.
- [x] **End-to-end verified on RTX 5060 Ti** (`models/llama_int8_forward_cuda_test.go`): full Llama forward with int8 weights matches f32 at **rel-err-norm 0.0086**. ~4× weight VRAM reduction.

### Phase 2 — NN modules (15% weight, ✅ complete)
- [x] **LLM attention variants** (`nn/attention.go`): reusable device-agnostic **`SDPA`** building block (softmax(QKᵀ/√d + addMask)@V; CPU/WebGPU/CUDA, incl. f16/bf16); **Sliding Window Attention** mask (`SlidingWindowMask`, Mistral); **ALiBi** (`ALiBiSlopes`/`ALiBiBias`, Falcon). Tests in `nn/attention_variants_test.go` + CUDA parity. Each variant = an additive mask on SDPA.
- [x] **SWA wired into the model**: `LlamaConfig.SlidingWindow` + windowed `causalMask` + `Mistral7B()` config — Mistral sliding-window attention works end-to-end (verified: window changes logits). Bridges into Phase 3 (Mistral).
- [x] **Conv modules**: grouped/**depthwise Conv2d** (`nn.Conv2d` + `WithGroups`; MobileNet/EfficientNet), **ConvTranspose2d** (`nn.ConvTranspose2d`; UNet/VAE decoder), **GroupNorm** (`nn.GroupNorm`; UNet/ResNet/diffusion). Device-agnostic (backend kernel for groups=1 conv; CPU for grouped/transpose/groupnorm). Tests verify depthwise scaling, GroupNorm stats, ConvT upsample shape. **BatchNorm2d** already correct (per-channel, from the norm refactor).
- [x] **FlashAttention** — fused memory-efficient attention (online softmax, **no materialized S×S scores**, O(S·D) memory). CUDA kernel `cuda_flash_attention` (causal + non-causal, headDim≤256) via `storage.FlashAttentioner`; device-agnostic `tensor.FlashAttention` with a CPU online-softmax fallback. Verified vs reference softmax attention on CPU + CUDA (`tensor/flash_attn_test.go` + `_cuda_test.go`). (Simple one-row-per-thread kernel — correct + memory-efficient; shared-mem tiling is a perf follow-up.)
- [x] **Conv1d** (`nn.Conv1d`), **Conv3d** (`nn.Conv3d`), **InstanceNorm2d** (= GroupNorm w/ groups=C), **Upsample** (nearest), **Dropout2d**, **BatchNorm train-mode** — device-resident (reduction + broadcast, no CPU round-trip). Tests in `nn/conv_test.go` + `nn/norm_test.go`.
- [x] **Native CUDA kernels** for **GroupNorm**, **grouped/depthwise Conv2d**, **ConvTranspose2d** (`cuda_groupnorm`/`cuda_conv2d_grouped`/`cuda_conv_transpose2d` via optional `storage.GroupNormer`/`GroupedConv2Der`/`ConvTranspose2Der`) — these now run **on-device, no CPU round-trip**; verified vs CPU (`tensor/conv_norm_cuda_test.go`).
- [x] **FlashAttention shared-mem tiling** — `k_flash_attn_tiled` loads K/V tiles into shared memory (each key read once per block, not per query), online softmax; host picks tileK to fit shared mem, falls back to the simple kernel otherwise. Verified vs reference.

### Phase 3 — Model architectures (45%, 0%) 🟡
✅ Mistral (SWA), Falcon (ALiBi/MQA), VibeThinker-3B, **Gemma 2** (sandwich norm + GeGLU + attn/final logit soft-cap + alternating SWA), **T5** (encoder-decoder, relative position bias, gated-GELU, cross-attn), **BERT / RoBERTa** (encoder-only, learned abs+type embeds, post-LN, pooler + MLM head), **Whisper** (audio encoder-decoder, Conv1d mel stem, sinusoidal pos, cross-attn). ⬜ Mamba (SSM), EnCodec, diffusion (VAE + schedulers + cross-attn + CFG), EfficientNet/MobileNet/ConvNeXt/Swin/SAM/YOLOv8, and **TRELLIS.2** (see §10). Tokenizers: **Unigram** (Gemma/T5), **WordPiece** (BERT), BPE (RoBERTa/LLMs/Whisper).

### Phase 4 — Backends (15%, 0%) ⬜
cuDNN (conv/attention accel), Metal (Apple Silicon), MKL (Intel CPU BLAS).

### Phase 5 — Quantization (12%, 0%) ⬜
Q8 (Q8_0), Q4 (Q4_K), GGUF loader (llama.cpp compat).

### Phase 6 — Ecosystem (8%, 0%) ⬜
HF Datasets integration, HF Hub auto-download.

---

## 3. Backend × dtype headline

- **Op coverage is near-identical across WebGPU and CUDA** — every canonical compute op has both a WGSL shader and a CUDA kernel. The divergence is **dtype**, not op count.
- **WebGPU = f32 only.** WGSL expresses only `f32`/`u32`; `f64`/`i64` are not representable. `f16` is possible via the `shader-f16` extension (not yet used).
- **CUDA = f32/f64/i32/i64** for generic (elementwise/reduce/movement) ops; f32-only for float-math (matmul, norms, attention, conv, pools).
- **CPU = all dtypes, all ops** (reference).

### Dtype support per device

| Dtype | CPU | CUDA | WebGPU | Note |
|-------|:---:|:----:|:------:|------|
| f32 | ✅ | ✅ | ✅ | |
| f64 | ✅ | ✅ | ❌ | not representable in WGSL |
| f16 | promote→f32 | promote→f32 | via `shader-f16` (planned) | |
| bf16 | promote→f32 | promote→f32 | emulate u16 (planned) | |
| i32 / u32 | ✅ | ✅ | ✅ achievable | WGSL-native |
| i64 | ✅ | ✅ | ❌ | not representable in WGSL |
| int8 | planned (1c) | planned (1c) | pack into i32 (planned) | inference quant |

---

## 4. Tensor operations (per-backend)

| Operation | Candle | light-cpu | light-webgpu | light-cuda | Notes |
|-----------|:------:|:---------:|:------------:|:----------:|-------|
| Add / Sub / Mul / Div | ✅ | ✅ | ✅ | ✅ | native kernels; CUDA multi-dtype |
| Scalar add / mul / pow / clamp | ✅ | ✅ | ✅ | ✅ | |
| Neg / Abs | ✅ | ✅ | ✅ | ✅ | |
| Exp / Log / Sqrt | ✅ | ✅ | ✅ | ✅ | |
| ReLU / Sigmoid / Tanh / GELU / SiLU | ✅ | ✅ | ✅ | ✅ | unary kernels |
| ELU / LeakyReLU | ✅ | ✅ | ✅ | ✅ | added 2026-06-15 |
| Mish / HardSigmoid / HardSwish | ✅ | ✅ | ✅ | ✅ | added 2026-06-15 |
| Sum / Mean / Max / Min (global) | ✅ | ✅ | ✅ | ✅ | reduce kernels |
| ArgMax / ArgMin (global) | ✅ | ✅ | 🔄 | 🔄 | CPU compute |
| Sum / Mean along dim | ✅ | ✅ | ✅ | ✅ | reducedim kernels |
| Var along dim | ✅ | ✅ | 🔄 | 🔄 | |
| Max / Min along dim | ✅ | ✅ | 🔄 | 🔄 | `MaxDim`/`MinDim`, 2026-06-15 |
| ArgMax / ArgMin along dim | ✅ | ✅ | 🔄 | 🔄 | `ArgMaxDim`/`ArgMinDim`, 2026-06-15 |
| MatMul (2D + batched ND) | ✅ | ✅ | ✅ | ✅ | cuBLAS (f32) |
| Softmax (last dim) | ✅ | ✅ | ✅ | ✅ | |
| Softmax along any dim | ✅ | ✅ | 🔄 | 🔄 | `SoftmaxDim`, 2026-06-15 |
| LogSoftmax | ✅ | ✅ | 🔄 | 🔄 | 2026-06-15 |
| Reshape / View / Contiguous | ✅ | ✅ | ✅ | ✅ | metadata / device copy |
| Squeeze / Unsqueeze | ✅ | ✅ | ✅ | ✅ | |
| Transpose (2D) | ✅ | ✅ | ✅ | ✅ | transpose kernel |
| Permute4D | ✅ | ✅ | ✅ | ✅ | native kernel |
| Permute (arbitrary N-D) | ✅ | ✅ | 🔄 | 🔄 | `Permute`, 2026-06-15 |
| Cat / Stack | ✅ | ✅ | ✅ | ✅ | copy kernel |
| Repeat / Expand / BroadcastTo | ✅ | ✅ | ✅ | ✅ | repeat kernel; `BroadcastTo`/`Expand` 2026-06-16 |
| Broadcasting (in binops) | ✅ | ✅ | ✅ | ✅ | CUDA pre-expand bug fixed 2026-06-16 |
| Narrow / Slice / Select | ✅ | ✅ | 🔄 | 🔄 | |
| IndexSelect / Gather | ✅ | ✅ | ✅ | ✅ | indexselect kernel |
| Scatter / ScatterAdd | ✅ | ✅ | 🔄 | 🔄 | |
| MaskedFill / MaskedSelect | ✅ | ✅ | 🔄 | 🔄 | |
| NonZero / Where | ✅ | ✅ | 🔄 | 🔄 | |
| Einsum | ✅ | ✅ | 🔄 | 🔄 | Light has full einsum |
| Triu / Tril / Eye | ✅ | ✅ | 🔄 | 🔄 | |
| Arange / Linspace | ✅ | ✅ | ✅ | ✅ | |
| Zeros / Ones / Full / Rand | ✅ | ✅ | ✅ | ✅ | |
| Chunk / Split | ✅ | ✅ | 🔄 | 🔄 | 2026-06-15 |
| Flip / Roll / Pad | ✅ | ✅ | 🔄 | 🔄 | 2026-06-15 |
| Cumsum / TopK / Sort | ✅ | ✅ | 🔄 | 🔄 | 2026-06-15 |
| Norm (L1 / L2 / Lp) | ✅ | ✅ | 🔄 | 🔄 | `Norm(x, p)`, 2026-06-15 |
| In-place ops (Add/Mul/Fill) | ⚠️ | ✅ | ✅ | ✅ | Light has explicit in-place GPU kernels |
| Interpolate (upsample) | ✅ | ❌ | ❌ | ❌ | |
| Dtype casting (`to_dtype`) | ✅ | ⚠️ | ⚠️ | ⚠️ | load-time f16/bf16→f32 only; no runtime cast (1b TODO) |
| Device transfer | ✅ | ✅ | ✅ | ✅ | `.CPU()` / `.GPU()` / `.CUDA()` |
| Pretty print | ✅ | ✅ | ✅ | ✅ | |

> 🔄 ops are correct on GPU/CUDA tensors but currently compute on the host (device-preserving readback). Promoting hot ones (e.g. `Permute`, `SoftmaxDim`) to native kernels is future work.

---

## 5. NN modules (per-backend)

| Module | Candle | light-cpu | light-webgpu | light-cuda | Notes |
|--------|:------:|:---------:|:------------:|:----------:|-------|
| Linear | ✅ | ✅ | ✅ | ✅ | matmul + bias |
| Embedding | ✅ | ✅ | ✅ | ✅ | index/gather |
| Conv2d (groups=1) | ✅ | ✅ | ✅ | ✅ | conv2d kernel |
| Conv2d (depthwise / grouped) | ✅ | ✅ | 🔄 | ✅ | `nn.Conv2d`+`WithGroups`; native CUDA kernel |
| ConvTranspose2d | ✅ | ✅ | 🔄 | ✅ | `nn.ConvTranspose2d`; native CUDA kernel |
| Conv1d / Conv3d | ✅ | ✅ | 🔄 | 🔄 | `nn.Conv1d`/`nn.Conv3d` (CPU) |
| MaxPool2d / AvgPool2d / AdaptiveAvgPool2d | ✅ | ✅ | ✅ | ✅ | |
| MaxPool1d | ✅ | ❌ | ❌ | ❌ | |
| LayerNorm / RMSNorm | ✅ | ✅ | ✅ | ✅ | |
| BatchNorm1d | ✅ | ⚠️ | ⚠️ | ⚠️ | eval mode only (no running-stats update) |
| GroupNorm | ✅ | ✅ | 🔄 | ✅ | `nn.GroupNorm`; native CUDA kernel |
| BatchNorm2d | ✅ | ✅ | 🔄 | 🔄 | per-channel (norm refactor) |
| InstanceNorm | ✅ | ✅ | 🔄 | 🔄 | `nn.InstanceNorm2d` (= GroupNorm, groups=C) |
| Activations (ReLU/Sigmoid/Tanh/GELU/SiLU/…) | ✅ | ✅ | ✅ | ✅ | see §4 |
| Dropout | ✅ | ✅ | ✅ | ✅ | eval passthrough |
| Dropout2d | ✅ | ❌ | ❌ | ❌ | |
| MultiHeadAttention | ✅ | ✅ | ✅ | ✅ | fused AttnScores + softmax |
| FlashAttention / true SDPA | ✅ | ✅ | 🔄 | ✅ | `tensor.FlashAttention` (CUDA fused; CPU fallback) |
| RoPE | ✅ | ✅ | ✅ | ✅ | RoPECache has GPU path |
| GQA (Grouped Query Attention) | ✅ | ✅ | ✅ | ✅ | |
| Sliding Window Attention | ✅ | ✅ | ✅ | ✅ | `SlidingWindowMask` + `LlamaConfig.SlidingWindow` |
| ALiBi positional bias | ✅ | ✅ | ✅ | ✅ | `ALiBiSlopes`/`ALiBiBias` (Falcon) |
| TransformerBlock / Sequential / Flatten | ✅ | ✅ | ✅ | ✅ | |
| Upsample (nearest/bilinear) | ✅ | ❌ | ❌ | ❌ | |
| Squeeze-and-Excitation / FPN / LSTM / GRU | ❌ | ❌ | ❌ | ❌ | neither framework |

---

## 6. Model architectures

| Model | Candle | Light | Notes |
|-------|:------:|:-------:|-------|
| **LLMs** | | | |
| GPT-2 (+ fine-tuning) | ✅ (infer) | ✅ | Light adds `GPT2Trainer` |
| Llama 3.x (1B/3B/8B) | ✅ | ✅ | GQA, RoPE, SwiGLU |
| Qwen2.5 (incl. Coder, Instruct) | ✅ | ✅ | `Qwen25_*` configs; Instruct/Coder = same arch, different weights |
| VibeThinker-3B | ✅ | ✅ | `VibeThinker3B()` config (Qwen2.5-Coder-3B arch); load weights to run ([arxiv](https://arxiv.org/abs/2606.16140)) |
| Phi-3 / 3.5 | ✅ | ⚠️ | partial RoPE factor missing |
| Mistral 7B (SWA) | ✅ | ✅ | `Mistral7B()` + SWA, end-to-end forward verified; SentencePiece needed only for official weights |
| Gemma / Gemma 2 | ✅ | ✅ | `Gemma2_2B/9B` (sandwich norm, GeGLU, logit soft-cap, alternating SWA); Unigram tokenizer supported |
| Falcon | ✅ | ✅ | `models.Falcon` — parallel attn+MLP block, MQA, ALiBi (Falcon7B config); forward verified |
| T5 (encoder-decoder) | ✅ | ✅ | `T5Small/Base`: relative position bias (bucketed), gated-GELU, cross-attn, tied embeddings; Unigram tokenizer |
| BERT / RoBERTa (encoder-only) | ✅ | ✅ | `BERTBase/Large`, `RoBERTaBase`, `BERTForMaskedLM`: learned abs+segment embeds, post-LN, GELU FFN, pooler, tied MLM head; WordPiece tokenizer |
| Mamba (SSM) | ✅ | ❌ | |
| Quantized GGUF | ✅ | ❌ | Phase 5 |
| **Vision** | | | |
| ResNet / AlexNet / ViT / LeNet | ✅ | ✅ | |
| YOLO v3-tiny | ❌ | ✅ | Light unique |
| EfficientNet / MobileNet / ConvNeXt / Swin / SAM / YOLOv8 | ✅ | ❌ | (depthwise conv now available for MobileNet/EfficientNet) |
| PP-OCRv6 (OCR pipeline) | ❌ | ❌ (target) | PaddleOCR, rel. 2026-06-11: PPLCNetV4 CNN backbone + LightSVTR neck (1×7 **depthwise conv** ✓ + transformer) + CTC head. Needs SVTR block + CTC decode + Paddle→safetensors weights ([docs](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)) |
| **Multimodal** | | | |
| CLIP | ✅ | ✅ | |
| LLaVA / moondream2 / PaliGemma / Florence-2 | ✅ | ❌ | |
| **Diffusion / Generative** | | | |
| SD 1.5 (UNet) | ✅ | ⚠️ | UNet arch only; missing VAE/scheduler/CFG/cross-attn |
| SD 2.1 / SDXL / Flux / schedulers / VAE / pipelines | ✅ | ❌ | Phase 3 |
| DCGAN / GAN | ❌ | ✅ | Light unique |
| **Audio** | | | |
| Whisper (audio enc-dec) | ✅ | ✅ | `WhisperTiny/Base`: Conv1d mel stem, sinusoidal enc pos, learned dec pos, cross-attn, tied head (mel front-end external) |
| EnCodec / TTS | ✅ | ❌ | |

---

## 7. Backends, training, precision

### Backends & hardware

| Backend | Candle | Light |
|---------|:------:|:-------:|
| CPU (f32 / f64 / i32 / i64) | ✅ | ✅ |
| CPU (native f16 / bf16 compute) | ✅ | ❌ (promotes to f32) |
| CUDA (NVIDIA) | ✅ | ✅ (custom CGo kernels + cuBLAS) |
| WebGPU | ❌ | ✅ (Light unique) |
| Metal (Apple) | ✅ | ❌ (Phase 4) |
| cuDNN | ✅ | ❌ (Phase 4) |
| MKL / Accelerate | ✅ | ❌ (Phase 4) |
| Multi-GPU / Distributed | ❌ | ❌ |

### Training (Light's biggest edge)

Light has a **full training stack** vs Candle's inference-first design: reverse-mode autograd (`CheckGrad`), losses (CE/MSE/BCE/BCEWithLogits), optimizers (SGD+momentum, Adam, AdamW, RMSProp), LR schedulers (StepLR, CosineAnnealingLR, LinearWarmup), gradient clipping, `GPT2Trainer`. Proven: FashionMNIST >85%, GPT-2 fine-tuning.

### Quantization & precision

| Feature | Candle | Light |
|---------|:------:|:-------:|
| F32 / F64 compute | ✅ | ✅ |
| F16 / BF16 compute | ✅ | ❌ (Phase 1d/1e) |
| F16 / BF16 storage (safetensors) | ✅ | ✅ |
| INT8 quant | ✅ | ❌ (Phase 1c) |
| INT4 / Q4K / Q8_0 (GGUF) | ✅ | ❌ (Phase 5) |
| INT32 / INT64 tensors | ✅ | ✅ |

### Tokenizers & loading

| Feature | Candle | Light |
|---------|:------:|:-------:|
| BPE / ByteLevel BPE (tokenizer.json) | ✅ | ✅ |
| SentencePiece Unigram (tokenizer.json) | ✅ | ✅ (Viterbi segmentation; T5/Gemma/ALBERT) |
| SentencePiece (`.model` protobuf) | ✅ | ⚠️ (use the HF `tokenizer.json` form) |
| WordPiece (tokenizer.json) | ✅ | ✅ (greedy longest-match, ## prefix, BasicTokenizer split; BERT) |
| Tiktoken | ✅ | ❌ |
| top-k / top-p sampling | ✅ | ✅ |
| Repetition penalty / chat templates | ✅ | ❌ |
| Safetensors (single / sharded / f16-bf16) | ✅ | ✅ |
| PyTorch `.bin` / GGUF | ✅ | ❌ |
| HF Hub auto-download / HF Datasets | ✅ | ❌ (Phase 6) |
| DataLoader + transforms (Mixup/CutMix/…) | ❌ | ✅ (Light unique) |

---

## 8. Key gaps (prioritized)

**Critical (blocks real models):** native F16/BF16 compute (1d/1e) · INT4/INT8/GGUF quant (1c, 5) · VAE enc/dec (3) · diffusion schedulers (3). (SentencePiece Unigram tokenizer ✅.)

**High value:** FlashAttention/SDPA · Sliding Window Attention (Mistral) · Mamba · Metal backend (4) · grouped/depthwise Conv2d · ConvTranspose2d · GroupNorm/BatchNorm2d (Whisper/BERT/T5 ✅).

**Recently closed (this effort):** ELU/Mish/LeakyReLU/HardSigmoid/HardSwish · N-D Permute · LogSoftmax/SoftmaxDim · Max/MinDim + ArgMaxDim/ArgMinDim · Chunk/Split/Pad/Flip/Roll/Cumsum/TopK/Sort/Norm · full broadcasting + `BroadcastTo`/`Expand` (+ CUDA binop broadcast bugfix).

**Remaining lower-priority:** runtime dtype cast (`to_dtype`) · repetition penalty · Tiktoken · PyTorch `.bin` loading · Interpolate/Upsample.

---

## 9. Things Light has that Candle lacks

Full training stack · WebGPU backend · custom CUDA-via-CGo kernels · explicit in-place GPU ops · rich DataLoader + image transforms (Mixup/CutMix/RandomErasing) · YOLO v3-tiny · GPT-2 fine-tuning · DCGAN/GAN · property + fuzz tests · benchmark regression tracking (`cmd/benchcmp`) · numerical gradient checking (`CheckGrad`) · full Einsum · GPU memory pool (`PoolGet/PoolPut`) · fused kernels (LayerNorm+Linear, causal-masked softmax).

---

## 10. TRELLIS.2 readiness

**TRELLIS.2** (Microsoft, open-sourced Dec 2025): 4B-param image-to-3D. Sparse **O-Voxel** structured latents, **Sparse 3D VAE** (16× downsampling), rectified-flow / diffusion transformer, PBR material gen, mesh extraction at 512³–1536³.

**Not a config-level "add a model" task** — gated on primitives Light lacks:

| Requirement | Present? | Blocked on |
|-------------|:--------:|------------|
| Sparse 3D conv / sparse tensors | ❌ | no sparse ops; no Conv3d |
| VAE encoder + decoder | ❌ | Phase 3 |
| Rectified-flow / diffusion scheduler | ❌ | Phase 3 |
| ConvTranspose / upsampling decoder | ❌ | Phase 2 |
| GroupNorm | ✅ `nn.GroupNorm` | done (Phase 2) |
| Cross-attention image conditioning (DINOv2-style) | ❌ | Phase 3 |
| Native f16/bf16 (4B params ≈ 16GB in f32) | ❌ | Phase 1d/1e |
| Voxel grid / mesh extraction (marching cubes) | ❌ | new 3D layer |

Tracked as a Phase 3 **target**, not a supported model. Sources: [project page](https://microsoft.github.io/TRELLIS.2/) · [GitHub](https://github.com/microsoft/TRELLIS.2).

---

## 11. Parity target & the WGSL wall

Full dtype parity on **every** device is impossible as literally stated: WGSL cannot express f64/i64, so the WebGPU device can never match Candle's full dtype set. Realistic per-device target:
- **CPU / CUDA:** full Candle dtype set (add native f16/bf16 + i32/i64 on float-math ops + quantization).
- **WebGPU:** f32 + f16(ext) + i32/u32, bf16 emulated; f64/i64 documented as a known divergence (Candle has no WebGPU backend to compare against).

---

## 12. Summary

| Category | Candle | Light |
|----------|:------:|:-------:|
| Tensor ops completeness | ~90% | ~80% (was ~65%) |
| NN modules completeness | ~80% | ~55% |
| Model variety | ~50+ | ~12 |
| Training | minimal | full |
| Quantization | INT4/INT8/GGUF | none (Phase 1c/5) |
| Native half-precision compute | ✅ | ❌ (Phase 1d/1e) |
| Backends | CPU, CUDA, Metal, Accel | CPU, CUDA, WebGPU |
| Tokenizers | BPE, SP, Tiktoken, WP | BPE + SP-Unigram + WordPiece |
| Data loading | bring-your-own | full DataLoader + transforms |
| Ecosystem maturity | production-ready | active development |

---

## Testing: op × dtype × device matrix (done)
`tensor/dtype_matrix_test.go` (+ `_half_cuda_test.go`) is a table-driven harness covering core ops (Add/Sub/Mul/Div/Neg/Abs/scalar/Sum/Max/Min/Transpose/Permute, plus Sqrt/Exp/ReLU/MatMul) across:
- **dtypes:** f32, f64, i32, i64 (closes the old f64=3/i64=0 gap), plus **bf16/f16** on CUDA.
- **devices:** CPU (all dtypes) + WebGPU/CUDA for f32 (device result checked against CPU reference); half-matrix asserts results **stay on CUDA** (no round-trip).

**WebGPU compute is f32-only** for generic dtypes (WGSL has no f64/i64). **CUDA now supports all four numeric dtypes** (f32/f64/i32/i64) — non-f32 upload/download wired via CUDA buffer factories + a generic `ToDevice(CUDA)` path, making the templated `k_*_t<double/int/long long>` kernels reachable; the matrix exercises them on-device. (MatMul stays f32-only on device — cuBLAS Sgemm; no Dgemm/integer-gemm wired — so f64 matmul runs on CPU.)

## Architecture rules (enforced going forward)
1. All operands of an op are co-located on one device before dispatch.
2. Ops are **device-agnostic** — no `gpu`/`cuda` imports or `Device()==GPU` branches in op code; device choice lives only in `tensor` dispatch + `storage.Backend`.
3. No GPU↔CPU round-trips for compute; device-resident data stays on-device.

Status: **norms + conv refactored** to this pattern. `nn/norm.go` and `nn/conv.go` no longer import `gpu` (0 device refs); they route via `tensor.{RMSNorm,LayerNorm,BatchNorm,Conv2D,MaxPool2D,AvgPool2D,AdaptiveAvgPool2D}` → `storage.Backend`. CUDA + half now flow through automatically (CUDA conv/norms previously fell to CPU). Fixed two latent bugs exposed by the refactor: BatchNorm2d per-channel correctness, and WebGPU Conv2D null-bias binding when `hasBias=false`.

## WebGPU buffer-size limit (known constraint)
WebGPU caps a **single buffer at the adapter's granted `MaxBufferSize`** — **256 MB** on the RTX 5060 Ti (measured; the adapter advertises a u64::MAX sentinel, but the device grants 256 MB). `MaxStorageBufferBindingSize` is 2 GB. Implication: tensors >256 MB (e.g. LLM embedding/LM-head matrices, ~1–2 GB) **cannot live in one WebGPU buffer** — use the CUDA backend for large models. Decision (2026-06-16): **keep WebGPU** for ≤256 MB workloads (vision, CNNs, training, small/mid models, browser) — its unique niche; CUDA handles 7B+ LLMs. `gpu.NewGPUBuffer*` now returns a clear error pointing to `.CUDA()` when a buffer exceeds the cap (instead of an opaque wgpu validation error).

## Changelog
- **2026-06-18** — **Phase 3 started (LLM family)**: `VibeThinker3B()` config (Qwen2.5-Coder-3B), Mistral end-to-end verified (SWA), new **Falcon** model (parallel attn+MLP, MQA, ALiBi). Tests in `models/falcon_test.go`. Phase 3 0%→25%; overall ~37%→~40%. Also filled Phase-2 test gaps (SDPA-CUDA, Dropout2d train, GroupNorm groups).
- **2026-06-18** — **Root md consolidation**: merged `TODO.md`/`stretch.md`/`TODO_huggingface.md`/`MODELS.md`/`models.md` into this doc (§13 model target catalog & backlog) and deleted them — this doc is now the single source of truth.
- **2026-06-18** — **Whisper** added (`models/whisper.go`): first audio encoder-decoder. Audio stem = two `nn.Conv1d` over the log-mel spectrogram (k3 s1 → GELU → k3 s2 → GELU, halving the time axis) + **sinusoidal** position embeddings + pre-LN bidirectional encoder blocks; text decoder = learned token+absolute-position embeddings, pre-LN **causal self-attn + cross-attn** to the encoder, tied output projection. Whisper conventions: scaled SDPA, q/v-proj with bias but **k-proj bias-free**. `WhisperTiny/Base` configs; `EncodeAudio`/`Decode`/`Forward`. Hit and fixed a real **device co-location** issue — the conv stem places hidden states on the compute device (default GPU) while embeddings default to CPU; co-located sinusoids/token-embeds/pos/causal-mask to the data device via `ToDevice` (honors the "operands co-located, ops device-agnostic" rule; no hardcoded device in op code). Tests `models/whisper_test.go` (encode/forward shape + finiteness, sinusoid pos-0 = sin0/cos1 + bounds, configs, k-proj-no-bias). Phase 3 40%→45%.
- **2026-06-18** — **BERT / RoBERTa** added (`models/bert.go`): first encoder-only architecture. Learned absolute position + token-type (segment) embeddings summed + LayerNorm; **post-LN** residual blocks (vs pre-norm LLMs); bidirectional `nn.SDPA` (no causal mask); GELU FFN with bias on all projections; **pooler** (Linear+tanh on [CLS]); `BERTForMaskedLM` head (transform→GELU→LN→tied decoder + bias). `BERTBase/Large`, `RoBERTaBase` configs. **WordPiece tokenizer** added to `tokenizer/` (BasicTokenizer whitespace+punctuation split, optional lowercasing, greedy longest-match-first subwords with `##` continuation prefix, whole-word→UNK on failure; `##`-aware decode). Tests `models/bert_test.go` (forward/pool/MLM shape + finiteness + tanh bounds, configs) and `tokenizer/wordpiece_test.go` (greedy subword, multi-subword, punctuation, lowercase, unknown, decode). Phase 3 35%→40%.
- **2026-06-18** — **T5** added (`models/t5.go`): first encoder-decoder architecture. Bucketed **relative position bias** (`relPosBucket` matching transformers, bidirectional encoder / causal decoder, shared from layer 0), no QKᵀ scaling (T5 convention), T5LayerNorm = RMSNorm, **gated-GELU** FFN (`wo(gelu(wi_0)*wi_1)`), decoder **cross-attention** to encoder output (no position bias), tied embeddings with `d_model^-0.5` logit scaling. `T5Small/Base` configs; `Encode`/`Decode`/`Forward`. Tests `models/t5_test.go` (bucket bounds + sign separation + causal collapse + far-past saturation, encode/forward shape + finiteness, rel-bias ownership). Phase 3 30%→35%.
- **2026-06-18** — **SentencePiece Unigram tokenizer** (`tokenizer/tokenizer.go`): polymorphic model-vocab parse (BPE map OR Unigram `[piece,logprob]` array), Viterbi maximum-log-prob segmentation with single-char unk fallback, metaspace (▁) + prefix-space defaults. Unblocks T5 / Gemma / ALBERT tokenization from the HF `tokenizer.json` form (no protobuf needed). Tests `tokenizer/unigram_test.go` (parse, segmentation, unk fallback, decode round-trip); existing BPE path unaffected.
- **2026-06-18** — **Gemma 2** added (`models/gemma.go`): faithful sandwich-norm block (pre+post RMSNorm on both sublayers), GeGLU FFN, (1+weight) RMSNorm, embedding ×√hidden, attention + final logit soft-cap (`cap·tanh(x/cap)`), query pre-attn scaling, head_dim≠hidden/heads, alternating sliding-window/global attention, tied LM head. `Gemma2_2B/9B` configs. Tests `models/gemma_test.go` (shape, finiteness, soft-cap bounds, layer alternation). Phase 3 25%→30%.
- **2026-06-17** — **Phase 2 complete**: Conv3d; native CUDA kernels for GroupNorm/grouped-Conv2d/ConvTranspose2d (no more CPU round-trip, verified vs CPU); FlashAttention shared-mem tiling. Phase 2 85%→100%; overall ~34%→~37%.
- **2026-06-17** — Finished Phase-2 modules: Conv1d, InstanceNorm2d, Upsample, Dropout2d, BatchNorm **train-mode** (made **device-resident** — removed a CPU round-trip the user flagged). Renamed test file → `conv_test.go`/`norm_test.go`. Phase 2 55%→85%; overall ~30%→~34%.
- **2026-06-17** — **FlashAttention** (fused memory-efficient attention, CUDA + CPU fallback; causal/non-causal; verified vs reference). Phase 2 45%→55%; overall ~28%→~30%.
- **2026-06-17** — **Phase 2 conv modules**: grouped/depthwise Conv2d (`WithGroups`), ConvTranspose2d, GroupNorm (device-agnostic, CPU compute; backend for groups=1 conv). Added VibeThinker-3B (Qwen2.5-Coder-3B arch) + PP-OCRv6 to the model targets. Phase 2 25%→45%; overall ~25%→~28%.
- **2026-06-17** — **Phase 2 (LLM attention variants)**: reusable device-agnostic `SDPA` + Sliding Window Attention mask (Mistral) + ALiBi slopes/bias (Falcon); CPU+CUDA verified. FlashAttention (fused O(1) kernel) noted as a larger follow-up. Phase 2 0%→20%; overall ~23%→~25%.
- **2026-06-17** — **INT8 model integration done**: `cuda.Int8Buffer` + `storage.Int8Quantizer` + `tensor.ToInt8`/`Linear.ToInt8`/`Llama.QuantizeInt8`; CUDA MatMul detects int8 weights. Full int8 Llama forward verified (rel-err 0.0086). Phase 1 65%→90%; overall ~16%→~23%.
- **2026-06-17** — **INT8 core** (weight-only, symmetric per-channel): `dtype.Int8`, `QuantizeInt8PerColumn`, CUDA `cuda_matmul_int8` + `QuantizedMatMulInt8`. Verified vs f32 (rel-err 0.0051).
- **2026-06-17** — **WebGPU f16** (minimal-viable) implemented: `GPUHalfBuffer`, f16 matmul+elementwise shaders, backend routing, `SupportsHalf` capability probe. Runtime-blocked by cogentcore-v0.23 naga (no `enable f16`); detected via probe, degrades gracefully, ready for the v29 fork. Links back to the original wgpu-native v29 upgrade thread.
- **2026-06-16** — Wired **non-f32 CUDA** upload/download (CUDA buffer factories + generic `ToDevice(CUDA)`); f64/i32/i64 generic ops now run on-device (matrix verifies). MatMul stays f32-only on device. Confirmed adapter supports `shader-f16` (WebGPU f16 feasible).
- **2026-06-16** — Added table-driven **op×dtype×device test matrix** (f32/f64/i32/i64 + bf16/f16; CPU/WebGPU/CUDA). Closes the f64/i64 coverage gap; documents WebGPU/CUDA-as-f32-only for generic dtypes.
- **2026-06-16** — **Full bf16 Llama forward on-device** (CUDA): added half movement kernels + softmax/norms + mixed bf16×f32 elementwise coercion + half gates/contiguous. Verified end-to-end (rel-err 0.0049, no CPU round-trips). Phase 1 45%→65%; overall ~11%→~16%.
- **2026-06-16** — WebGPU: probed buffer limit (256 MB/buffer on this adapter); added a clear over-limit error pointing to CUDA; kept WebGPU for sub-256MB workloads. CUDA conv/norms now reachable (device-agnostic dispatch).
- **2026-06-16** — Architecture: made norms device-agnostic (`tensor.RMSNorm/LayerNorm/BatchNorm` dispatch through `storage.Backend`; `nn/norm.go` drops `gpu` import). Fixed a real **BatchNorm2d correctness bug** (4D `[N,C,H,W]` now normalizes per-channel, not just shape-preserving). CUDA/half norms now reachable from the model forward. Conv has the same anti-pattern — queued.
- **2026-06-16** — Renamed library identifiers `gotorch`→`light` (CUDA enum `LightDtype`/`LIGHT_*`, lib `liblight_cuda.a`, doc strings); all tests green.
- **2026-06-16** — Half norm/softmax CUDA kernels (RMSNorm/LayerNorm/Softmax) + plugin routing; verified on RTX 5060 Ti. Model-forward wiring still pending (`nn/norm.go` has no CUDA path).
- **2026-06-16** — Completed 1b (runtime `tensor.Cast[From,To]`); added native half elementwise/unary/scalar CUDA kernels (1d/1e inc 3b). Half tensors now do matmul + elementwise + unary + scalar. Phase 1 30%→45%; overall ~8%→~11%.
- **2026-06-16** — Plumbed real Llama weights through `ToBF16` (`Linear.ToHalf`, `Llama.Half`); demo verified on RTX 5060 Ti. Decided: finish Phase 1 before writing the dtype×op test matrix.
- **2026-06-16** — Phase 1d/1e increment 3: device-buffer integration — `cuda.HalfBuffer` (f32 host API, 16-bit device), `storage.HalfConverter`, CUDA MatMul half-routing, `tensor.ToBF16`/`ToF16`. Verified on RTX 5060 Ti. Phase 1 25%→30%; overall ~7%→~8%.
- **2026-06-16** — Phase 1d/1e increment 2: native CUDA half **matmul** (f16 + bf16) via `cublasGemmEx`, f32-accumulate; verified on RTX 5060 Ti vs CPU f32 reference. Phase 1 23%→25%; overall ~6%→~7%.
- **2026-06-16** — Phase 1d/1e increment 1: chose device-native half design; added `dtype.F16` type + `FloatToHalf` + F16 slice converters (`dtype/half_f16_test.go`). INT8 (1c) deferred to end per plan. Foundation only — no native kernels yet.
- **2026-06-16** — Merged `candle-parity.md` + `feature-coverage.md` + `parity-roadmap.md` into this file; added per-backend (light-cpu/webgpu/cuda) columns to the op tables.
- **2026-06-16** — Phase 1b broadcasting: `BroadcastTo` + `Expand`; fixed CUDA `k_binop_t` broadcast bug. Phase 1 20%→23%; overall ~5%→~6%.
- **2026-06-15** — Phase 1b: Permute(N-D), Flip, Roll, Pad, Chunk, Split, Cumsum, LogSoftmax, SoftmaxDim, Max/MinDim, ArgMax/ArgMinDim, ArgMin, Sort, TopK, Norm. Phase 1 8%→20%.
- **2026-06-15** — Phase 1a verified on RTX 5060 Ti: WebGPU + CUDA activation parity tests pass.
- **2026-06-15** — API: renamed generic constraint `dtype.Supported` → `dtype.Numeric` (403 refs / 51 files).
- **2026-06-15** — Phase 1a: ELU, LeakyReLU, Mish, HardSigmoid, HardSwish across CPU/WebGPU/CUDA.

---

## 13. Model target catalog & backlog (merged from root `TODO.md` / `TODO_huggingface.md` / `MODELS.md` / `models.md` / `stretch.md`)

> These root files were consolidated here on 2026-06-18 so this doc is the single source of truth. Status reflects the current codebase. ✅ done · 🟡 partial · ⬜ not started.

### 13.1 HF text-generation targets (RTX 5060 16 GB; F16/BF16 ≈ 2 B/param, ~7 B ceiling)
| Model | Params | F16 VRAM | Status |
|-------|-------|----------|--------|
| Llama 3.2 1B/3B/8B | 1–8 B | 2–16 GB | ✅ `Llama1B/3B/8B` |
| Qwen2.5 0.5–7B (+ Coder) | 0.5–7 B | 1–14 GB | ✅ `Qwen25_*` |
| Mistral 7B (SWA) | 7 B | 14 GB | ✅ `Mistral7B` (arch; SentencePiece for official weights) |
| Falcon (MQA/ALiBi) | 7 B | 14 GB | ✅ `models.Falcon` |
| VibeThinker-3B | 3 B | 6 GB | ✅ `VibeThinker3B` (Qwen-Coder-3B arch) |
| Phi-3.5 Mini | 3.8 B | 7.6 GB | 🟡 config; partial RoPE factor |
| Gemma 2 2B / 9B | 2–9 B | 4–18 GB | ✅ `Gemma2_2B/9B` (arch + Unigram tokenizer) |
| SmolLM2 1.7B | 1.7 B | 3.4 GB | 🟡 Llama-arch config away |
| DeepSeek-R1-distill 7B / Yi-1.5 6B | 6–7 B | 12–14 GB | ⬜ (Llama/Qwen-family configs) |

### 13.2 Diffusion (text→image / image→image)
VAE enc/dec ⬜ · schedulers (DDIM/DPM++/Euler) ⬜ · time-embedding ⬜ · cross-attention UNet 🟡 (UNet arch exists) · CFG ⬜ · txt2img/img2img/inpaint pipelines ⬜ · ControlNet ⬜ · image PNG/JPEG I/O ⬜. Building blocks now ready: **ConvTranspose2d ✅, GroupNorm ✅**. Targets: SD1.5/2.1, SDXL(+Turbo/Lightning), LCM, PixArt-Σ, SD3/3.5, Stable Cascade. (Flux.1 12B doesn't fit 16 GB F16.)

### 13.3 Multimodal (VLM)
moondream2, PaliGemma 3B, Phi-3.5-Vision, LLaVA 1.5/NeXT, Qwen2-VL, InternVL2, MiniCPM-V, BLIP-2, SmolVLM — all ⬜. Need: vision projector, LLaVA bridge, dynamic tiling, Q-Former. CLIP/ViT encoders ✅.

### 13.4 Vision (torchvision) backlog
Done: ✅ LeNet, AlexNet, ResNet(18–152), ViT, YOLOv3-tiny, (SD-)UNet, CLIP. ⬜: VGG, SqueezeNet, DenseNet, GoogLeNet/Inception, ResNeXt/WideResNet, MnasNet, **MobileNet v2/v3**, ShuffleNet v2, **EfficientNet**, RegNet, ConvNeXt, Swin/MaxViT; detection (Faster/Mask R-CNN, RetinaNet, SSD, FCOS, DETR); segmentation (FCN, DeepLabv3, LRASPP); video (R3D, R(2+1)D, MViT, S3D, Swin3D); optical flow (RAFT). PP-OCRv6 (CNN+SVTR) ⬜.
**Prereqs status:** depthwise/grouped Conv2d ✅ · BatchNorm train ✅ · AdaptiveAvgPool ✅ · 3D conv ✅ · INT8 ✅ · **Squeeze-and-Excitation ⬜ · window attention ⬜ · FPN ⬜ · RoI Align/Pool ⬜ · patch embed ✅ (ViT)**.

### 13.5 Backlog — not doing now (revisit)
ONNX model import · video/audio dataset loader · YOLO-tiny Darknet weight loader (model exists) · mixed-precision *training* (f16 fwd / f32 bwd) · beam search · repetition penalty (sampling). Stretch: Light 3D branching-flow NN module.
