# Light ⇄ Candle parity (2026-10-28)

Reference: HuggingFace Candle 0.11.0 (crates.io 2026-06-26) plus `main` as of 2026-10-14.
Candle is still the HF Rust framework; it has not been renamed or superseded.
Light's side is read from the live registry (`go run ./scripts/opmatrix`) and
FEATURES.md / MODELS.md. This file supersedes the GoLight-era
`docs/archive-golight/parity.md`.

Legend: ✅ have · ⚠️ partial · ❌ missing · — not applicable.

## 1. Headline

| Area | Candle 0.11 | Light | Gap size |
|------|-------------|-------|----------|
| Backends | CPU, CUDA (cuBLAS, optional cuDNN, NCCL multi-GPU, cuTile JIT), Metal, WASM-CPU. WGPU is an open PR, not merged | CPU, CUDA (cuBLAS, cuDNN), WebGPU (merged, F32) | Metal ❌; WebGPU is ours only. NCCL (multi-GPU collectives) is irrelevant on one GPU. cuTile is a Python/Rust tile DSL with no Go frontend: not applicable; our equivalent is nvcc→PTX, and the capabilities Candle builds with it (fused MoE) are tracked as ops, not as a platform gap |
| DTypes | U8 U32 I16 I32 I64 BF16 F16 F32 F64 F8E4M3 (+ storage-only F6/F4/F8E8M0) | F16 BF16 F32 F64 F8E4M3 U8 I8 I16 U16 U32 I32 U64 I64 (CPU); CUDA compute F32/F64 broad, BF16 + F16 GEMM/elementwise, F8E4M3 cast + GEMM | none: F8E4M3 landed 2026-10-28 with cast (CPU/CUDA) and a cuBLASLt FP8 GEMM, which Candle lacks |
| Core ops | ~45 op families, all three backends | 80 registered ops; 4 without CUDA kernel (einsum, lpnorm, sort, topk) | small |
| Fused attention | FA2 + FA3 (CUDA), Metal SDPA, CPU flash-attn, paged | FlashAttention CUDA tiled (D≤256) + CPU | ⚠️ no paged, no varlen |
| **Quantized inference** | GGUF Q4_0…Q8_K matmul on CPU/CUDA/Metal, MMVQ/MMQ fast kernels, quantized MoE, quantized embeddings | GGUF dequant for all 12 types; fused dequant-matmul Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K on CUDA (v1 kernels, 2026-10-27); int8 per-column and FP8 E4M3 per-tensor safetensors paths | ⚠️ kernel speed (MMVQ/MMQ) |
| MoE | fused `moe_gemm` (f16/bf16/GGUF, CUDA) | ✅ `ops.MoEGemm` dense F32/BF16/F16 + `ops.MoEGemmQ` GGUF Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K experts (fused CUDA kernels, CPU reference), `MoERoute`, `nn.SparseMoE` (2026-10-28) | WMMA tensor-core tiles are a perf item |
| Models | ~110 modules (see §4) | see MODELS.md (Llama family, Qwen2/3, Gemma 3, GPT-2, BERT, Whisper, Qwen3-ASR, SD, vision refs) | large, by design |
| Audio | whisper, encodec, mimi, dac, snac, metavoice, parler_tts, csm | Whisper + Qwen3-ASR (STT), Mimi codec, Moshi speech-to-speech LM (2026-11-06) | TTS gap (parler/metavoice/csm) |
| Training | tape autograd, SGD (no momentum), AdamW, 5 losses; no schedulers; 2 examples | dynamic DAG autograd, SGD+momentum, AdamW, CosineLR, 10+ training examples, 90% CIFAR recipe | Light ahead (but LLM fine-tune blocked, see §6) |
| ONNX | candle-onnx, ~80 ops | ❌ | ecosystem |
| Tokenizers | `tokenizers` crate (everything) + GGUF-metadata tokenizer | BPE byte-level, BPE SentencePiece, WordPiece, Unigram, CLIP, GGUF-metadata tokenizer (2026-10-29, llama.cpp-exact); no Jinja chat templates (hand-coded per family in `engine`) | ⚠️ templates only |

## 2. Core ops, per family

| Family | Candle | Light | Notes |
|--------|--------|-------|-------|
| Unary: exp log sin cos tanh neg sqrt abs floor ceil round relu silu gelu gelu_erf sigmoid | ✅ | ✅ | |
| recip, sqr, sign, erf, elu, powf (scalar), clamp, round_to, affine | ✅ | ✅ (2026-10-28: fused CPU + CUDA kernels F32/F64/BF16; sign also I32/I64; `RoundTo` composed) | plus mish, hard_swish, hard_sigmoid, leaky_relu |
| Binary add sub mul div + broadcast | ✅ | ✅ | |
| maximum / minimum (elementwise) | ✅ | ✅ (2026-10-28, broadcast; F32/F64/BF16/I32/I64; autograd routes grad to the winner) | |
| Compare eq ne lt le gt ge → U8 | ✅ | ✅ (CUDA F32/F64/I32/I64/U8/BF16; where likewise) | widened 2026-10-28 |
| Reduce sum min max argmin argmax (+ mean var logsumexp norm) | ✅ | ✅ (var, logsumexp composed) | |
| where, index_select, gather, scatter_add | ✅ | ✅ | |
| scatter / scatter_set / index_add / slice_scatter / slice_assign | ✅ | ✅ (2026-10-28: `Scatter`, `IndexAdd`, `SliceScatter`, `SliceAssign`; gather/scatter/scatter_add/index_add on CPU + CUDA for F32/F64/I32/I64/BF16/F16) | |
| matmul, batched, mv, dot | ✅ | ✅ F32/F64/F16/BF16 2-D (plain + transposed-weight) and batched on CUDA via cuBLAS GemmEx (F16/BF16 + plain-layout BF16 landed 2026-10-28); `MatMulMixed` F32×{BF16,F16}; `MatMulF8` E4M3 via cuBLASLt (K%16==0) | Light ahead on FP8 |
| conv1d, conv2d (groups, dilation), conv_transpose1d/2d | ✅ | ✅ conv1d/2d/3d; conv_transpose1d/2d (stride, padding, output_padding, dilation, groups; CPU F32/F64, CUDA F32/BF16) 2026-10-28 | matches `F.conv_transpose2d` |
| avg/max pool2d, upsample_nearest 1d/2d, upsample_bilinear2d, interpolate | ✅ | ✅ pools + nearest 2d + bilinear 2d (align_corners both ways, up and down) 2026-10-28 | |
| cat, narrow, chunk, pad, flip, repeat, unfold, tril/triu/eye, meshgrid | ✅ | ✅ (`ops.Unfold` 2026-10-28; `tensor.Eye` / `tensor.Meshgrid` existed) | |
| sort / argsort (last dim) | ✅ all backends | ✅ CPU + CUDA bitonic (F32/F64/I32/I64, stable tie order) | done 2026-10-27 |
| cumsum | matmul trick | ✅ kernel | Light ahead |
| topk | ❌ (sampling only) | ✅ CPU + CUDA (sort + narrow) | Light ahead |
| einsum | ❌ | ✅ CPU; CUDA F32 fully on-device (diagonals, N-ary) | Light ahead |
| rope (rotate-half), rope_i (interleaved), rope_thd | ✅ ×3 | ✅ ×3 (`RoPEInterleaved` CPU+CUDA F32/F64/BF16, `RoPETHD` layout wrapper, 2026-10-28) | |
| fused rms_norm / layer_norm | ✅ (CPU F16/BF16/F32, CUDA, Metal) | ✅ (CUDA F32/BF16) | |
| softmax / log_softmax | ✅ | ✅ | |
| custom op API | CustomOp1/2/3 + inplace | registry key + kernel func | equivalent |

## 3. candle-nn

| Module | Candle | Light |
|--------|--------|-------|
| Linear, Conv1d/2d, Embedding, LayerNorm, RmsNorm, GroupNorm, BatchNorm, Dropout, Sequential | ✅ | ✅ |
| ConvTranspose1d/2d | ✅ | ✅ `nn.ConvTranspose2d` (inference) + `ops.ConvTranspose1d/2d` |
| LSTM, GRU | ✅ | ✅ `nn.LSTM` / `nn.GRU` (PyTorch weight layout, differentiable unroll; match torch to 1e-5 on CPU + CUDA) |
| PReLU, activation enum (14 variants) | ✅ | ✅ ReLU/ReLU6/SiLU/Sigmoid/Tanh/GELU/ELU/LeakyReLU/Mish/HardSwish/HardSigmoid/PReLU/SwiGLU modules, all with gradients |
| KV caches: KvCache, RotatingKvCache (sliding window), Scattered (batched fixed ctx), Concat, Interleaved | ✅ | ✅ rotating/sliding window in `models.KVCache` (per-layer window; concat-grown, no Scattered/Interleaved) |
| rotary_emb, sdpa (Metal), flash_attn (CPU), varlen | ✅ | ✅ RoPE, SDPA masks, FlashAttention |
| moe_gemm | ✅ | ✅ dense (`ops.MoEGemm`) and GGUF (`ops.MoEGemmQ`, same six formats as `matmul_q`), 2026-10-28 |
| VarBuilder (safetensors mmap, sharded, npz, pth, varmap save/load) | ✅ | ✅ safetensors + sharded, `npy` (npy/npz read+write), `pth` (torch.save zip: state dicts, checkpoint keys, pickled modules), `nn.VarMap` save/load (2026-10-29) |
| Optimizers: SGD, AdamW | ✅ | ✅ + momentum + CosineLR |
| Losses: nll, cross_entropy, mse, bce_with_logits, huber | ✅ | ✅ all five (autograd, grad-checked) |
| pixel_shuffle/unshuffle, replication_pad2d | ✅ | ✅ (`ops.PixelShuffle/PixelUnshuffle/ReplicationPad2d`, device-agnostic compositions) |

## 4. Models

Candle `candle-transformers/src/models` (main, Oct 2026), grouped, with Light's status. **2026 additions in bold.**

**LLMs (dense):** llama ✅, mistral ⚠️ (loader path + sliding window done 2026-10-28, no real-weight logit check yet), phi ❌ (Candle's `phi` = Phi-1/1.5/2 parallel-block architecture, model_type `phi`; not ported), phi3 ✅ (Phi-3.5-mini; **Phi-4-mini** verified 2026-10-29 with partial rotary 0.75 + longrope + tied head + GPT-4o tokenizer, which Candle's phi3 lacks), gemma3 ✅ (1B text 2026-10-28, 4B text-only 2026-10-29 with linear rope_scaling; 12B/27B same path, over 16 GB BF16; SigLIP vision input open), gemma2 ✅ (2B verified 2026-10-29: softcaps, no QK-norm), **gemma4** ✅ (E2B text verified 2026-10-29 with PLE, KV-shared layers, double-wide MLP, proportional RoPE; Candle's gemma4 text port lacks PLE/KV sharing, so Light is ahead on E2B/E4B), gemma (v1) ❌, qwen2 ✅, qwen3 ✅ (dense via Qwen3-ASR/SDAR bodies; standalone checkpoint logit check still open), deepseek2 ❌, **falcon** ✅ (2026-10-29: all three decoder layouts incl. ALiBi and the 40B grouped-K/V architecture, which Candle's falcon.rs rejects; falcon-rw-1b verified vs transformers), bigcode / starcoder2 ❌, mpt ❌, persimmon ❌, stable_lm ❌, yi ❌ (= llama), olmo / olmo2 ❌, granite ❌, helium ❌, chatglm / glm4 / glm4_new / codegeex4 ❌, **lfm2** ❌, **smol** (SmolLM3) ❌, llama2_c ❌, mixformer (phi-1/2) ❌.
**LLMs (MoE / recurrent):** qwen3_moe ✅ (2026-10-28, HF + GGUF), mixtral / qwen2_moe ⚠️ (same `nn.SparseMoE` block, loaders not wired), granitemoehybrid, based, mamba, **mamba2**, rwkv_v5/v6/**v7**, recurrent_gemma ❌ (no SSM/linear-attention scan ops).
**Quantized GGUF variants:** 19 modules in Candle; Light: one generic `LoadLlamaGGUF` covering the llama / qwen2 / qwen3 / mistral / qwen3moe GGUF architectures with quantized embedding, tied head and MoE experts (2026-10-27…29); verified against llama.cpp tokenizer fixtures and transformers logits (Qwen2 0.5B, Llama-3.2, Qwen3-MoE).
**Encoders / embeddings:** bert ✅, **embedding_gemma2** ✅ (EmbeddingGemma 2 verified 2026-10-29; Candle has no port, Light ahead), distilbert ❌, jina_bert ❌, **nomic_bert** ❌, modernbert ❌, debertav2 ❌, xlm_roberta ❌, stella_en_v5 ❌, nvembed_v2 ❌, colpali ❌, t5 ❌, marian ❌.
**Vision:** resnet ✅, vgg ✅ (random init), vit ✅, beit / dinov2 / dinov2reg4 / eva2 ❌, convnext ❌, convmixer ❌, efficientnet / efficientvit ❌, mobileone / mobilenetv4 / fastvit ❌ (we have mobilenetv2 instead), hiera ❌, repvgg ❌, segformer ❌, segment_anything ❌, depth_anything_v2 ❌, trocr ❌. Light-only: yolov5 (random init), resnet9, lenet, alexnet.
**Multimodal:** clip ⚠️ (SD text tower only), openclip / chinese_clip / siglip / mobileclip ❌, blip ❌, llava ❌, moondream ❌, paligemma ❌, pixtral ❌, qwen3_vl ❌, **paddleocr_vl** ❌, voxtral ❌.
**Diffusion:** stable_diffusion ⚠️ (1.5 only; Candle: 1.5/2.1/XL/Turbo), mmdit (SD3) ❌, flux ❌, wuerstchen ❌, **z_image** ❌.
**Audio:** whisper ✅ (2026-10-28, tiny…large-v3, verified), Light-only qwen3_asr ✅ (2026-10-28), **mimi** ✅ (2026-10-29, verified vs transformers), **moshi** ✅ (2026-11-06, Q8 GGUF, verified vs Rust moshi-core: text top-5 + 47/48 codes over 6 frames; full-duplex protocol + offline speech-to-speech); encodec, dac, snac, metavoice, parler_tts, csm ❌.

Validated-with-real-weights count (2026-11-06): Candle ~110 modules (its examples run them); Light 24 (Moshi LM, Mimi, Phi-4-mini, Falcon, GPT-2, Qwen2, Llama-3.2, Phi-3.5, BERT/bge-small, SD 1.5, ResNet-50, MobileNetV2, ViT-B/16, MercyLM, Whisper-tiny, Qwen3-ASR-1.7B, Qwen3-MoE (tiny-random), SDAR-1.7B, Gemma 3 1B, Gemma 3 4B, Gemma 2 2B, Gemma 4 E2B, EmbeddingGemma 2, plus GPT-2 on WebGPU).

## 5. Quantization and precision

| | Candle | Light |
|--|--------|-------|
| GGUF read / write | ✅ / ✅ | ✅ / minimal |
| GGML dtypes | F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1 Q2K–Q8K (no IQ/TQ) | same set, dequant only |
| Quantized matmul CPU | f32 + f16, NEON/AVX-512 repacked kernels | ⚠️ dequant + matmul fallback (correct, slow) |
| Quantized matmul CUDA | dequant, MMVQ (gemv), MMQ (gemm) for Q4_0…Q8_0, Q2K–Q6K; bf16/f16/f32 activations | ⚠️ fused dequant-matmul for Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K (F32 activations), v1 kernel; others dequant+cuBLAS. Done 2026-10-27, see MODELS.md |
| GGUF model loader | 19 quantized model modules | ✅ `LoadLlamaGGUF` (llama / qwen2 / mistral arch) |
| Quantized embedding, quantized MoE | ✅ | ✅ both: quantized MoE (2026-10-28); quantized embedding lookup (2026-10-29: the GGUF table stays in block format, `IndexSelect` gathers rows of blocks on CPU/CUDA and only those are dequantized; tied LM head reuses the quantized storage where a fused matmul exists) |
| Tokenizer from GGUF metadata | ✅ | ✅ `tokenizer.LoadGGUF` (2026-10-29): SPM + byte-level BPE with llama.cpp pre-tokenizer families; matches llama.cpp ids on its vocab fixtures |
| safetensors int8 / fp8 / GPTQ / AWQ | ❌ | int8 per-column; `F8_E4M3` tensors load, `nn.Linear` runs F8 weights with a per-tensor `WScale` (blockwise DeepSeek-style scales not yet) | neither does GPTQ/AWQ |
| BF16 LLM path | ✅ | ✅ |
| F8E4M3 | dtype + cast | ✅ dtype + cast (CPU/CUDA, RNE + saturate) + FP8 GEMM (cuBLASLt, F32/BF16/F16 out) + `QuantizeF8` (2026-10-28) |

## 6. Training

Light is ahead on everything except breadth of backward ops. Candle: tape autograd; backward for conv1d/2d, conv_transpose2d, pools, upsample_nearest, index ops, cat/narrow, reductions; no backward for fused norms, rope, sdpa, flash-attn, quantized matmul, MoE; SGD without momentum; no schedulers; two training examples. Light: dynamic DAG, 30 differentiable functions incl. fused LayerNorm/RMSNorm, BatchNorm2d train mode, AdamW, momentum SGD, CosineLR, ten training examples. Light gaps that block LLM fine-tuning: no backward for RoPE, Concat, Cast, GroupNorm, FlashAttention; no grad clipping, loss scaler, LoRA, checkpoint helper. Deferred until inference parity is reached (user decision 2026-10-27).

## 7. What Light has that Candle lacks

Merged WebGPU backend · einsum · cumsum/topk kernels · Conv3d · InstanceNorm · momentum SGD + LR scheduler + real training recipes · BatchNorm train-mode CUDA kernels · FP8 E4M3 GEMM on tensor cores (cuBLASLt) · int8 per-column safetensors quant · op×device×dtype matrix generator · verification harness vs transformers with a stated bar.

## 8. Ordered plan to close the inference gap

Ranked by (models unlocked on a 16 GB GPU) ÷ (effort). Validation bar for each model: top-5 logits vs transformers on a ≥15-token prompt (MODELS.md).

1. ~~**Quantized matmul on CUDA + CPU**~~ — ✅ 2026-10-27: `ops.MatMulQ` (Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K fused; others dequant+cuBLAS), `models.LoadLlamaGGUF`. Remaining: kernel performance (v1 is 17 tok/s on 0.5B vs 31 F32). GGUF-metadata tokenizer and quantized embedding lookup done 2026-10-29.
2. ~~**Sort/argsort + TopK CUDA kernels**~~ — ✅ 2026-10-27 (bitonic; einsum N-ary/diagonal on CUDA done alongside).
2b. **Performance pass** (requested): benchmark harness, time + peak memory, CPU and CUDA, per op and per model; MMVQ-style int8-activation quant kernel; MMQ tiles for prefill.
3. ~~**Sliding-window attention + RotatingKvCache in `models.Llama`**~~ done 2026-10-28, verified vs transformers (Qwen2 with `use_sliding_window`, window 8/6, chunked decode == single shot). Mistral 7B GGUF Q4 end-to-end run still pending a checkpoint download; Gemma 2/3 next (alternating `layer_types` already parsed).
4. ~~**Whisper**~~ done 2026-10-28 (`models.Whisper` + `audio` front end, verified on whisper-tiny; any openai/whisper-* size loads).
4b. ~~**Qwen3-ASR-1.7B**~~ done 2026-10-28 (`models.Qwen3ASR`: Conv2d chunk stem + windowed AuT encoder + projector + Qwen3 decoder through `Llama.ForwardEmbeds`; verified vs transformers 5.21: mel/encoder/projector/step-0 top-5/greedy). Brought `LlamaConfig.QKNorm` (Qwen3 dense decoder) with it.
3c. **Gemma 3** (2026-10-28): `models.Gemma3` verified on gemma-3-1b-it (F32 4 dp vs transformers; BF16 greedy identical). 2026-10-29: gemma-3-4b-it loads text-only from the multimodal checkpoint (linear RoPE ×8 on global layers) and matches transformers at BF16 (top-5 ids + greedy). Gemma 2 2B verified 2026-10-29 (F32 4 dp, greedy identical; QK-norm made optional). SigLIP tower for image input open.
3b. **Chat engine + Studio** (2026-10-28): `engine/` (catalog, templates, streaming, OpenAI API), `cmd/light-serve`, `studio/` Wails v3 + React desktop app. Candle has no equivalent; this is the Light-only product surface.
5. ~~**EmbeddingGemma 2 (270M text)**~~ ✅ 2026-10-29 (`models.EmbeddingGemma2` shares `gemma4Block`; bidirectional masks + projection-only PLE); next on the encoder path: nomic-bert / modernbert.
6. ~~Qwen3 dense~~ (qk-norm landed with Qwen3-ASR; a standalone Qwen3 checkpoint still needs the logit check) → ~~Qwen3-MoE~~ (2026-10-28: `qwen3_moe` loads on the Llama stack with `nn.SparseMoE`; tiny-random checkpoint matches transformers to 5 dp on CPU/CUDA; GGUF `qwen3moe` loads with quantized experts (`moe_gemm_gguf` done 2026-10-28, verified on a Q8_0/Q4_0 round trip of the tiny checkpoint); Qwen3-30B-A3B Q4_K_M (≈18 GB) still exceeds the 16 GB card, Q3/Q2 or CPU offload needed) → Mixtral (same block, `block_sparse_moe.*` names) → Mellum 2.1, Qwen3.6-35B-A3B.
7. ~~**Gemma 3 / 4**~~ (Gemma 3 1B/4B, Gemma 2 2B, Gemma 4 E2B all verified by 2026-10-29) and **Muse Glimmer 30B** (local/global + NoPE) share the SWA machinery from step 3.
8. **TTS**: Kokoro-82M (StyleTTS2 decoder, needs phonemizer) or Parler-TTS (Candle has it); EnCodec/SNAC codec first since both TTS models emit codec tokens.
9. **Qwen3.8-27B / Qwen3-Next**: Gated DeltaNet chunked scan + gated attention + MRoPE + MTP — the largest single item; do after MoE and quant are solid.
10. **Vision/multimodal catch-up**: ~~ConvTranspose2d + bilinear upsample~~ (done 2026-10-28), DINOv2, SigLIP, Qwen3-VL tower (→ WeVisOne-VL), Segment Anything, Depth Anything v2.
11. **Diffusion**: SDXL, then a DiT (SD3 mmdit / Qwen-Image / Z-Image).
12. **Ecosystem**: Metal backend (defer), candle-onnx equivalent (defer). ~~F8E4M3~~ done 2026-10-28.

**User-queued targets (2026-10-28, in the order given):** LTX-2.5 (video DiT + video VAE), `black-forest-labs/flux-3-image-klein-comfy` (FLUX image model, ComfyUI checkpoint layout), `XHToken/Spark-X2.6-4B`, LFM3 (Liquid AI), `HelpingAI/PixelGen`. Each starts with a config/architecture inspection, then the usual validation bar. Added later the same day: ~~**diffusion LLMs**~~ — done 2026-10-28 with SDAR-1.7B-Chat (block diffusion on a Qwen3 body: `Llama.ForwardDiffusion` + `GenerateDiffusion`, verified vs transformers). LLaDA-8B / Dream-7B (full-sequence masked diffusion, 14–16 GB BF16) would need the same mask machinery with block = whole sequence plus a quantized path to fit the card; Candle has no diffusion LM at all.


~~Small parity items~~ — all landed 2026-10-28 (maximum/minimum, clamp, erf, sign, elu, leaky_relu, mish, hard_swish/hard_sigmoid, recip/sqr/affine/powf/round_to, unfold, rope_i, rope_thd, losses nll/mse/bce/huber, PReLU, pixel_shuffle, replication_pad2d, LSTM/GRU) plus conv_transpose1d/2d and upsample_bilinear2d. F16/BF16/FP8 CUDA GEMMs landed the same day. scatter / index_add / slice_scatter / slice_assign and the dense `moe_gemm` landed 2026-10-28 too. `moe_gemm_gguf` landed 2026-10-28 as well. npz/pth loading and `nn.VarMap` landed 2026-10-29, together with full-dtype CUDA casts (every dtype ↔ F32 kernel, other pairs hop through F32) and a fix to the CPU BF16 conversion (truncation → round-to-nearest-even, so CPU BF16 now matches CUDA and torch bit for bit). No op-level loader gaps remain; the remaining non-model parity items are paged/varlen attention and quantized-kernel speed (GGUF-metadata tokenizer and quantized embedding lookup landed 2026-10-29).
