# Light — Feature Tracker

**Legend:** `[x]` complete & tested (CPU + CUDA parity where applicable), `[~]` in-progress, `[ ]` not started.

**Phase-done rule:** A phase is marked *complete* only when every row in its section has CPU + CUDA implementations **and** a parity test. "CPU only" rows mean the phase is still `[~] in progress`, not done.

**Kernel coverage is generated, not hand-edited.** Run `go run -tags webgpu ./scripts/opmatrix -md` and paste the table into the section below after adding kernels. Everything else in this file is maintained by hand.

_Last updated: 2026-11-06 (stale open items reconciled with CHECKPOINTS.md)._

---

## Roadmap addition

- **Phase 11 — OpenMythos compatibility.** Goal: run [OpenMythos](https://github.com/kyegomez/OpenMythos) models on this backend. Scoped after the core stack (Phases 1-9) reaches CPU↔CUDA parity.

---

## Core types

- [x] `dtype.DType` enum + `Size()` / `IsFloat()` / `IsInt()` / `IsSigned()` / `IsQuantized()` / `BlockElems()` / `BlockBytes()` / `ByteSize()` / `String()` / `Valid()`
- [x] `dtype.Numeric` constraint and `dtype.Of[T]()`
- [x] `dtype.Half` / `dtype.BFloat16` software conversion (round-trip, subnormals, NaN/Inf)
- [x] `dtype.F8E4M3` + `dtype.Float8` conversion (fn encoding: RNE, saturate to ±448, NaN 0x7F; every code round-trips) — 2026-10-28
- [x] `shape.Shape` (immutable, negative indexing)
- [x] `shape.Strides` + contiguity check + `Permute` + `ContigWithDim0Broadcast`
- [x] `shape.Broadcast` / `BroadcastN` / `BroadcastStrides`
- [x] `device.Device` interface: `device.CPU()`, `device.CUDA(ordinal)`, `device.WebGPU()` (`-tags webgpu`), `device.Auto()` (CUDA → WebGPU → CPU), `IsCPU` / `IsCUDA` / `IsWebGPU`, `MustCUDA`, `CUDAAvailable`
- [x] `storage.Storage` CPU + `runtime.AddCleanup` via `storage.Track[T]`
- [x] `storage.Storage` CUDA (`cudaMalloc`/`cudaFree`, `cudaMemset` zero-init, refcount-driven cleanup)
- [x] `storage.Storage` WebGPU (`storage.NewWebGPU`, wgpu buffers; 2 GiB per-buffer ceiling requested explicitly at device init)
- [x] `tensor.Tensor` constructors: `Zeros`, `Ones`, `Full[T]`, `New[T]`, `Arange`, `Rand`
- [x] `tensor.Tensor` constructor: `RandN`
- [x] Views: `Reshape` (with -1 infer), `Permute`, `Transpose`, `Contiguous`, `Narrow`, `Slice`, `Expand`
- [x] `tensor.To(dev)` across all three device kinds; per-kind `Contiguous` / `Copy` hooks (`tensor.RegisterContig`, `tensor.RegisterCopy`)
- [x] View sharing via atomic refcount (`storage.Track` / `Release`)
- [x] `tensor.ToSlice[T]` / `MustSlice[T]` / `ToHalfSlice` / `ToBF16Slice`
- [x] Autograd fields on `Tensor` (`requiresGrad`, `grad`, `gradFn`, `outputNr`, `version`)

## Kernel coverage (generated 2026-10-28)

103 registered ops. Rows list the dtypes with a registered kernel per device. `adaptive_avgpool2d` and `batchnorm_eval` are composed from other ops on CPU/CUDA and only appear as native kernels on WebGPU.

**CUDA parity:** every op runs on the device. `sort` has bitonic kernels (F32/F64/I32/I64); `topk` lowers to sort + narrow; `einsum` F32 runs fully on CUDA (diagonals via `AsStrided`, N operands pairwise, pairs via batched matmul; F64 einsum still falls back to the host); `lpnorm` lowers to abs/pow/sum. The three rows without a CUDA kernel entry are these lowered ops, not gaps. `matmul_q` is CUDA-only by design (CPU dequantizes).

**WebGPU gaps:** no F16/BF16/F64 arithmetic; no `rope`, `conv1d`, `gather`, `cumsum`, compare, `groupnorm`, `where`, `flash_attention`.

| Op | CPU | CUDA | WebGPU |
|----|-----|------|--------|
| `abs` | F16 BF16 F32 F64 U8 I64 I32 I8 | F32 F64 | F32 |
| `adaptive_avgpool2d` | — | — | F32 |
| `add` | F16 BF16 F32 F64 U8 I64 I8 I32 | F16 BF16 F32 F64 I64 I32 | F32 |
| `affine` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `all` | U8 | U8 | — |
| `any` | U8 | U8 | — |
| `argmax` | F32 F64 I64 I32 | F32 | — |
| `argmin` | F32 F64 I64 I32 | F32 | — |
| `avgpool2d` | F32 F64 | F32 | F32 |
| `batchnorm_eval` | — | — | F32 |
| `cast` | F16 BF16 F32 F64 U8 U32 I64 I32 I16 I8 U16 U64 F8E4M3 | F16 BF16 F32 F64 U8 U32 I64 I8 I16 F8E4M3 U64 I32 U16 | F32 U32 I64 I32 |
| `ceil` | F16 BF16 F32 F64 | F32 F64 | F32 |
| `clamp` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `concat` | F16 BF16 F32 F64 U8 U32 I64 I32 I16 U16 U64 I8 | F16 BF16 F32 F64 U8 U32 I64 I32 I8 U16 I16 U64 | F32 F64 U32 I64 I32 U64 |
| `conv1d` | F32 F64 | F32 | — |
| `conv2d` | F32 F64 | BF16 F32 | F32 |
| `conv3d` | F32 F64 | F32 | — |
| `conv_transpose2d` | F32 F64 | BF16 F32 | — |
| `cos` | F16 BF16 F32 F64 | F32 F64 | F32 |
| `cumprod` | F32 F64 I64 I32 | F32 | — |
| `cumsum` | F32 F64 I64 I32 | F32 | — |
| `div` | F16 BF16 F32 F64 U8 I64 I8 I32 | F16 BF16 F32 F64 | F32 |
| `dropout` | F32 F64 | F32 | — |
| `einsum` | F32 F64 | — | — |
| `elu` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `eq` | F16 BF16 F32 F64 U8 U32 I64 I16 I8 I32 U16 U64 | BF16 F32 F64 U8 I64 I32 | — |
| `erf` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `exp` | F16 BF16 F32 F64 | F16 BF16 F32 F64 | F32 |
| `flash_attention` | F32 F64 | F32 | — |
| `flip` | F32 F64 U8 I64 I32 I8 | F32 | — |
| `floor` | F16 BF16 F32 F64 | F32 F64 | F32 |
| `gather` | F16 BF16 F32 F64 I64 I32 | F16 BF16 F32 F64 I64 I32 | — |
| `ge` | F16 BF16 F32 F64 U8 U32 I64 I32 U64 I8 I16 U16 | BF16 F32 F64 U8 I64 I32 | — |
| `geglu` | — | BF16 F32 | — |
| `gelu` | F16 BF16 F32 F64 | BF16 F32 F64 | F32 |
| `gelu_erf` | F16 BF16 F32 F64 | F32 F64 | — |
| `groupnorm` | F32 F64 | BF16 F32 | — |
| `gt` | F16 BF16 F32 F64 U8 U32 I64 I16 I8 U64 I32 U16 | BF16 F32 F64 U8 I64 I32 | — |
| `hard_sigmoid` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `hard_swish` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `index_add` | F16 BF16 F32 F64 I64 I32 | F16 BF16 F32 F64 I64 I32 | — |
| `index_select` | F16 BF16 F32 F64 U8 U32 I64 Q8_0 F8E4M3 Q4_0 U16 U64 I8 Q6_K I16 Q5_K Q4_1 Q8_1 Q5_0 Q5_1 Q2_K Q3_K I32 Q4_K Q8_K | F16 BF16 F32 U8 U32 I8 Q5_0 Q8_1 Q4_K Q5_K U64 Q2_K Q5_1 Q4_0 Q6_K Q3_K F8E4M3 I16 Q4_1 U16 Q8_K Q8_0 | F32 F64 U32 I64 U64 I32 |
| `l2norm` | F32 F64 | F32 F64 | — |
| `layernorm` | F32 F64 | BF16 F32 | F32 |
| `le` | F16 BF16 F32 F64 U8 U32 I64 U64 U16 I8 I32 I16 | BF16 F32 F64 U8 I64 I32 | — |
| `leaky_relu` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `log` | F16 BF16 F32 F64 | F16 BF16 F32 F64 | F32 |
| `logical_and` | U8 | U8 | — |
| `logical_not` | U8 | U8 | — |
| `logical_or` | U8 | U8 | — |
| `logsoftmax` | F32 F64 | F32 | F32 |
| `lpnorm` | F32 F64 | — | — |
| `lt` | F16 BF16 F32 F64 U8 U32 I64 U64 I16 U16 I8 I32 | BF16 F32 F64 U8 I64 I32 | — |
| `matmul` | F16 BF16 F32 F64 I64 I32 | F16 BF16 F32 F64 | F32 |
| `matmul_batched` | F16 BF16 F32 F64 | F16 BF16 F32 | F32 |
| `matmul_f8` | F16 BF16 F32 | F16 BF16 F32 | — |
| `matmul_int8` | F32 | F32 | — |
| `matmul_mixed` | F16 BF16 F32 | F16 BF16 F32 | — |
| `matmul_q` | — | Q5_0 Q8_0 Q5_K Q4_0 Q4_K Q6_K | — |
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
| `ne` | F16 BF16 F32 F64 U8 U32 I64 U16 I16 U64 I8 I32 | BF16 F32 F64 U8 I64 I32 | — |
| `neg` | F16 BF16 F32 F64 U8 I64 I32 I8 | F16 BF16 F32 F64 | F32 |
| `pad` | F32 F64 U8 I64 I32 I8 | F32 | — |
| `pow` | F16 BF16 F32 F64 U8 I64 I8 I32 | F32 F64 | F32 |
| `powf` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `prod` | F32 F64 I64 I32 | F32 F64 | F32 |
| `recip` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `relu` | F16 BF16 F32 F64 | F16 BF16 F32 F64 | F32 |
| `rmsnorm` | F32 F64 | BF16 F32 | F32 |
| `roll` | F32 F64 U8 I64 I32 I8 | F32 | — |
| `rope` | F32 F64 | BF16 F32 | — |
| `rope_i` | BF16 F32 F64 | BF16 F32 F64 | — |
| `rope_scaled` | F32 F64 | BF16 F32 | — |
| `round` | F16 BF16 F32 F64 | F32 F64 | F32 |
| `scatter` | F16 BF16 F32 F64 I64 I32 | F16 BF16 F32 F64 I64 I32 | — |
| `scatter_add` | F16 BF16 F32 F64 I64 I32 | F16 BF16 F32 F64 I64 I32 | — |
| `sigmoid` | F16 BF16 F32 F64 | F16 BF16 F32 F64 | F32 |
| `sign` | F16 BF16 F32 F64 U8 I64 I32 I8 | BF16 F32 F64 I64 I32 | — |
| `silu` | F16 BF16 F32 F64 | F16 BF16 F32 F64 | F32 |
| `sin` | F16 BF16 F32 F64 | F32 F64 | F32 |
| `softmax` | F32 F64 | BF16 F32 F64 | F32 |
| `sort` | F32 F64 I64 I32 | F32 F64 I64 I32 | — |
| `sqr` | F16 BF16 F32 F64 | BF16 F32 F64 | — |
| `sqrt` | F16 BF16 F32 F64 | F16 BF16 F32 F64 | F32 |
| `sub` | F16 BF16 F32 F64 U8 I64 I32 I8 | F16 BF16 F32 F64 | F32 |
| `sum` | F16 BF16 F32 F64 I64 I32 | F32 F64 | F32 |
| `tan` | F16 BF16 F32 F64 | F32 F64 | F32 |
| `tanh` | F16 BF16 F32 F64 | F16 BF16 F32 F64 | F32 |
| `topk` | F32 F64 I64 I32 | — | — |
| `tril` | F32 F64 | F32 | — |
| `triu` | F32 F64 | F32 | — |
| `upsample_bilinear2d` | F32 F64 | BF16 F32 | — |
| `upsample_nearest` | F32 F64 | BF16 F32 | — |
| `where` | F16 BF16 F32 F64 U8 I64 I32 I8 | BF16 F32 F64 U8 I64 I32 | — |

## Linear algebra

- [x] MatMul 2-D (CPU naive F32/F64/F16/BF16/I32/I64; CUDA cuBLAS Sgemm/Dgemm + `GemmEx` for F16/BF16 with the transposed-weight fast path and plain layout; auto-contiguifies strided operands) — F16/BF16 CUDA 2026-10-28
- [x] Batched MatMul (CPU F32/F64/F16/BF16; CUDA `cublasGemmStridedBatched(Ex)` F32/F16/BF16)
- [x] Mixed-precision MatMul (`matmul_mixed`: F32×{BF16,F16}→F32, BF16×BF16→BF16, F16×F16→F16; F32 accumulate via `cublasGemmEx`; CPU reference kernel)
- [x] **Quantized MoE grouped GEMM** (`ops.MoEGemmQ`: F32 activations × GGUF Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K expert stacks (E, N, K), fused dequant CUDA kernels in `moe_gguf.cu`, dequant + MoEGemm reference elsewhere; `nn.SparseMoE` runs quantized experts; GGUF `qwen3moe` loader fuses `ffn_gate_exps`/`ffn_up_exps` at the byte level) + `quant.Quantize` encoders for Q4_0/Q5_0/Q8_0 — 2026-10-28
- [x] **Quantized embedding lookup**: `IndexSelect` on every GGUF block format and the narrow fixed-width dtypes (byte-row gather, CPU + CUDA); `nn.Embedding` with a quantized table gathers the rows' blocks and dequantizes only those; `LoadLlamaGGUF` keeps `token_embd` quantized (tied head via `ops.MatMulQFused`) — 2026-10-29
- [x] **MoE grouped GEMM** (`ops.MoEGemm`, Candle `moe_gemm` contract: stacked (E, N, K) expert weights, expert-sorted (token, slot) pairs, optional routing weights, per-token or per-pair input rows; CUDA tiled kernel over all experts in one launch, CPU reference; F32/BF16/F16) + `ops.MoERoute` (top-k, `norm_topk_prob`, expert sort on device) + `ops.MoECombine` + `nn.SparseMoE` (router → gate/up grouped GEMM → act ⊙ up → down grouped GEMM, routing-weighted sum) — 2026-10-28, matches the per-expert index_select/index_add loop to 1e-5
- [x] **FP8 MatMul** (`ops.MatMulF8`: E4M3 × E4M3 with per-tensor scales → F32/BF16/F16; CUDA via cuBLASLt TN GEMM on tensor cores, K % 16 == 0; CPU reference; `ops.QuantizeF8` amax/448 quantiser; `nn.Linear` runs F8 weights with `WScale`) — 2026-10-28
- [x] Int8 weight MatMul (`ops.MatMulInt8`, per-column scales; CPU + CUDA F32 activations)
- [x] Einsum (`ops.Einsum`, general CPU; CUDA F32 fully on-device: diagonals via `tensor.AsStrided`, N-ary pairwise contraction, batched matmul; F64 on CUDA falls back to host)
- [x] MatMulQuant (GGUF quantized × float, fused dequant) — shipped as `ops.MatMulQ`, see Quantization

## Shape / layout ops

- [x] Cast (all dtype pairs on CPU; CUDA F16/BF16/F32/F64/I32/I64; WebGPU F32/U32/I32/I64)
- [x] Concat / Stack (all dtypes, all three devices)
- [x] Split / Chunk (`ops/split.go`)
- [x] IndexSelect, Gather, ScatterAdd, Scatter (set), IndexAdd, SliceScatter, SliceAssign (CPU + CUDA; gather/scatter/scatter_add/index_add at F32/F64/I32/I64/BF16/F16, CUDA accumulations atomic) — index ops widened 2026-10-28
- [x] Pad, Flip, Roll (CPU + CUDA F32)
- [x] Cumsum / Cumprod (CPU + CUDA F32)
- [x] Tril / Triu, mask ops, Where (CPU + CUDA)
- [x] Sort / TopK (CPU all; CUDA bitonic kernels F32/F64/I32/I64, bit-identical tie order; TopK = sort + narrow on CUDA)
- [x] Linspace / Meshgrid / Eye (`tensor/constructors.go`)
- [ ] Unique

## Normalization & activations

- [x] Softmax / LogSoftmax (fused, block reduction; CUDA F32/F64 + BF16 softmax)
- [x] LayerNorm, RMSNorm (fused; CUDA F32 + BF16)
- [x] GroupNorm (CUDA F32 + BF16; CPU F32/F64) — SD 1.5
- [x] InstanceNorm2d (`ops.InstanceNorm`, `nn.InstanceNorm2d`)
- [x] BatchNorm2d train (native CUDA stats + apply + backward) and eval (composed)
- [x] Dropout (train-mode inverted, CPU + CUDA)
- [x] GELU tanh-approx + exact-erf `ops.GeluErf` (BERT), GeGLU fused (CUDA), ReLU6 (composed)

## Convolution & pooling

- [x] Conv1d (CPU + CUDA F32 forward; no backward)
- [x] Conv2d (cuDNN on CUDA incl. BF16; forward + backward; grouped / depthwise)
- [x] Conv3d (CPU + CUDA F32 forward)
- [x] ConvTranspose2d (`ops.ConvTranspose2d`, `nn.ConvTranspose2d`; CUDA `conv_transpose.cu`, 2026-10-28)
- [x] MaxPool2d / AvgPool2d (forward + backward, CPU + CUDA F32)
- [x] AdaptiveAvgPool2d (composed on CPU/CUDA; native WebGPU)
- [x] UpsampleNearest (forward; CUDA F32 + BF16)

## Attention & transformers

- [x] RoPE (`ops.RoPE`, rotate-half / HF convention, pinned by `TestRoPEMatchesHFRotateHalf`; interleaved GPT-J pairing not supported) + frequency scaling (`ops.RoPEWithInvFreq`, `models.RopeScaling`: `llama3`, `longrope` with attention factor); CUDA F32 + BF16
- [x] SDPA (`nn.SDPA`) with causal, sliding-window and ALiBi masks
- [x] FlashAttention (`ops.FlashAttention`; CPU online softmax; CUDA tiled D∈{32,64,128} + generic ≤256). 16× faster than SDPA at S=1024.
- [x] SelfAttention / MultiHeadAttention / CrossAttention (autograd-aware, batched)
- [x] GroupedQueryAttention (in `models.Llama`), KVCache (prefill + decode)
- [x] TransformerBlock (pre-norm)
- [x] BF16 inference path (weights, activations, KV cache, scores, LM head; norms stay F32)
- [x] Sliding-window attention inside `models.Llama` (`LlamaConfig.SlidingWindow` + per-layer `SlidingLayers`, rotating KV cache; matches transformers `use_sliding_window` on Qwen2-0.5B to 5e-3; HF `sliding_window`/`layer_types`/`max_window_layers` and GGUF `attention.sliding_window` parsed)
- [x] Whisper speech-to-text (`models.Whisper`, `audio/` WAV + resampler + log-mel): encoder/decoder with cross-attention KV cache, language detection, token suppression; verified vs transformers on tiny (mel, encoder, logits, greedy). Example 41, `/v1/audio/transcriptions`, studio 🎤 button
- [x] Qwen3-ASR speech-to-text (`models.Qwen3ASR`): AuT audio encoder (Conv2d×3 stem over 1-second mel chunks, sinusoid chunk positions, 24-layer pre-LN encoder with 8-second attention windows run as batches), GELU projector, Qwen3 decoder fed through `Llama.ForwardEmbeds`; auto / forced language, hotword system prompt; verified vs transformers 5.21 on `Qwen/Qwen3-ASR-1.7B-hf` (filterbank, mel, encoder, projector, step-0 top-5, greedy). Example 42, same `/v1/audio/transcriptions` and studio 🎤 path
- [x] Qwen3 dense decoder (`LlamaConfig.QKNorm`: per-head RMSNorm on Q/K before RoPE; `model_type` qwen3 in `LoadCausalLMFromHF`), `Llama.Embed` + `Llama.ForwardEmbeds` for multimodal front ends
- [x] `audio.LogMelFrames` (any length) + `audio.MelFilterBank` (slaney, matches transformers' `mel_filter_bank`)
- [x] Candle-parity op batch (2026-10-28, CPU + CUDA kernels with parity tests): `Sign`, `Erf`, `Recip`, `Sqr`, `Mish`, `HardSwish`, `HardSigmoid`, `Clamp`, `ELU`, `LeakyReLU`, `PowScalar`, `Affine`, `RoundTo`, `Maximum`/`Minimum` (broadcast), compare/where on F64/I32/I64/U8/BF16 CUDA, `RoPEInterleaved` (GPT-J), `RoPETHD`, `Unfold`, `PixelShuffle`/`PixelUnshuffle`, `ReplicationPad2d`, `ConvTranspose1d/2d` (stride/padding/output_padding/dilation/groups), `UpsampleBilinear2d`
- [x] Autograd for the above: `Exp`, `Log`, `Abs`, `Sqr`, `ELU`, `LeakyReLU`, `Mish`, `HardSwish`, `HardSigmoid`, `Clamp`, `Maximum`, `Minimum`, `PReLU`, `Narrow`, `Concat`; losses `MSELoss`, `NLLLoss`, `BCEWithLogitsLoss`, `HuberLoss` (all grad-checked)
- [x] `nn`: ELU, LeakyReLU, Mish, HardSwish, HardSigmoid, GELU, ReLU6, PReLU, SwiGLU modules; `nn.LSTM` / `nn.GRU` (torch layout, match torch to 1e-5, trainable); `nn.ConvTranspose2d`
- [x] Gemma 3 text decoder (`models.Gemma3`): (1+w) RMSNorm, QK-norm, GELU-tanh MLP, local/global layer pattern, logit softcap; `models.CausalLM` interface shared with `models.Llama`
- [x] Chat engine (`engine/`): catalog of HF-cache + GGUF models, chat templates (ChatML, Llama-3, Phi-3, Mistral, Gemma), streaming generation, OpenAI-compatible `/v1/chat/completions`, `/v1/audio/transcriptions`, `/v1/embeddings`, speech-to-speech `/v1/audio/conversations` (`cmd/light-serve`), Wails v3 + React desktop app (`studio/`)
- [x] MoE gate / experts (`nn.SparseMoE`, `ops.MoEGemm` incl. quantized experts; Qwen3-MoE, 2026-10-28)

## Autograd

- [x] `Function` / `Context` / `FunctionNode` (PyTorch-style dynamic DAG), reverse-mode `Backward`, `GradCheck`
- [x] Functions: Add, Sub, Mul, Div, Neg, MatMul (batched), ReLU, ReLU6, Sigmoid, Tanh, SiLU, GELU, Sum, Mean, AxisSum, AxisMean, Reshape, Transpose, Permute, Contiguous, IndexSelect, LayerNorm, RMSNorm, Softmax, LogSoftmax, CrossEntropyLoss, Conv2d, Conv2dGrouped, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, BatchNorm2d, Dropout
- [x] `nn.LayerNorm` / `nn.RMSNorm` route through autograd only when a gradient is wanted (fixed 2026-10-27; had silently broken training through norms)
- [ ] Autograd for RoPE, GroupNorm, FlashAttention, Conv1d/Conv3d, Embedding gather on CUDA
- [ ] In-place safety via version counter
- [ ] `no_grad` scope
- [ ] Hooks

## NN modules

- [x] `nn.Module`, `nn.Sequential`, `nn.Linear` (+ `Tied`, `Uninit`, `WithDType`; F32/F64/BF16/F16/F8E4M3/GGUF weights), `nn.Embedding`
- [x] `nn.ReLU` / `SiLU` / `Sigmoid` / `Tanh`
- [x] `nn.LayerNorm`, `nn.RMSNorm`, `nn.GroupNorm`, `nn.InstanceNorm2d`, `nn.BatchNorm2d`, `nn.Dropout`
- [x] `nn.Conv2d` / `GroupedConv2d` / `DepthwiseConv2d` / `MaxPool2d` / `AvgPool2d` / `AdaptiveAvgPool2d` / `Flatten`
- [x] `nn.SelfAttention`, `nn.MultiHeadAttention`, `nn.CrossAttention`, `nn.SDPA`, `nn.TransformerBlock`
- [ ] `nn.ModuleList` / `nn.ModuleDict`

## Loaders

- [x] safetensors single-file + **sharded** (`model.safetensors.index.json`; reads HF Python cache layout), read + write, direct-to-CUDA
- [x] GGUF (read + minimal write; all 12 dtypes)
- [x] NumPy `.npy` / `.npz` read + write (`light/npy`: all NumPy numeric dtypes, Fortran order, stored or deflated members, lazy `Npz.Get`) — 2026-10-29
- [x] PyTorch `.pth` / `.pt` / `.bin` (`light/pth`: pickle VM for protocols 2–5, `_rebuild_tensor_v2` / `_rebuild_parameter` / `_rebuild_from_type_v2`, strided and offset views gathered contiguous, state dicts, checkpoint dicts by key, pickled `nn.Module` trees flattened to dotted names) — 2026-10-29
- [x] `nn.VarMap` (Candle VarMap): named tensor set, `Save` to safetensors/npz, `Load` in place from safetensors/npz/pth with shape checks, `Prefixed` views; `nn.ReadCheckpoint` dispatches on extension — 2026-10-29
- [x] HF Hub downloader (`hub.Download`, `hub.DownloadModelSafetensors`; `HF_TOKEN` / `HF_ENDPOINT`)
- [x] MNIST, CIFAR-10 (`light/data`) + augmentation (`RandomCropPad`, `RandomHorizontalFlip`, `NormalizeCIFAR10`)

## Tokenizers

- [x] BPE byte-level (GPT-2 / Llama-3 / Qwen), special tokens from `added_tokens` (+ `lstrip` / `rstrip`)
- [x] BPE SentencePiece-style (Llama-2 / Mistral / Phi-3: Prepend `▁` normalizer, rune-level merges, `<0xNN>` byte fallback) — auto-detected from `tokenizer.json`; ids identical to HF on Phi-3.5
- [x] CLIP tokenizer (SD 1.5)
- [x] WordPiece + BERT normalizer + `[CLS]`/`[SEP]` post-processor (exact match vs HF on bge-small)
- [x] Unigram / SentencePiece (Viterbi, byte fallback)
- [x] `tokenizer.Load` / `tokenizer.Parse` dispatch on `tokenizer.json` model type
- [x] **GGUF-metadata tokenizer** (`tokenizer.LoadGGUF` / `FromGGUFMeta`): `llama` (SentencePiece: score-ranked merges, `▁` prefix, byte fallback) and `gpt2` (byte-level BPE with llama.cpp's per-family pre-tokenizer regex chains for `tokenizer.ggml.pre` = default / llama-bpe / qwen2 / gpt-2 / deepseek-llm / deepseek-coder / falcon / starcoder-family / chatglm / poro …, `ignore_merges`), control + user-defined tokens as specials, BOS/EOS/add_bos; identical ids to llama.cpp on all 46 cases of its 10 vocab fixtures — 2026-10-29
- [x] `models.BOSTokenID` from `tokenizer.json` post_processor; `models.EOSTokenIDs` from generation_config / config
- [ ] Chat templates (Jinja2 subset) — Qwen / Llama-3 templates are hand-coded in the examples

## Quantization

- [x] Dequant Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q8_1 / Q2_K–Q8_K (CPU + CUDA parity)
- [x] Quantize Q4_0 / Q5_0 / Q8_0 (`quant.Quantize`, ggml reference rounding; round-trip tested) — 2026-10-28
- [x] Int8 per-column weight quant (`quant.QuantizeInt8PerColumn`) + int8 × F32 matmul
- [x] **GGUF quantized inference**: `ops.MatMulQ` (fused dequant-matmul CUDA kernels for Q4_0 / Q5_0 / Q8_0 / Q4_K / Q5_K / Q6_K; dequant+matmul fallback for the rest and on CPU), `nn.Linear` accepts quantized weights, `models.LoadLlamaGGUF` (llama / qwen2 / mistral GGUF, Q/K un-permute for llama arch, `rope_freqs` factors). Qwen2-0.5B Q8_0 / Q4_K_M and Llama-3.2-1B Q4_K_M match transformers greedy; logits within quant noise.
- [ ] MatMulQ performance: v1 kernel (warp per column, byte loads) decodes at ~17 tok/s on 0.5B vs 31 tok/s F32 cuBLAS — needs vectorised block loads / MMVQ-style int8 dot (Q8 activations) and an MMQ tile path for prefill
- [x] Tokenizer from GGUF metadata (`tokenizer.LoadGGUF`, 2026-10-29); quantized embedding lookup (`index_select` on GGUF blocks, 2026-10-29)
- [ ] IQ / TQ GGUF types (Candle lacks them too)
- [ ] GPTQ / AWQ loaders
- [ ] cuBLAS INT8 tensor-core GEMM
- [x] FP8 E4M3 tensor-core GEMM (cuBLASLt) + safetensors `F8_E4M3` load/save — 2026-10-28; blockwise (DeepSeek `weight_scale_inv`) scales not yet

## Models

See `MODELS.md` for the full catalogue. Validated against reference weights:

- [x] GPT-2 (HF `gpt2`) — greedy matches PyTorch; CPU, CUDA and WebGPU identical
- [x] Qwen2 / Qwen2.5 (0.5B, 3B; F32 + BF16; KV cache; chat)
- [x] Llama 3.x (`LoadLlama3FromHF`, `LoadCausalLMFromHF`; llama3 rope scaling; BOS/EOS) — Llama-3.2-1B-Instruct matches transformers top-1
- [x] BERT encoder (`LoadBertFromHF`; CLS/mean pooling) — bge-small-en-v1.5 max abs err < 1e-6 vs PyTorch
- [x] Stable Diffusion 1.5 (CLIP text + UNet + VAE decoder + DDIM) — BF16 txt2img
- [x] ResNet-50, MobileNetV2, ViT-B/16 from HF weights
- [x] MercyLM (small GPT, safetensors)
- [x] Gemma 2 (2B verified vs transformers 2026-10-29: softcaps 50/30, no QK-norm, alternating local/global; `Gemma3Config.NoQKNorm`)
- [x] Mimi neural audio codec (`models.Mimi`: causal SEANet encoder/decoder, sliding-window transformers with layer scale, split RVQ 1+31 codebooks, 12.5 Hz frames; verified vs transformers F32 2026-10-29 on codes and decoded audio; streaming `MimiStream` 2026-11-06 equals the full-sequence path frame by frame; first Moshiko component)
- [x] Moshi LM (`models.Moshi`: 7B temporal transformer + 8-slice depformer from the kyutai Q8 GGUF, `MoshiGen` full-duplex protocol, frame-by-frame `Converse` through the streaming Mimi; `engine.Speech` slot (releases the chat model), `POST /v1/audio/conversations` (WAV in → text + WAV reply, json or wav), Studio speech↔speech catalog kind + 🗣 Talk button; raw SentencePiece `.model` tokenizer; verified 2026-11-06 vs Rust moshi-core on the same Q8 GGUF: text top-5 and 47/48 codes over 6 frames; speech-to-speech works end to end)
- [x] Phi-4-mini (`LlamaConfig.PartialRotary`: partial rotary over the longrope table, tied head, GPT-4o tokenizer `Split` form; verified vs transformers BF16 2026-10-29; `phi4` chat template)
- [x] Falcon (`models.Falcon`: ALiBi or rotary, multi-head / multi-query / grouped K/V fused projections, sequential or parallel blocks with one or two LayerNorms; falcon-rw-1b verified vs transformers F32 2026-10-29, 7B/40B layouts on tiny-random checkpoints; pytorch_model.bin via `light/pth`)
- [x] EmbeddingGemma 2 (`models.EmbeddingGemma2`: bidirectional Gemma 4 style encoder, projection-only PLE, 512-dim single-KV-head global layers, bidirectional sliding radius, 768-dim projection, padding-aware mean pooling + L2 norm; verified vs transformers BF16 2026-10-29; `engine.Embedder` + OpenAI `/v1/embeddings`, Studio load/unload)
- [x] Gemma 4 text (`models.Gemma4`: per-layer embeddings, KV-shared layers, double-wide MLP, proportional RoPE on 512-dim global heads, V-norm, layer scalars; E2B verified vs transformers BF16 2026-10-29; per-tensor safetensors loader)
- [x] Gemma 3 4B text-only from the multimodal `gemma3` checkpoint (nested text_config, `language_model.` prefix, linear rope_scaling on global layers) — verified vs transformers BF16 2026-10-29
- [x] Random-init verified: VGG, YOLOv5, ResNet-CIFAR, ResNet9, LeNet, AlexNet
- [x] SDAR block-diffusion LM (`model_type` `sdar`; `Llama.ForwardDiffusion` block mask + no-store KV, `GenerateDiffusion` denoising loop, engine streaming per block) — SDAR-1.7B-Chat BF16 matches transformers top-5 and greedy decode (2026-10-28)
- [x] Qwen3-MoE (`LoadCausalLMFromHF` `qwen3_moe`; routed-expert FFN via `nn.SparseMoE` + `ops.MoEGemm`) — tiny-random checkpoint matches transformers 5.21 top-5 logits to 5 dp on CPU F32 / CUDA F32, BF16 within BF16 noise (2026-10-28)
- [x] Phi-3 / 3.5 (`LoadPhi3FromHF`; fused qkv/gate_up split, LongRoPE) — Phi-3.5-mini BF16 matches transformers BF16 logits; greedy identical
- [ ] Mistral (loader path exists via `LoadCausalLMFromHF`; untested, needs sliding window)
- [x] Whisper (`models.Whisper`, 2026-10-28); Gemma 2 is listed above
- [ ] T5, CLIP vision, SDXL

## Training

- [x] SGD (momentum + weight decay), AdamW, CosineLR (+ warmup)
- [x] Real recipes: MNIST 98.2%, CIFAR-10 ResNet-20 82.8%, ResNet9 90%+
- [ ] Gradient clipping, Adafactor, checkpoint save/load helper
- [ ] Mixed-precision + grad scaling
- [ ] LoRA, DataLoader

## Infra

- [x] Op dispatch registry (`internal/registry`), `scripts/opmatrix` coverage dump
- [x] PTX build (`scripts/build_ptx.sh`, compute_80, embedded, driver JIT); cuBLAS, cuDNN bindings
- [x] CUDA driver-context binding per thread (`cudart.ensureContext`, fixes intermittent code 201 under parallel tests)
- [x] CUDA default build; `-tags nocuda` opt-out; `-tags webgpu` opt-in
- [x] WebGPU runtime (`internal/wgpudev`, 16 WGSL shaders in `kernels/webgpu`)
- [x] Reference harness: `scripts/verify/` (numpy + transformers + BERT reference scripts)
- [ ] Multi-stream CUDA, CUDA graphs for decode, benchmark harness (Phase 10)

## Examples

01–25 as before (tensor basics → autograd training → GPT-2 pretrained). Added since:

- [x] `26_resnet50`, `27_mobilenetv2`, `28_vgg16`, `29_vit_b16`, `30_yolov5n` — random-init forward (+ train/infer splits for 26–29)
- [x] `31_vit_pretrained`, `32_resnet_pretrained`, `33_mobilenet_pretrained` — HF weights
- [x] `34_resnet9_cifar10` — augmented training recipe
- [x] `35_qwen_inference` — generic causal LM (`-repo`, `-bf16`, `-bos`, `-rope-scaling`); Qwen2 and Llama-3.2
- [x] `36_qwen_chat` — multi-turn with KV reuse
- [x] `37_sd15_txt2img` — Stable Diffusion 1.5
- [x] `38_mercy_chat` — MercyLM
- [x] `40_bert_embeddings` — sentence embeddings, cosine matrix
- [x] `25_gpt2_pretrained -device webgpu` — GPT-2 on WebGPU
