# Light — Status

What works today, with the evidence for it. Open work lives in
[`ROADMAP.md`](ROADMAP.md); the dated work log in
[`CHECKPOINTS.md`](CHECKPOINTS.md) is the source of truth, and every status
here agrees with it as of **2026-11-09**.

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
catalog lists them.

### 1.1 Language models

| Model | File | Status | Notes |
|-------|------|--------|-------|
| **GPT-2 (small / medium)** | `models/gpt2.go` | Done — pretrained | `LoadGPT2FromHF("gpt2", dev)` — 124.4M tied params; greedy matches PyTorch; CPU, CUDA and WebGPU identical; `examples/24_gpt2_forward`, `examples/25_gpt2_pretrained` (`-device webgpu`: "the capital of the French Republic, and", same as CPU; `TestGPT2WebGPUMatchesCPU` logits within 1.4e-4) |
| GPT-2 large / xl | `models/gpt2.go` | Architecture-ready | Same loader; the sharded safetensors loader it was waiting for exists now; not run |
| **Qwen2 / Qwen2.5** | `models/llama.go` | Done — pretrained | Generic LLaMA-arch decoder (RoPE + RMSNorm + SwiGLU + GQA + KV cache + sharded safetensors + BF16 / F32); `LoadQwen2FromHF`, `LoadQwen2FromHFMixed`; `examples/35_qwen_inference`, `examples/36_qwen_chat` (multi-turn with KV reuse). Qwen2-0.5B F32 top-5 matches transformers. GGUF Q8_0 / Q4_K_M via `LoadLlamaGGUF` (greedy matches transformers, logits within quant noise). **GPTQ-Int4 / AWQ** (2026-11-08): Qwen2.5-0.5B-Instruct GPTQ and AWQ on CPU F32 vs transformers on the independently dequantized weights: top-5 within 1e-3, greedy 16 / 16; A100: CUDA F32 top-5 = CPU, CUDA BF16 greedy 16 / 16. **WebGPU** (2026-11-08, Intel iGPU): F32 and Q4_K_M GGUF prefill logits within 6.4e-5 / 1.0e-4 of the CPU, 12 greedy tokens identical; GPTQ / AWQ also run there. BF16 Linear + bias rounds once in F32 (2026-11-08), as transformers does |
| **Qwen3 (dense)** | `models/llama.go` (`QKNorm`) | Done as a decoder body | QK-norm, no QKV bias, rope_theta from `rope_parameters`. Verified as the Qwen3-ASR decoder and the SDAR body; standalone Qwen3-0.6B/1.7B/4B load via `LoadCausalLMFromHF` (`qwen3`) but standalone logits are not yet checked (ROADMAP) |
| **Qwen3-MoE** | `models/llama.go` (`LlamaConfig.NumExperts…`, `llamaBlock.MoE`), `qwen_load.go` (`mapHFMoE`), `nn/moe.go` | ✅ Verified 2026-10-28 (architecture) | `LoadCausalLMFromHF` (`qwen3_moe`): QK-norm attention + routed FFN on every `decoder_sparse_step` layer (`mlp_only_layers` stay dense), `norm_topk_prob`, both HF expert layouts (transformers-5 fused `experts.gate_up_proj`/`down_proj`, hub per-expert `experts.{e}.*`), router `gate.weight`/`router.weight`. On `hf-internal-testing/tiny-random-Qwen3MoeForCausalLM` (2 layers, 4 experts, top-2, 20-token prompt) CPU F32 and CUDA F32 reproduce transformers 5.21's top-5 ids and logits to 5 decimals; CUDA BF16 matches the BF16 reference. GGUF `qwen3moe` files load via `LoadLlamaGGUF` with quantized experts kept quantized (`ops.MoEGemmQ`; `attn_q_norm`/`attn_k_norm` and `ffn_gate_inp` handled): a Q8_0 / Q4_0 round trip of the tiny checkpoint reproduces the reference logits within 1e-3 / 1e-2 on CPU and CUDA. **4-bit (GPTQ / AWQ) MoE experts** (2026-11-08, `ops.MoEInt4`, only the routed experts dequantized): tiny GPTQ-layout qwen3moe fixture within 2e-4 of transformers, greedy 8 / 8. Qwen3-30B-A3B Q4_K_M is ≈18 GB, above the 16 GB card |
| **Qwen3.5 / 3.6 / 3.8 text (Gated DeltaNet hybrid)** | `models/qwen3_5.go`, `qwen3_5_load.go`, `qwen3_5_gguf.go`; `ops.GatedDeltaRule`, `kernels/cuda/src/gated_delta.cu` | ✅ Verified 2026-11-08 (0.8B, 9B, 27B BF16 / FP8 / GGUF) | `LoadQwen35FromHF` (`qwen3_5` / `qwen3_5_text`; vision tower and `mtp.*` skipped), `LoadQwen35GGUF` (llama.cpp `qwen35`). Hybrid stack: Gated DeltaNet mixer (`in_proj_qkv` → causal depthwise conv + SiLU, β = sigmoid(b), g = −exp(A_log)·softplus(a + dt_bias), gated delta rule in F32 with in-kernel q/k L2 norm, gated RMSNorm · silu(z)) + gated full attention (per-head query gate, QK-norm, partial RoPE) + SwiGLU MLP; zero-centred norms. Ops: recurrent + chunked delta rule on CPU (golden fixtures from transformers within 1e-5), CUDA recurrent kernel and a chunked 16-token-block prefill kernel for 128 × 128 heads (2026-11-07). **0.8B** (2026-11-06): F32 top-5 within 1e-4 and 16-token greedy identical on CPU, BF16 within one BF16 step; engine (2026-11-07) with the model's own Jinja template: prompt ids and 24-token greedy reply identical to transformers. **9B** BF16 on an A100 (2026-11-07): top-5 ids equal, greedy 16 / 16, chunked vs recurrent prefill top-5 within 0.125. **27B** (2026-11-08, A100 80 GB): FP8 blockwise checkpoint (`Qwen/Qwen3.5-27B-FP8`, 128 × 128 `weight_scale_inv`, fused `fp8_block` GEMV / dequant kernels) top-5 ids equal to transformers BF16, greedy 16 / 16; GGUF Q4_K_M (Q4_K / Q5_K / Q6_K / Q8_0, tiled value-head order `VHeadsTiled`) top-5 ids equal, greedy 16 / 16. GGUF loads through the engine (`LoadChatModelGGUF`). CPU 8.9 tok/s decode on the 0.8B (laptop, after the CPU pass); **WebGPU** (2026-11-09): Gated DeltaNet / causal conv / softplus in WGSL, 0.8B logits within 3.8e-5 of the CPU, greedy equal |
| **Llama-3 / 3.1 / 3.2** | `models/llama3_load.go` | Done — pretrained | `LoadLlama3FromHF` / `LoadCausalLMFromHF` (dispatch on `model_type`); llama3 RoPE scaling (`models/rope_scaling.go`); BOS from `tokenizer.json`, EOS list from `generation_config.json`. Llama-3.2-1B-Instruct: F32 top-5 matches transformers to 4 dp, " Paris", "The answer is 4."; F32 29 tok/s, BF16 34 tok/s; GGUF Q4_K_M same top-3 as F32, greedy identical. `examples/35_qwen_inference -repo meta-llama/Llama-3.2-1B-Instruct`. **GPTQ act-order** (2026-11-08): TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ (desc_act) on CPU F32 top-5 within 1e-3 of transformers, greedy 16 / 16 |
| **Mistral 7B** | `models/llama3_load.go` | ✅ Verified 2026-11-09 (BF16) | `mistralai/Mistral-7B-Instruct-v0.3` (`mistral`: GQA 32 / 8, rope θ 1e6, vocab 32768) in BF16 on the CPU vs transformers BF16: top-5 ids equal, 16 / 16 greedy. Mixtral (MoE) loader not wired |
| **Granite 3.x / Granite 4.0 (attention)** | `models/granite_load.go` | ✅ Verified 2026-11-09 (granite-4.0-micro BF16) | `LoadGraniteFromHF` (`granite`, `granitemoehybrid` with attention-only layers): Llama decoder plus embedding / attention / residual / logits multipliers (F32 then one rounding). granite-4.0-micro (3B, 40 layers) BF16 vs transformers: 16 / 16 greedy. Mamba-hybrid Granite 4.0 layers not supported |
| **Ministral 3 (text)** | `models/ministral_load.go` | ✅ Verified 2026-11-09 (3B, FP8 → BF16) | `LoadMinistralFromHF` (`mistral3` wrapper, Pixtral tower skipped, or bare `ministral3`): YaRN RoPE, Llama 4 position-dependent query scaling, per-tensor FP8 weights dequantized to BF16 at load as transformers does. Ministral-3-3B-Instruct-2512 BF16 vs transformers: 16 / 16 greedy |
| Mixtral | `models/llama3_load.go` | Architecture-ready (SWA done) | `LoadCausalLMFromHF` accepts `model_type: mistral`; sliding-window attention + rotating KV cache verified against transformers on Qwen2 (window 8). A Mistral 7B checkpoint has not been run yet; HF `mixtral` is not routed (the `nn.SparseMoE` block and the GGUF MoE metadata path for mixtral converters exist). Both are ROADMAP items |
| **Phi-3 / Phi-3.5** | `models/llama3_load.go` (`LoadPhi3FromHF`) | Done — pretrained | Llama template + fused `qkv_proj` / `gate_up_proj` split + LongRoPE (`models.RopeScaling{Type: "longrope"}`) + SentencePiece BPE. `microsoft/Phi-3.5-mini-instruct` BF16: top-5 logits match transformers BF16; "What is 2+2?" → "The sum of 2 and 2". 7.6 GB, 15 tok/s. `examples/35_qwen_inference -repo microsoft/Phi-3.5-mini-instruct -bf16` |
| **Phi-4-mini** | `models/llama.go` (`PartialRotary`), `llama3_load.go` | ✅ Verified 2026-10-29 | `LoadCausalLMFromHF` (`model_type: phi3`): Phi-3 stack plus `partial_rotary_factor` 0.75 (RoPE rotates the first 96 of 128 head dims with 48 longrope factors, the rest pass through), tied LM head, GPT-4o 200k vocabulary. `microsoft/Phi-4-mini-instruct` BF16 on CUDA vs transformers 5.21 (`ref_phi4.py`, 21-token system+user chat prompt): top-5 ids {976 37513 15952/15957 30112} with the reference's 'Blue'/'Light' tie at 40.5, logits 47.50/42.25/40.50/40.25/40.25 vs 47.75/42.5/40.5/40.5/40.25, 16-token greedy identical (`The sky is blue due to Rayleigh scattering, where shorter blue wavelengths of sunlight`); 3.84 B params, 7.7 GB. Tokenizer: tokenizer.json `Split` with behavior `Removed`+`invert` (GPT-4o form) now honoured; ids identical to HF on chat markers, mixed Unicode and code. Engine: `phi4` chat template (`<\|role\|>…<\|end\|>`, no newlines; detected by the `<\|tool\|>` marker Phi-3.5 lacks), EOS 200020/199999. Candle's `phi3` has no partial rotary |
| Phi-4 (14B) | — | Architecture-ready | Same template (no LongRoPE); 14B needs 4-bit to fit 16 GB |
| **Gemma 3 (text)** | `models/gemma.go`, `gemma_load.go` | ✅ Verified 2026-10-28 | `LoadGemma3FromHF` (`gemma3_text`). √D embed scale, (1+w) RMSNorm, 4 norms/block, QK-norm, GELU-tanh gated MLP, 5:1 local(512, θ=1e4)/global(θ=1e6) layers, rotating cache. gemma-3-1b-it F32 top-5 logits match transformers to 4 dp (54.9099 vs 54.9100); BF16 greedy "2 + 2 = 4" identical; 33 tok/s. Weights from the ungated `unsloth/gemma-3-1b-it` mirror (google/ repo is gated). **4B** (2026-10-29): the multimodal `gemma3` checkpoint loads text-only (nested `text_config`, `language_model.model.*` keys, linear `rope_scaling` ×8 on global layers, SigLIP tower skipped); `unsloth/gemma-3-4b-it` BF16 on CUDA matches transformers: identical top-5 ids (56.0/34.0/33.0/32.5/32.0 vs 55.25/33.25/32.5/32.25/31.5), greedy `2 + 2 = 4\n<end_of_turn>` identical. 12B/27B use the same path but exceed 16 GB at BF16. Re-verified on the A100 after the Muse Glimmer switches (2026-11-08). Vision input open (ROADMAP) |
| **Gemma 2** | `models/gemma.go`, `gemma_load.go` | ✅ Verified 2026-10-29 | Same stack as Gemma 3 with `NoQKNorm`, attention softcap 50, final logit softcap 30, alternating local(4096)/global layers, RoPE base 1e4 everywhere. `unsloth/gemma-2-2b-it` F32 on CUDA matches transformers to 4 dp (21.2942 vs 21.2943; all top-5 ids identical), greedy `2 + 2 = 4 \n<end_of_turn>` identical; BF16 through the engine gives the same reply. 9B/27B load through the same path (27B exceeds 16 GB) |
| **Gemma 4 (text)** | `models/gemma4.go`, `gemma4_load.go` | ✅ Verified 2026-10-29 | `LoadGemma4FromHF` (`gemma4` multimodal checkpoints text-only, `gemma4_text`). Per-layer embeddings (262144×L·256 table + projected context, RMS-normed, mixed /√2, gated into each block), KV sharing across the last `num_kv_shared_layers` layers (reuse the last non-shared layer of the same type, no k/v weights), double-wide MLP on shared layers, global layers with head dim 512 and proportional RoPE (first 25 % of the pairs rotate, θ=1e6; local θ=1e4 full), attention scaling 1 (q_norm carries it), unweighted RMS norm on V, per-layer `layer_scalar`, direct RMSNorm weights, final softcap 30. Tensors are loaded one at a time straight to the device (no 10 GB host staging). `unsloth/gemma-4-E2B-it` BF16 on CUDA vs transformers: top-5 ids identical [236778 236795 818 7925 100], logits 21.85/18.29/16.98/12.19/9.93 vs 22.13/18.0/17.0/12.75/9.88, greedy `2 + 2 = **4**<turn|>` identical; 4.63 B text params. E4B loads the same way (≈16 GB BF16, over the card); 12B/31B dense would need a quantized path; 26B-A4B MoE blocks and the vision/audio towers are open (ROADMAP) |
| **Muse Glimmer 30B (text)** | `models/muse_glimmer_load.go`, `muse_glimmer_gguf.go` (on the `Gemma3` skeleton) | ✅ Verified 2026-11-08 (30B GGUF Q4) | `LoadMuseGlimmerFromHF` (`muse_glimmer` / `muse_glimmer_text`, vision tower skipped) and `LoadMuseGlimmerGGUF` (llama.cpp `muse-glimmer`). Differences from Gemma 3 as `Gemma3Config` switches (all zero for Gemma 2 / 3): `EmbedNorm` (weightless RMSNorm after the lookup instead of √D), `LayerRopeBase` (per-layer θ; 0 = NoPE on the global layers), `AttnOutputGate`, `QScale` (q × 3.87 after a weightless qk-norm, in F32), `SiLUMLP`, `PostNormEps` 1e-8, `OutputMultiplier` 0.196 before softcap 20, `UntiedHead`; GQA 32/2, local/local/local/global with a 2048 window. Tiny-random checkpoint (NoPE global layer, window 6 on 18 tokens) within 2e-4 of transformers F32, greedy 8 / 8, chunked = single shot; tiny GGUF conversion equals the HF load exactly in F32, within 4.8e-3 at Q8_0. **30B** unsloth UD-Q4_K_XL (15.9 GB, fits an L4) on an A100: top-1 " to" 19.09 vs 18.125 for transformers BF16 (runners-up reorder under Q4, within 1.5), greedy 16 / 16. Engine: Muse's harmony-style chat template + `llama4` pre-tokenizer give exactly transformers' 43 prompt ids; `<\|eot\|>` (EOT token) joins the stop set. A full engine generation with the 30B GGUF is pending a GPU run |
| **Mellum 2.1 12B-A2.5B** | `models/llama3_load.go` (`model_type` `mellum`, Qwen3-MoE path) | ✅ Verified 2026-11-07 (12B BF16, A100) | Qwen3-MoE path (QK-norm, softmax top-k router, `norm_topk_prob`, 64 experts top-8) plus: **YaRN** RoPE (`RopeScaling{Type: "yarn"}`, table equals transformers to 1e-6, attention factor kept in F32), **per-layer-type RoPE** (`SlidingRopeBase` / `SlidingRopeScaling`), `mlp_layer_types` dense layers, `num_local_experts`, attention width ≠ hidden size (NH·HD 4096 on a 2304-wide model). Tiny-random checkpoint (window 6, YaRN on full layers, a dense layer) top-5 to 4 dp and 8-token greedy identical on CPU (2026-11-06). `TestMellum12BMatchesTransformersBF16` (24.3 GB): greedy 16 / 16 equal to transformers BF16; logits closer to transformers F32 than transformers' own BF16 run (<think> 7.94 vs F32 7.78, BF16 8.44) |
| **DeepSeek-V2 / V3** | `models/deepseek_v2.go`, `deepseek_v2_load.go`, `deepseek_route.go` | ✅ Verified 2026-11-08 (V2-Lite-Chat, A100); V3 routing tiny-verified | `LoadDeepseekV2FromHF` (`deepseek_v2`, `deepseek_v3`). **MLA** in the expanded form: q from q_proj or q-LoRA (q_a_proj → RMSNorm → q_b_proj), split into no-RoPE / RoPE parts; kv_a_proj_with_mqa → RMS-normed KV latent + one shared RoPE key; kv_b_proj expands per head each step; the KV cache holds only the latent and the rotated shared key. Interleaved RoPE rows reordered to rotate-half on load (skipped when `rope_interleave` is false); YaRN with mscale (softmax scale × mscale²) in F32. **DeepSeekMoE** on `nn.SparseMoE` (`RouterF32`, `routed_scaling_factor`, shared experts, dense first layers). **Routing** (2026-11-08): greedy, V2 `group_limited_greedy`, and V3's sigmoid router with `e_score_correction_bias` (bias for the choice only, group score = sum of the two best, weights renormalised) via `nn.SparseMoE.Router` + `ops.MoERouteFromIDs`. Tiny-random checkpoints (q_proj and q-LoRA, YaRN + mscale, shared expert; V2 group-limited, V3 interleaved and rotate-half) within 2e-4 of transformers F32, greedy 8 / 8 through the latent cache, chunked = single shot (the routing variants on CPU; CUDA run pending). **DeepSeek-V2-Lite-Chat** BF16 on an A100: 15.706 B params, top-5 ids equal to transformers BF16 (logits within one BF16 step), greedy 16 / 16. V3's MTP layer is not loaded; full-size V2 / V3 are out of reach for a real-weight run |
| **GLM-4 (glm, glm4)** | `models/glm_load.go` | ✅ Verified 2026-11-08 (GLM-4-9B-0414, A100) | `LoadGLMFromHF` loads THUDM/glm-4-9b-chat-hf (`glm`) and GLM-4-0414 (`glm4`) as a `*Llama`: partial RoPE over half the head dim in interleaved pairs (rotary rows of q_proj / k_proj weights and biases reordered to rotate-half on load), Q/K/V biases, fused gate_up_proj split, and for glm4 two post-norms per layer (`LlamaConfig.PostNorms`). Tiny glm / glm4 within 2e-4 of transformers F32, greedy 8 / 8, chunked = single shot, CPU and CUDA. **GLM-4-9B-0414** BF16: 9.400 B params, top-5 ids equal to transformers BF16 (32.25 / 17.875 / 16.5 / 13.19 / 13.0 vs 32.0 / 17.75 / 16.5 / 13.25 / 13.0), greedy 16 / 16. `glm` (glm-4-9b-chat-hf) is checked on the tiny model only |
| **Nemotron** | `models/nemotron_load.go` | ✅ Verified 2026-11-08 (Nemotron-4-Mini 4B, A100) | `LoadNemotronFromHF` (`nemotron`: Nemotron-4 / Minitron / Nemotron-Mini) as a `*Llama` with `LayerNorm1P` (mean-centred LayerNorm with bias, effective weight 1 + w added in F32) and `ReLU2MLP` (down(relu(up(x))²)); partial rotate-half RoPE (default 0.5); head dim from head_dim / kv_channels. Tiny-random within 2e-4 of transformers F32 (BF16 within 0.05 of F32), greedy 8 / 8. **nvidia/Nemotron-4-Mini-Hindi-4B-Instruct** BF16: 4.191 B params, top-5 ids equal to transformers BF16, greedy 16 / 16. Nemotron-Mini-4B-Instruct / Minitron-4B ship only pytorch_model.bin, which this loader does not read |
| **MiMo (Xiaomi MiMo-7B)** | `models/llama3_load.go` (`mimo` → Qwen2 loader), `models/mimo_mtp.go`, `models/spec_decode.go` | ✅ Verified 2026-11-08 (MiMo-7B-RL, A100) | MiMoForCausalLM = Qwen2 + `model.mtp_layers` (multi-token prediction). MiMo-7B-RL F32 vs transformers F32: top-5 identical, logits within 1e-3; BF16 greedy 16 / 16 vs transformers BF16 (BF16 attention is very sensitive here: k_proj biases up to 55; Light's single-rounding bias makes BF16 closer to F32 than transformers' BF16). **MTP speculative decoding** (2026-11-08): `models.MTPHead` + `Llama.MTPDraft`, `models.SpecGreedy` (draft one token, verify [x, d] in one forward, `KVCache.Truncate` on reject; output equals plain greedy); draft logits within 2e-4 of a PyTorch re-expression of the MTP layer (CPU); engine routes greedy requests on MTP models through `generateSpec` (`Stats.Drafted / Accepted`). The GPU speed / acceptance run (`TestGenerateMiMoSpeculativeBF16`) is pending |
| **MiniMax-Text-01** | `models/minimax.go`, `minimax_load.go` | ✅ Tiny-verified 2026-11-08 | `LoadMiniMaxFromHF` (`minimax`). **Lightning attention**: per-head d × d state with fixed decay exp(−slope), slopes from the config, q / k / v = SiLU(qkv_proj), RMSNorm over all heads, sigmoid output gate; run in transformers' block algorithm (chunks of 256, closed form inside a chunk, state carried across; O(T·C + (T/C)·d²) work, all batched matmuls, so CUDA needs no dedicated kernel). Softmax layers: GQA, full-head RoPE. DeepNorm-style residuals (alpha · norm(x) + beta · f(norm(x)) in F32). MoE in every layer (Mixtral-style w1 / w3 / w2 experts). Tiny-random (alternating lightning / softmax, block size 4 crossing five blocks, residual factors ≠ 1, 4 experts top-2): logits within 2e-4 of transformers F32, greedy 8 / 8, uneven chunked prefill = single shot; chunk sizes 1 / 4 / 5 / 7 equal one chunk within 1e-5 (CUDA run pending). The real model (456 B) is out of reach |
| **Falcon** | `models/falcon.go`, `falcon_load.go` | ✅ Verified 2026-10-29 | `LoadFalconFromHF` (`model_type: falcon`, also the legacy `RefinedWebModel`/`RefinedWeb`): every transformers decoder variant — falcon-rw-1b (ALiBi, multi-head, sequential attention→MLP, biases), falcon-7b (rotary, multi-query, parallel attention+MLP on one LayerNorm, no bias) and the 40B/180B "new decoder architecture" (grouped K/V in `(kv, g+2, hd)` fused layout, `ln_attn`+`ln_mlp`). Fused `query_key_value` split per layout, exact-erf GELU, tied or separate `lm_head`, safetensors or `pytorch_model.bin` (via `light/pth`), local checkpoint directories. ALiBi slopes follow `build_alibi_tensor` (slope·position rounded in BF16 as transformers does), folded into the causal mask with the 1/√d scale. `tiiuae/falcon-rw-1b` F32 on CUDA vs transformers: top-5 ids and logits identical to 4 dp (35.2608 / 31.7368 / 30.4928 / 30.3152 / 29.2224), greedy identical (` Rome. The capital of the United Kingdom`); tiny-random checkpoints for the 7B and 40B layouts (`models/testdata/falcon`, `hf-internal-testing/tiny-random-FalconForCausalLM`) match to 2e-3 on CPU/CUDA F32 and BF16, chunked decode = single shot. Note: transformers' *eager* Falcon adds ALiBi twice (inside attention and again folded into the mask); Light matches its SDPA path and the original Falcon code. Candle's falcon.rs only handles the 7B layout. Engine: catalog lists torch-zip checkpoints (`kind: pytorch`), 29 tok/s BF16 on the 5060 |
| **SDAR-1.7B-Chat (block-diffusion LM)** | `models/sdar.go` (`Llama.ForwardDiffusion`, `GenerateDiffusion`, `SDAROpts`), `models/llama.go` (block-bidirectional mask, no-store KV pass), `engine/engine.go` (diffusion branch) | ✅ Verified 2026-10-28 | First diffusion LLM: `JetLM/SDAR-1.7B-Chat` is a Qwen3-1.7B body (`model_type` `sdar`, QK-norm) decoded by blocks of 4 `<\|MASK\|>` tokens — bidirectional attention inside a block, causal across blocks, up to `DenoisingSteps`+1 passes per block committing the most confident predictions (static schedule or `low_confidence_dynamic` threshold), KV cache stored once per finished block. On CUDA BF16 the first denoising pass reproduces transformers' top-5 ids with logits within BF16 noise (53.0 vs 54.0) and greedy block decoding of "What is 2+2?" gives the identical `2 + 2 = 4<\|im_end\|>`. Engine `Generate` streams per finished block. Larger SDAR variants (4B/8B, b8–b64 block sizes, 30B-A3B MoE) use the same loader with `BlockLength` set accordingly |
| **MercyLM** | `models/mercy.go` | Done — pretrained | 15.4 M GPT-style LM from safetensors; `examples/38_mercy_chat` |
| **BERT** | `models/bert.go`, `models/bert_load.go` | Done — pretrained | `LoadBertFromHF` (encoder, CLS/mean pooling, L2 normalize) + `BertForMaskedLM` (shape-tested only). `unsloth/bge-small-en-v1.5` matches PyTorch to 7.8e-7; WordPiece ids identical. `examples/40_bert_embeddings` |
| **EmbeddingGemma 2** | `models/embeddinggemma2.go`, `embeddinggemma2_load.go` | ✅ Verified 2026-10-29 | `LoadEmbeddingGemma2FromHF` (`embedding_gemma2`, text encoder only; vision/audio towers skipped). Gemma 4 style blocks (shared `gemma4Block`) run bidirectionally: projection-only per-layer embeddings (norm(proj(h)·D^-½), no PLE token table, gated into every block), 4 query heads with 256-dim 2-KV-head sliding layers and 512-dim single-KV-head full-attention layers (every 6th; `per_layer_config`), full-width RoPE per type (θ 1e4 local / 1e6 global), attention scaling 1 with q/k norm and unweighted V norm, direct RMSNorm weights, `layer_scalar`, sliding layers attend within \|i−j\| ≤ 512 in both directions, final norm → 512→768 `embedding_projection`; `Embed` mean-pools (padding-aware) and L2-normalises (sentence-transformers 1_Pooling + 2_Normalize). `unsloth/embeddinggemma-2` vs transformers 5.21 BF16 (`ref_embeddinggemma2.py`): first 8 dims and the 3×3 cosine matrix within 1.5e-2 on CUDA BF16 and CPU F32 (cos(query, doc) 0.853 vs 0.854; off-topic 0.593), padded batch = unpadded, CPU↔CUDA F32 within 2e-3; 271 M text params. Task prefixes (`task: search result \| query: `, `title: none \| text: `) are the caller's. Engine: `/v1/embeddings` (`engine.Embedder`), catalog kind `embedding`. Candle has no EmbeddingGemma port |

### 1.2 Speech and audio

| Model | File | Status | Notes |
|-------|------|--------|-------|
| **Whisper (tiny…large-v3)** | `models/whisper.go`, `whisper_load.go`, `audio/` | ✅ Verified 2026-10-28 | `LoadWhisperFromHF`; host log-mel identical to `WhisperFeatureExtractor` (filters from preprocessor_config.json), encoder (Conv1d×2 + sinusoid positions), decoder with self-KV + cross-KV cache, greedy with `suppress_tokens`, language auto-detect. whisper-tiny: mel/encoder values and step-0 top-5 logits match transformers F32; "Front Center." greedy identical; 104 ms encode + 78 ms decode on the GPU. Larger sizes untested but same layout. `examples/41_whisper`, `/v1/audio/transcriptions` |
| **Qwen3-ASR (1.7B / 0.6B)** | `models/qwen3_asr.go`, `qwen3_asr_load.go` | ✅ Verified 2026-10-28 | `LoadQwen3ASRFromHF(repo, dev, dt)`; 128-bin slaney log-mel built in Go, Conv2d×3 chunk stem (100 mel frames → 13 tokens), 24-layer encoder with 104-token attention windows, GELU projector into the Qwen3-1.7B decoder (BF16) via `ForwardEmbeds`; auto / forced language, hotword system prompt. On `Qwen/Qwen3-ASR-1.7B-hf`: filterbank/mel/encoder/projector values and step-0 top-5 match transformers 5.21 F32; greedy "language English<asr_text>Front, center." identical, forced-language prefill identical. 29 ms encode + 194 ms decode (7 tokens) on the GPU after warm-up. `examples/42_qwen3_asr` |
| **Mimi (codec)** | `models/mimi.go`, `mimi_load.go` | ✅ Verified 2026-10-29 | `LoadMimiFromHF` (`kyutai/mimi`, 96 M params, F32). Causal SEANet encoder (conv 7 → 4× [resnet, ELU, strided conv 2r/r] → ELU → conv 3), 8-layer causal transformer (RoPE, sliding window 250, pre-LayerNorm with bias, layer scales, exact-erf GELU MLP), stride-2 replicate-padded downsample to 12.5 Hz, split RVQ (1 semantic + 31 acoustic Euclidean codebooks, 2048×256, `embed_sum / cluster_usage`), grouped transposed-conv upsample, decoder transformer, SEANet decoder with trimmed causal transposed convs. `Encode(audio, n)` → codes (B, n, T) I64, `Decode(codes)` → audio. On `testdata/mimi/front_center_24k.wav` (1.43 s) vs transformers MimiModel F32: 141/144 codes identical (the 3 differ on a 5e-4 near-tie where torch's matmul-form cdist rounds the other way), decoded audio first 8 samples within 1e-3, RMS 0.06754 vs 0.06738, peak 0.44606 vs 0.44597; per-stage activations match to ≤3e-5 (`TestMimiStagesAgainstDump`); CPU = CUDA. Streaming (2026-11-06): `Mimi.NewStream()` → `EncodeStep` (k·1920 samples → k frames) / `DecodeStep` (k frames → k·1920 samples) with kyutai's streaming states (conv memories of kernel−stride inputs, zero or first-sample for replicate pads; transposed-conv overlap carry minus bias; sliding-window KV caches); `TestMimiStreamingMatchesFullSequence`: 136/136 codes identical and audio within 1e-6 of the full-sequence path for 1- and 3-frame chunks, CPU and CUDA. Candle has a Mimi port too (parity) |
| **Moshi (speech-to-speech LM)** | `models/moshi.go`, `moshi_load.go`, `moshi_converse.go`, `tokenizer/spm_model.go` | ✅ Verified 2026-11-06; real time on an L4 2026-11-08 | `LoadMoshiFromHF` / `LoadMoshiGGUF` (`kyutai/moshiko-candle-q8` model.q8.gguf, 7.69 B params Q8_0 + F32 norms, CUDA fused MatMulQ; 8.2 GB VRAM). Temporal transformer (32 layers, d 4096, 32 heads, fused QKV, interleaved RoPE θ 1e4 in F32, RMSNorm eps 1e-8 with `alpha` weights, SiLU-gated MLP split in halves, sliding context 3000 keys) over text + 16 Mimi codebooks (8 own + 8 user), `text_linear` head, 8-slice depformer (6 layers, d 1024, no positional embedding, per-slice weights, one shared KV cache per frame; slice 0 embeds the text token, slice s the code of slice s-1; heads of width 2048 so pad is never emitted). `MoshiGen` runs kyutai's multistream protocol (acoustic delay 2 with forced pads, user codes lagged one frame, text fed back); `Converse(mimi, audio, extra)` runs a turn frame by frame through the streaming Mimi (80 ms in → step → 80 ms out), as the live system does. Raw SentencePiece unigram text tokenizer (`tokenizer.LoadSentencePieceModel`, byte fallback) verified vs the sentencepiece package. Tiny-random: full-sequence = stepwise through the cache (window via mask and trimming), CPU = CUDA. Real checkpoint: `TestMoshikoConverse` on the Front Center clip answers "Hello, what's up?" with 3.3 s of speech (greedy). Vs Rust moshi-core on the same GGUF (`~/workspace/experimental/rust/moshi-ref`, CPU, argmax, 6 frames of LCG user codes): text top-5 identical every frame with logits within 0.15, 47/48 depformer codes identical (one 0.16-logit near-tie in slice 7 of frame 1; Candle's CPU path quantizes activations to Q8). Engine: `engine.Speech` (`LoadSpeech`, `Converse` with kyutai's default top-k sampling or greedy), runtime speech slot that releases the chat model, catalog kind `speech` (GGUF with depformer tensors, no config.json), `POST /v1/audio/conversations`. **Perf pass (2026-11-08)**: ring KV cache (`ops.KVRingWrite`) + split-K decode attention (`ops.AttnDecode`, CUDA `kv_ring.cu`); on an L4 the model loop runs at 61.9 ms per 80 ms frame (LM forward 34.2 ms, near the bandwidth floor) and speech-to-speech through the engine at **58 ms / frame, 1.4× real time** (230 ms / 0.3× before) |

### 1.3 Diffusion

| Model | File | Status | Notes |
|-------|------|--------|-------|
| **Stable Diffusion 1.5** | `models/sd*.go` | Done — pretrained | CLIP text encoder + UNet2DCondition + VAE decoder + DDIM; BF16 on RTX 5060 Ti. `LoadSD15FromHF`; `examples/37_sd15_txt2img` |

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
- [x] **GGUF quantized inference**: `ops.MatMulQ` (fused dequant-matmul CUDA kernels for Q4_0 / Q5_0 / Q8_0 / Q4_K / Q5_K / Q6_K; dequant+matmul fallback for the rest), `nn.Linear` accepts quantized weights, `models.LoadLlamaGGUF` (Q/K un-permute for llama arch, `rope_freqs` factors). Qwen2-0.5B Q8_0 / Q4_K_M and Llama-3.2-1B Q4_K_M match transformers greedy; logits within quant noise
- [x] **Quantized CPU kernels** (2026-11-08): GGUF `matmul_q` for Q4_0 / Q5_0 / Q8_0 / Q4_K / Q6_K and GPTQ / AWQ `int4_gemv` keep weights quantized; AVX2 row decoders (`kernels/cpu/qdecode_amd64.s`, bit-identical to the Go decoders, 5–9× faster per row) feed 48-row tiles into `simd.MatTB`; GGUF tied heads stay quantized (`MatMulQFusedOn`)
- [x] Quantized embedding lookup: `IndexSelect` on every GGUF block format and the narrow fixed-width dtypes (byte-row gather, CPU + CUDA + WebGPU); `nn.Embedding` with a quantized table dequantizes only the gathered rows; `LoadLlamaGGUF` keeps `token_embd` quantized (tied head via `ops.MatMulQFused`) — 2026-10-29
- [x] Tokenizer from GGUF metadata (`tokenizer.LoadGGUF`, 2026-10-29)
- [x] **GPTQ / AWQ 4-bit** (`quant.Int4` canonical layout, `quant.FromGPTQ` incl. act-order `Perm`, `quant.FromAWQ`; `ops.DequantizeInt4`, `ops.MatMulInt4` with fused CUDA `int4_gemv` for ≤ 8 rows and dequant-into-scratch + GEMM above; CPU and WebGPU kernels; 4-bit MoE experts) — 2026-11-08
- [x] FP8 E4M3 tensor-core GEMM (cuBLASLt) + safetensors `F8_E4M3` load/save — 2026-10-28; blockwise (DeepSeek / Qwen `weight_scale_inv`) scales with fused kernels — 2026-11-07

### Backends

- [x] **CUDA**: default build; PTX modules built with NVRTC 13.0 (`scripts/nvrtc_ptx.py`, no nvcc or host compiler needed; loads on CUDA 13.0 drivers) or `scripts/build_ptx.sh` (compute_80), embedded and JIT-compiled by the driver; cuBLAS / cuBLASLt / cuDNN bindings; driver-context binding per thread (`cudart.ensureContext`) and module-load retry; caching allocator; weight-sized temporaries reuse one scratch buffer per device / dtype. `-tags nocuda` opts out; `CGO_ENABLED=0` picks the CPU-only build automatically
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

Candle `candle-transformers/src/models` (main, Oct 2026), grouped, with Light's status. **2026 Candle additions in bold.**

**LLMs (dense):** llama ✅, mistral ✅ (Mistral 7B v0.3 BF16, 2026-11-09), phi ❌ (Candle's `phi` = Phi-1/1.5/2 parallel-block architecture; not ported), phi3 ✅ (Phi-3.5-mini; Phi-4-mini with partial rotary 0.75 + longrope + tied head + GPT-4o tokenizer, which Candle's phi3 lacks), gemma3 ✅ (1B, 4B text-only; SigLIP vision input open), gemma2 ✅, **gemma4** ✅ (E2B text with PLE and KV sharing; Candle's gemma4 text port lacks PLE/KV sharing, so Light is ahead on E2B/E4B), gemma (v1) ❌, qwen2 ✅, qwen3 ✅ (dense via Qwen3-ASR/SDAR bodies; standalone checkpoint logit check open), deepseek2 ✅ (2026-11-08: V2-Lite-Chat verified; V2 group-limited and V3 sigmoid routing tiny-verified), **falcon** ✅ (all three decoder layouts incl. ALiBi and the 40B grouped-K/V architecture, which Candle's falcon.rs rejects), chatglm / glm4 / glm4_new / codegeex4: glm4 ✅ (GLM-4-9B-0414 verified 2026-11-08; `glm` tiny-verified; chatglm / codegeex4 ❌), bigcode / starcoder2 ❌, mpt ❌, persimmon ❌, stable_lm ❌, yi ❌ (= llama), olmo / olmo2 ❌, granite ❌, helium ❌, **lfm2** ❌, **smol** (SmolLM3) ❌, llama2_c ❌, mixformer (phi-1/2) ❌.
**LLMs (MoE / recurrent):** qwen3_moe ✅ (HF + GGUF + 4-bit experts), mixtral / qwen2_moe ⚠️ (same `nn.SparseMoE` block, HF loaders not wired), granitemoehybrid ⚠️ (attention-only layers, e.g. granite-4.0-micro; Mamba layers ❌), based, mamba, **mamba2**, rwkv_v5/v6/**v7**, recurrent_gemma ❌ (Light has gated-delta and lightning-attention scans, not Mamba / RWKV ones).
**Light models without a Candle module in this list:** Qwen3.5 (Gated DeltaNet hybrid, 0.8B–27B), Muse Glimmer 30B, Mellum 2.1, Nemotron, MiMo (+ MTP speculative decoding), MiniMax-Text-01 (tiny-verified), SDAR (block-diffusion LM), Qwen3-ASR, EmbeddingGemma 2.
**Quantized GGUF variants:** 19 modules in Candle; Light: `LoadLlamaGGUF` covering the llama / qwen2 / qwen3 / mistral / qwen3moe GGUF architectures with quantized embedding, tied head and MoE experts, plus `LoadQwen35GGUF`, `LoadMuseGlimmerGGUF` and `LoadMoshiGGUF`; verified against llama.cpp tokenizer fixtures and transformers logits (Qwen2 0.5B, Llama-3.2, Qwen3-MoE, Qwen3.5-27B, Muse Glimmer 30B).
**Encoders / embeddings:** bert ✅, **embedding_gemma2** ✅ (Candle has no port, Light ahead), distilbert ❌, jina_bert ❌, **nomic_bert** ❌, modernbert ❌, debertav2 ❌, xlm_roberta ❌, stella_en_v5 ❌, nvembed_v2 ❌, colpali ❌, t5 ❌, marian ❌.
**Vision:** resnet ✅, vgg ✅ (random init), vit ✅, beit / dinov2 / dinov2reg4 / eva2 ❌, convnext ❌, convmixer ❌, efficientnet / efficientvit ❌, mobileone / mobilenetv4 / fastvit ❌ (we have mobilenetv2 instead), hiera ❌, repvgg ❌, segformer ❌, segment_anything ❌, depth_anything_v2 ❌, trocr ❌. Light-only: yolov5 (random init), resnet9, lenet, alexnet.
**Multimodal:** clip ⚠️ (SD text tower only), openclip / chinese_clip / siglip / mobileclip ❌, blip ❌, llava ❌, moondream ❌, paligemma ❌, pixtral ❌, qwen3_vl ❌, **paddleocr_vl** ❌, voxtral ❌.
**Diffusion:** stable_diffusion ⚠️ (1.5 only; Candle: 1.5/2.1/XL/Turbo), mmdit (SD3) ❌, flux ❌, wuerstchen ❌, **z_image** ❌.
**Audio:** whisper ✅, Light-only qwen3_asr ✅, **mimi** ✅ (+ streaming), **moshi** ✅ (Q8 GGUF, verified vs Rust moshi-core; full-duplex protocol, real time on an L4); encodec, dac, snac, metavoice, parler_tts, csm ❌.

Validated with real weights (2026-11-08): Candle ~110 modules (its examples run them). Light: the 24 of 2026-11-06 (Moshi LM, Mimi, Phi-4-mini, Falcon, GPT-2, Qwen2, Llama-3.2, Phi-3.5, BERT/bge-small, SD 1.5, ResNet-50, MobileNetV2, ViT-B/16, MercyLM, Whisper-tiny, Qwen3-ASR-1.7B, Qwen3-MoE (tiny-random), SDAR-1.7B, Gemma 3 1B, Gemma 3 4B, Gemma 2 2B, Gemma 4 E2B, EmbeddingGemma 2, plus GPT-2 on WebGPU) and since then Mellum 2.1 12B, Qwen3.5 0.8B / 9B / 27B (BF16, FP8, GGUF), Muse Glimmer 30B (GGUF), DeepSeek-V2-Lite-Chat, GLM-4-9B-0414, Nemotron-4-Mini 4B, MiMo-7B-RL, and the GPTQ / AWQ checkpoints of Qwen2.5-0.5B-Instruct and TinyLlama-1.1B; MiniMax-Text-01 on a tiny-random checkpoint.

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

### Colab L4 / A100

| Workload | Result |
|----------|--------|
| Moshi speech-to-speech (Moshiko Q8 + Mimi, L4) | 58 ms per 80 ms frame through the engine (1.4× real time); model loop 61.9 ms / frame (LM 34.2, depformer 10.7, Mimi 9.8 + 7.2) |
| Qwen3.5-9B BF16 heavy test (A100) | 32 s for load + prefill + 16 greedy steps |
| Qwen3.5-27B-FP8, DeepSeek-V2-Lite, GLM-4-9B (A100) | 47–48 s each for load + prefill + 16 decode steps |
| Qwen3.5-27B / Muse Glimmer 30B GGUF Q4 (A100) | 35–42 s each for load + prefill + 16 decode steps |
