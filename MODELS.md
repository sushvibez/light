# Light — Model Implementations

Tracking progress toward implementing all torchvision.models architectures **plus modern detection models (YOLO family) and language models (GPT-2 family)**.

Each model lives in `models/` as a `nn.Module`-compatible struct with a one-call constructor and a `Forward(x)` method. Inference paths are validated first; training is layered on top.

**Legend:** `Done — inference + GPU` ⟶ random-init forward verified on CUDA, param count matches reference. `Done — pretrained` ⟶ HF/torchvision weight loading shipped. `Training` ⟶ end-to-end training example green.

---

## Classification Models

### CNN Architectures

| Model | File | Status | Notes |
|-------|------|--------|-------|
| LeNet-5 | `models/lenet.go` | Training | MNIST 98.20% test acc in `examples/22_train_mnist` |
| AlexNet | `models/alexnet.go` | Training | 5.4M params; 100% eval on `examples/21_train_alexnet` |
| VGG (11/13/16/19) | `models/vgg.go` | Done — inference + GPU | 138.36M params (VGG-16) on RTX 5060 Ti, 166ms forward; `examples/28_vgg16` |
| SqueezeNet (1.0, 1.1) | `models/squeezenet.go` | Not started | Lightweight, GPU feasible |
| DenseNet (121/161/169/201) | `models/densenet.go` | Not started | Dense skip connections |
| GoogLeNet / Inception v1 | `models/googlenet.go` | Not started | Inception modules |
| Inception v3 | `models/inception.go` | Not started | Auxiliary classifiers |
| **ResNet (CIFAR variants 20/32/44/56)** | `models/resnet.go` | Training | `examples/23_train_resnet_cifar10` — ResNet-20 → 82.84% on CIFAR-10 (no augmentation). |
| **ResNet9 (DAWNBench / fastai)** | `models/resnet9.go` | Training | 6.57M params; `examples/34_resnet9_cifar10` — ResNet9 + RandomCrop(pad=4) + RandomHorizontalFlip + per-channel normalize + AdamW + CosineLR. Targets 90+% test accuracy in 8 epochs. |
| **ResNet (18/34/50/101/152)** | `models/resnet_imagenet.go` | Done — pretrained + train/infer split | torchvision-compatible (BasicBlock + Bottleneck); ResNet-50 = 25.56M params, 150ms; `examples/26_resnet50/{train,infer}` for from-scratch CIFAR-10, `examples/32_resnet_pretrained` for HF `microsoft/resnet-50` weights. |
| ResNeXt (50, 101) | `models/resnext.go` | Not started | Grouped convolutions (depthwise unblocked) |
| Wide ResNet (50, 101) | `models/wide_resnet.go` | Not started | Wider channels |
| MnasNet (0.5, 1.0, 1.3) | `models/mnasnet.go` | Not started | NAS-designed mobile net |
| **MobileNetV2** | `models/mobilenetv2.go` | Done — pretrained + train/infer split | 3.50M params (×1.0); inverted residuals + ReLU6; `examples/27_mobilenetv2/{train,infer}` for from-scratch CIFAR-10, `examples/33_mobilenet_pretrained` for HF `google/mobilenet_v2_1.0_224` weights. |
| MobileNet v3 (Small / Large) | `models/mobilenetv3.go` | Not started | Needs SE block + h-swish |
| ShuffleNet v2 (0.5/1.0/1.5/2.0) | `models/shufflenetv2.go` | Not started | Channel shuffle |
| EfficientNet (B0–B7) | `models/efficientnet.go` | Not started | Needs SE block |
| EfficientNet v2 (S/M/L) | `models/efficientnet.go` | Not started | Fused-MBConv |
| RegNet (X/Y) | `models/regnet.go` | Not started | Regular network design |
| ConvNeXt (Tiny/Small/Base/Large) | `models/convnext.go` | Not started | Modern CNN |

### Transformer / Attention Architectures

| Model | File | Status | Notes |
|-------|------|--------|-------|
| **ViT-B/16, ViT-L/16, ViT-H/14** | `models/vit.go` | Done — inference + GPU | ViT-B/16 = 86.57M params, 152ms; patch embed via Conv2d; CLS token + learned position; `examples/29_vit_b16` |
| Swin Transformer (T/S/B/L) | `models/swin_transformer.go` | Not started | Needs window attention |
| MaxViT | `models/maxvit.go` | Not started | Multi-axis attention |

---

## Language Models

| Model | File | Status | Notes |
|-------|------|--------|-------|
| **GPT-2 (small / medium)** | `models/gpt2.go` | Done — pretrained | `LoadGPT2FromHF("gpt2", dev)` — 124.4M tied params; greedy matches PyTorch; `examples/24_gpt2_forward`, `examples/25_gpt2_pretrained` |
| GPT-2 large / xl | `models/gpt2.go` | Architecture-ready | Need sharded safetensors loader for the multi-file checkpoint |
| **Qwen2 / Qwen2.5** | `models/llama.go` | Done — pretrained | Generic LLaMA-arch decoder (RoPE + RMSNorm + SwiGLU + GQA + KV cache + sharded safetensors + BF16 / F32); `LoadQwen2FromHF`, `LoadQwen2FromHFMixed`; `examples/35_qwen_inference`, `examples/36_qwen_chat` (multi-turn with KV reuse) |
| **Llama-3 / 3.1 / 3.2** | `models/llama3_load.go` | Done — pretrained | `LoadLlama3FromHF` / `LoadCausalLMFromHF` (dispatch on `model_type`); llama3 RoPE scaling (`models/rope_scaling.go`); BOS from `tokenizer.json`, EOS list from `generation_config.json`. Llama-3.2-1B-Instruct: " Paris", "The answer is 4."; F32 29 tok/s, BF16 34 tok/s. `examples/35_qwen_inference -repo meta-llama/Llama-3.2-1B-Instruct` |
| Mistral / Mixtral | `models/llama3_load.go` | Architecture-ready (SWA done) | `LoadCausalLMFromHF` accepts `model_type: mistral`; sliding-window attention + rotating KV cache verified against transformers on Qwen2 (window 8); a Mistral 7B checkpoint (GGUF Q4) has not been run on this box yet; MoE for Mixtral is open |
| **Whisper (tiny…large-v3)** | `models/whisper.go`, `whisper_load.go`, `audio/` | ✅ Verified 2026-10-28 | `LoadWhisperFromHF`; host log-mel identical to `WhisperFeatureExtractor` (filters from preprocessor_config.json), encoder (Conv1d×2 + sinusoid positions), decoder with self-KV + cross-KV cache, greedy with `suppress_tokens`, language auto-detect. whisper-tiny: mel/encoder values and step-0 top-5 logits match transformers F32; "Front Center." greedy identical; 104 ms encode + 78 ms decode on the GPU. Larger sizes untested but same layout. |
| **Qwen3-ASR (1.7B / 0.6B)** | `models/qwen3_asr.go`, `qwen3_asr_load.go` | ✅ Verified 2026-10-28 | `LoadQwen3ASRFromHF(repo, dev, dt)`; 128-bin slaney log-mel built in Go, Conv2d×3 chunk stem (100 mel frames → 13 tokens), 24-layer encoder with 104-token attention windows, GELU projector into the Qwen3-1.7B decoder (BF16) via `ForwardEmbeds`. On `Qwen/Qwen3-ASR-1.7B-hf`: filterbank/mel/encoder/projector values and step-0 top-5 match transformers 5.21 F32; greedy "language English<asr_text>Front, center." identical, forced-language prefill identical. 29 ms encode + 194 ms decode (7 tokens) on the GPU after warm-up. |
| **Gemma 3 (text)** | `models/gemma.go`, `gemma_load.go` | ✅ Verified 2026-10-28 | `LoadGemma3FromHF` (`gemma3_text`; `gemma2` config parsed incl. softcaps, untested). √D embed scale, (1+w) RMSNorm, 4 norms/block, QK-norm, GELU-tanh gated MLP, 5:1 local(512, θ=1e4)/global(θ=1e6) layers, rotating cache. gemma-3-1b-it F32 top-5 logits match transformers to 4 dp (54.9099 vs 54.9100); BF16 greedy "2 + 2 = 4" identical; 33 tok/s. Weights from the ungated `unsloth/gemma-3-1b-it` mirror (google/ repo is gated). **4B** (2026-10-29): the multimodal `gemma3` checkpoint loads text-only (nested `text_config`, `language_model.model.*` keys, linear `rope_scaling` ×8 on global layers, SigLIP tower skipped); `unsloth/gemma-3-4b-it` BF16 on CUDA matches transformers: identical top-5 ids (56.0/34.0/33.0/32.5/32.0 vs 55.25/33.25/32.5/32.25/31.5), greedy `2 + 2 = 4\n<end_of_turn>` identical. 12B/27B use the same path but exceed 16 GB at BF16. Vision input still open. |
| **Gemma 2** | `models/gemma.go`, `gemma_load.go` | ✅ Verified 2026-10-29 | Same stack as Gemma 3 with `NoQKNorm`, attention softcap 50, final logit softcap 30, alternating local(4096)/global layers, RoPE base 1e4 everywhere. `unsloth/gemma-2-2b-it` F32 on CUDA matches transformers to 4 dp (21.2942 vs 21.2943; all top-5 ids identical), greedy `2 + 2 = 4 \n<end_of_turn>` identical; BF16 through the engine gives the same reply. 9B/27B load through the same path (27B exceeds 16 GB). |
| **Gemma 4 (text)** | `models/gemma4.go`, `gemma4_load.go` | ✅ Verified 2026-10-29 | `LoadGemma4FromHF` (`gemma4` multimodal checkpoints text-only, `gemma4_text`). Per-layer embeddings (262144×L·256 table + projected context, RMS-normed, mixed /√2, gated into each block), KV sharing across the last `num_kv_shared_layers` layers (reuse the last non-shared layer of the same type, no k/v weights), double-wide MLP on shared layers, global layers with head dim 512 and proportional RoPE (first 25 % of the pairs rotate, θ=1e6; local θ=1e4 full), attention scaling 1 (q_norm carries it), unweighted RMS norm on V, per-layer `layer_scalar`, direct RMSNorm weights, final softcap 30. Tensors are loaded one at a time straight to the device (no 10 GB host staging). `unsloth/gemma-4-E2B-it` BF16 on CUDA vs transformers: top-5 ids identical [236778 236795 818 7925 100], logits 21.85/18.29/16.98/12.19/9.93 vs 22.13/18.0/17.0/12.75/9.88, greedy `2 + 2 = **4**<turn|>` identical; 4.63 B text params. E4B loads the same way (≈16 GB BF16, over the card); 12B/31B dense would need a quantized path; 26B-A4B MoE blocks not implemented. Vision/audio towers open. |
| **EmbeddingGemma 2** | `models/embeddinggemma2.go`, `embeddinggemma2_load.go` | ✅ Verified 2026-10-29 | `LoadEmbeddingGemma2FromHF` (`embedding_gemma2`, text encoder only; vision/audio towers skipped). Gemma 4 style blocks (shared `gemma4Block`) run bidirectionally: projection-only per-layer embeddings (norm(proj(h)·D^-½), no PLE token table, gated into every block), 4 query heads with 256-dim 2-KV-head sliding layers and 512-dim single-KV-head full-attention layers (every 6th; `per_layer_config`), full-width RoPE per type (θ 1e4 local / 1e6 global), attention scaling 1 with q/k norm and unweighted V norm, direct RMSNorm weights, `layer_scalar`, sliding layers attend within |i−j| ≤ 512 in both directions, final norm → 512→768 `embedding_projection`; `Embed` mean-pools (padding-aware) and L2-normalises (sentence-transformers 1_Pooling + 2_Normalize). `unsloth/embeddinggemma-2` vs transformers 5.21 BF16 (`ref_embeddinggemma2.py`): first 8 dims and the 3×3 cosine matrix within 1.5e-2 on CUDA BF16 and CPU F32 (cos(query, doc) 0.853 vs 0.854; off-topic 0.593), padded batch = unpadded, CPU↔CUDA F32 within 2e-3; 271 M text params. Task prefixes (`task: search result | query: `, `title: none | text: `) are the caller's. Engine: `/v1/embeddings` (`engine.Embedder`), catalog kind `embedding`, Studio load/unload. Candle has no EmbeddingGemma port. |
| **Falcon** | `models/falcon.go`, `falcon_load.go` | ✅ Verified 2026-10-29 | `LoadFalconFromHF` (`model_type: falcon`, also the legacy `RefinedWebModel`/`RefinedWeb`): every transformers decoder variant — falcon-rw-1b (ALiBi, multi-head, sequential attention→MLP, biases), falcon-7b (rotary, multi-query, parallel attention+MLP on one LayerNorm, no bias) and the 40B/180B "new decoder architecture" (grouped K/V in `(kv, g+2, hd)` fused layout, `ln_attn`+`ln_mlp`). Fused `query_key_value` split per layout, exact-erf GELU, tied or separate `lm_head`, safetensors or `pytorch_model.bin` (via `light/pth`), local checkpoint directories. ALiBi slopes follow `build_alibi_tensor` (slope·position rounded in BF16 as transformers does), folded into the causal mask with the 1/√d scale. `tiiuae/falcon-rw-1b` F32 on CUDA vs transformers: top-5 ids and logits identical to 4 dp (35.2608 / 31.7368 / 30.4928 / 30.3152 / 29.2224), greedy identical (` Rome. The capital of the United Kingdom`); tiny-random checkpoints for the 7B and 40B layouts (`models/testdata/falcon`, `hf-internal-testing/tiny-random-FalconForCausalLM`) match to 2e-3 on CPU/CUDA F32 and BF16, chunked decode = single shot. Note: transformers' *eager* Falcon adds ALiBi twice (inside attention and again folded into the mask); Light matches its SDPA path and the original Falcon code. Candle's falcon.rs only handles the 7B layout. Engine: catalog lists torch-zip checkpoints (`kind: pytorch`), 29 tok/s BF16 on the 5060. |
| **Phi-4-mini** | `models/llama.go` (`PartialRotary`), `llama3_load.go` | ✅ Verified 2026-10-29 | `LoadCausalLMFromHF` (`model_type: phi3`): Phi-3 stack plus `partial_rotary_factor` 0.75 (RoPE rotates the first 96 of 128 head dims with 48 longrope factors, the rest pass through), tied LM head, GPT-4o 200k vocabulary. `microsoft/Phi-4-mini-instruct` BF16 on CUDA vs transformers 5.21 (`ref_phi4.py`, 21-token system+user chat prompt): top-5 ids {976 37513 15952/15957 30112} with the reference's 'Blue'/'Light' tie at 40.5, logits 47.50/42.25/40.50/40.25/40.25 vs 47.75/42.5/40.5/40.5/40.25, 16-token greedy identical (`The sky is blue due to Rayleigh scattering, where shorter blue wavelengths of sunlight`); 3.84 B params, 7.7 GB. Tokenizer: tokenizer.json `Split` with behavior `Removed`+`invert` (GPT-4o form) now honoured; ids identical to HF on chat markers, mixed Unicode and code. Engine: `phi4` chat template (`<\|role\|>…<\|end\|>`, no newlines; detected by the `<\|tool\|>` marker Phi-3.5 lacks), EOS 200020/199999. Candle's `phi3` has no partial rotary. |
| **Mimi (codec)** | `models/mimi.go`, `mimi_load.go` | ✅ Verified 2026-10-29 | `LoadMimiFromHF` (`kyutai/mimi`, 96 M params, F32). Causal SEANet encoder (conv 7 → 4× [resnet, ELU, strided conv 2r/r] → ELU → conv 3), 8-layer causal transformer (RoPE, sliding window 250, pre-LayerNorm with bias, layer scales, exact-erf GELU MLP), stride-2 replicate-padded downsample to 12.5 Hz, split RVQ (1 semantic + 31 acoustic Euclidean codebooks, 2048×256, `embed_sum / cluster_usage`), grouped transposed-conv upsample, decoder transformer, SEANet decoder with trimmed causal transposed convs. `Encode(audio, n)` → codes (B, n, T) I64, `Decode(codes)` → audio. On `testdata/mimi/front_center_24k.wav` (1.43 s) vs transformers MimiModel F32: 141/144 codes identical (the 3 differ on a 5e-4 near-tie where torch's matmul-form cdist rounds the other way), decoded audio first 8 samples within 1e-3, RMS 0.06754 vs 0.06738, peak 0.44606 vs 0.44597; per-stage activations match to ≤3e-5 (`TestMimiStagesAgainstDump`); CPU = CUDA. Streaming (2026-11-06): `Mimi.NewStream()` → `EncodeStep` (k·1920 samples → k frames) / `DecodeStep` (k frames → k·1920 samples) with kyutai's streaming states (conv memories of kernel−stride inputs, zero or first-sample for replicate pads; transposed-conv overlap carry minus bias; sliding-window KV caches); `TestMimiStreamingMatchesFullSequence`: 136/136 codes identical and audio within 1e-6 of the full-sequence path for 1- and 3-frame chunks, CPU and CUDA. Candle has a Mimi port too (parity). |
| **Moshi (speech-to-speech LM)** | `models/moshi.go`, `moshi_load.go`, `moshi_converse.go`, `tokenizer/spm_model.go` | ✅ Verified 2026-11-06 | `LoadMoshiFromHF` / `LoadMoshiGGUF` (`kyutai/moshiko-candle-q8` model.q8.gguf, 7.69 B params Q8_0 + F32 norms, CUDA fused MatMulQ; 8.2 GB VRAM). Temporal transformer (32 layers, d 4096, 32 heads, fused QKV, interleaved RoPE θ 1e4 in F32, RMSNorm eps 1e-8 with `alpha` weights, SiLU-gated MLP split in halves, sliding context 3000 keys) over text + 16 Mimi codebooks (8 own + 8 user), `text_linear` head, 8-slice depformer (6 layers, d 1024, no positional embedding, per-slice weights, one shared KV cache per frame; slice 0 embeds the text token, slice s the code of slice s-1; heads of width 2048 so pad is never emitted). `MoshiGen` runs kyutai's multistream protocol (acoustic delay 2 with forced pads, user codes lagged one frame, text fed back); `Converse(mimi, audio, extra)` runs a turn frame by frame through the streaming Mimi (80 ms in → step → 80 ms out), as the live system does. Raw SentencePiece unigram text tokenizer (`tokenizer.LoadSentencePieceModel`, byte fallback) verified vs the sentencepiece package. Tiny-random: full-sequence = stepwise through the cache (window via mask and trimming), CPU = CUDA. Real checkpoint: `TestMoshikoConverse` on the Front Center clip answers "Hello, what's up?" with 3.3 s of speech (greedy). Vs Rust moshi-core on the same GGUF (`~/workspace/experimental/rust/moshi-ref`, CPU, argmax, 6 frames of LCG user codes): text top-5 identical every frame with logits within 0.15, 47/48 depformer codes identical (one 0.16-logit near-tie in slice 7 of frame 1; Candle's CPU path quantizes activations to Q8). Engine: `engine.Speech` (`LoadSpeech`, `Converse` with kyutai's default top-k sampling or greedy), runtime speech slot that releases the chat model, catalog kind `speech` (GGUF with depformer tensors, no config.json), `POST /v1/audio/conversations`; Studio: load/unload + 🗣 Talk (reply saved next to the input WAV). Speech-to-speech through the engine runs at 58 ms per 80 ms frame on an L4 (1.4× real time; 230 ms / 0.3× before the 2026-11-08 perf pass: ring KV cache + split-K decode attention). |
| **DeepSeek (V2 / V3)** | `models/deepseek.go` (planned) | Not started | MLA (Multi-head Latent Attention), DeepSeekMoE; significant deviation from LLaMA — own model file |
| **Phi-3 / Phi-3.5** | `models/llama3_load.go` (`LoadPhi3FromHF`) | Done — pretrained | Llama template + fused `qkv_proj` / `gate_up_proj` split + LongRoPE (`models.RopeScaling{Type: "longrope"}`) + SentencePiece BPE. `microsoft/Phi-3.5-mini-instruct` BF16: top-5 logits match transformers BF16; "What is 2+2?" → "The sum of 2 and 2". 7.6 GB, 15 tok/s. `examples/35_qwen_inference -repo microsoft/Phi-3.5-mini-instruct -bf16` |
| Phi-4 | — | Architecture-ready | Same template (no LongRoPE); 14B needs 4-bit to fit 16 GB |
| **MiniMax (Text-01)** | `models/minimax.go` (planned) | Not started | Lightning attention + softmax-attn hybrid; MoE gate; new attention kernel needed |
| **GLM (GLM-4 / ChatGLM)** | `models/glm.go` (planned) | Not started | Pre-LN + post-LN, RoPE half (rotate half the head dim), MQA; mostly an arch variant |
| **Nemotron (Nano / Mini / Super)** | `models/nemotron.go` (planned) | Not started | LLaMA-style with squared-ReLU MLP and selective layers; mostly a key-mapping job |
| MiMo (Xiaomi MiMo) | `models/mimo.go` (planned) | Not started | LLaMA-style; speculative decoding hooks deferred to Phase 10 |
| **BERT** | `models/bert.go`, `models/bert_load.go` | Done — pretrained | `LoadBertFromHF` (encoder, CLS/mean pooling, L2 normalize) + `BertForMaskedLM` (shape-tested only). `unsloth/bge-small-en-v1.5` matches PyTorch to 7.8e-7; WordPiece ids identical. `examples/40_bert_embeddings` |
| RoBERTa | `models/bert.go` | Not started | BERT with byte-level BPE + position offset |
| **MercyLM** | `models/mercy.go` | Done — pretrained | 15.4 M GPT-style LM from safetensors; `examples/38_mercy_chat` |
| T5 / Flan-T5 | `models/t5.go` | Not started | Encoder-decoder; relative position bias |
| Qwen3 (dense) | `models/llama.go` (`QKNorm`) | Done — used as the Qwen3-ASR decoder; standalone Qwen3-0.6B/1.7B/4B load via `LoadCausalLMFromHF` (`qwen3`), standalone logits not yet checked | QK-norm, no QKV bias, rope_theta from `rope_parameters` |
| **Qwen3-MoE** | `models/llama.go` (`LlamaConfig.NumExperts…`, `llamaBlock.MoE`), `qwen_load.go` (`mapHFMoE`), `nn/moe.go` | ✅ Verified 2026-10-28 (architecture) | `LoadCausalLMFromHF` (`qwen3_moe`): QK-norm attention + routed FFN on every `decoder_sparse_step` layer (`mlp_only_layers` stay dense), `norm_topk_prob`, both HF expert layouts (transformers-5 fused `experts.gate_up_proj`/`down_proj`, hub per-expert `experts.{e}.*`), router `gate.weight`/`router.weight`. On `hf-internal-testing/tiny-random-Qwen3MoeForCausalLM` (2 layers, 4 experts, top-2, 20-token prompt) CPU F32 and CUDA F32 reproduce transformers 5.21's top-5 ids and logits to 5 decimals; CUDA BF16 matches the BF16 reference. GGUF `qwen3moe` files load via `LoadLlamaGGUF` with quantized experts kept quantized (`ops.MoEGemmQ`; `attn_q_norm`/`attn_k_norm` and `ffn_gate_inp` handled): a Q8_0 / Q4_0 round trip of the tiny checkpoint reproduces the reference logits within 1e-3 / 1e-2 on CPU and CUDA. Qwen3-30B-A3B Q4_K_M is ≈18 GB, above the 16 GB card. |
| **SDAR-1.7B-Chat (block-diffusion LM)** | `models/sdar.go` (`Llama.ForwardDiffusion`, `GenerateDiffusion`, `SDAROpts`), `models/llama.go` (block-bidirectional mask, no-store KV pass), `engine/engine.go` (diffusion branch) | ✅ Verified 2026-10-28 | First diffusion LLM: `JetLM/SDAR-1.7B-Chat` is a Qwen3-1.7B body (`model_type` `sdar`, QK-norm) decoded by blocks of 4 `<|MASK|>` tokens — bidirectional attention inside a block, causal across blocks, up to `DenoisingSteps`+1 passes per block committing the most confident predictions (static schedule or `low_confidence_dynamic` threshold), KV cache stored once per finished block. On CUDA BF16 the first denoising pass reproduces transformers' top-5 ids with logits within BF16 noise (53.0 vs 54.0) and greedy block decoding of "What is 2+2?" gives the identical `2 + 2 = 4<|im_end|>`. Engine `Generate` streams per finished block. Larger SDAR variants (4B/8B, b8–b64 block sizes, 30B-A3B MoE) use the same loader with `BlockLength` set accordingly. |

## Diffusion Models

| Model | File | Status | Notes |
|-------|------|--------|-------|
| **Stable Diffusion 1.5** | `models/sd*.go` | Done — pretrained | CLIP text encoder + UNet2DCondition + VAE decoder + DDIM; BF16 on RTX 5060 Ti. `LoadSD15FromHF`; `examples/37_sd15_txt2img` |
| SDXL | — | Not started | Dual text encoders, bigger UNet |

---

## Object Detection Models

### YOLO Family
| Model | File | Status | Notes |
|-------|------|--------|-------|
| **YOLOv5 (n/s/m/l/x)** | `models/yolov5.go` | Done — inference + GPU | YOLOv5n = 1.87M params, 141ms forward at 640×640; CSPDarknet + PANet + Detect; `examples/30_yolov5n` |
| YOLOv8 (n/s/m/l/x) | `models/yolov8.go` | Not started | Anchor-free, decoupled head |
| YOLOv11 / YOLOv12 | `models/yolov11.go` | Not started | Latest Ultralytics releases |
| YOLOX | `models/yolox.go` | Not started | Decoupled head, SimOTA |

### Torchvision Detectors (`torchvision.models.detection`)
| Model | File | Status | Notes |
|-------|------|--------|-------|
| Faster R-CNN | `models/detection/faster_rcnn.go` | Not started | Needs RoI Align |
| Mask R-CNN | `models/detection/mask_rcnn.go` | Not started | Adds mask head |
| RetinaNet | `models/detection/retinanet.go` | Not started | Focal loss + FPN |
| SSD (SSD300, SSDLite320) | `models/detection/ssd.go` | Not started | Single-shot detector |
| FCOS | `models/detection/fcos.go` | Not started | Anchor-free |
| DETR | `models/detection/detr.go` | Not started | Transformer-based; ViT building blocks ready |

---

## Segmentation Models (`torchvision.models.segmentation`)

| Model | File | Status | Notes |
|-------|------|--------|-------|
| FCN (ResNet-50/101) | `models/segmentation/fcn.go` | Not started | Backbone ready |
| DeepLab v3 (ResNet-50/101, MobileNet v3) | `models/segmentation/deeplabv3.go` | Not started | Needs ASPP |
| LRASPP | `models/segmentation/lraspp.go` | Not started | Lite for MobileNet v3 |
| SAM / SAM 2 | `models/sam.go` | Not started | Promptable segmentation |

---

## Video Models (`torchvision.models.video`)

| Model | File | Status | Notes |
|-------|------|--------|-------|
| R3D-18 / MC3-18 / R(2+1)D-18 | `models/video/r3d.go` | Not started | Needs Conv3d |
| MViT v1 / v2 | `models/video/mvit.go` | Not started | 3D transformer |
| S3D | `models/video/s3d.go` | Not started | Separable 3D |
| Swin3D | `models/video/swin3d.go` | Not started | 3D Swin |

---

## Optical Flow Models (`torchvision.models.optical_flow`)

| Model | File | Status | Notes |
|-------|------|--------|-------|
| RAFT | `models/optical_flow/raft.go` | Not started | Recurrent all-pairs field transforms |

---

## Quantization Models

| Model | File | Status | Notes |
|-------|------|--------|-------|
| Quantized GoogLeNet, Inception v3, MobileNet v2/v3, ResNet (18/50), ShuffleNet v2 | — | Not started | Need INT8 tensor + ops |

---

## Landscape: open-weight releases Jun–Oct 2026 (surveyed 2026-10-27)

Web survey of what shipped after this doc's last roadmap pass; only models that fit a 16 GB GPU (BF16 or Q4) are listed. Items marked *unverified* rest on secondary sources; check the model card before implementing.

| Model | Released | Size | Repo | What it needs from us | License |
|-------|----------|------|------|-----------------------|---------|
| **Qwen3.8-27B** | Aug 2026, Alibaba | 27B dense (28B w/ vision) | `Qwen/Qwen3.8-27B` (+FP8) | Hybrid stack: 3× Gated DeltaNet + 1× gated attention per block, partial RoPE (rotary dim 64), MRoPE sections [11,11,10], MTP head. Q4 only on 16 GB. Same arch as Qwen3.5/3.6/Next, so one implementation covers the 2026 Qwen line. | Apache-2.0 |
| **Muse Glimmer 30B** | Aug 2026, Meta | 29.6B + 1.8B ViT | `meta-models/Muse-Glimmer-30B` | Dense GQA (32Q/2KV, d=128), local/local/local/global with 2048 sliding window, RoPE only on local layers (global = NoPE), vocab 202K. Mostly Llama-3 ops + SWA. Q3–Q4 on 16 GB. Meta's Llama successor, first Apache release. | Apache-2.0 |
| **Mellum 2.1 12B-A2.5B** | Oct 2026, JetBrains | 12B MoE, 2.5B active | `JetBrains/Mellum2.1-12B-A2.5B-Thinking` | First MoE: 64 experts top-8, SWA 1024 in 3 of 4 layers, GQA 32/4. ~7 GB Q4. Good MoE test bed. | Apache-2.0 |
| **Jais 3** (0.9B–32B) | Oct 2026, MBZUAI/Inception42 | 0.9B / 3.7B / 7B / 32B | `inception42/Jais-3-7B` | Custom `k2_horizon` arch, remote code; *arch unverified*. | Apache-2.0 |
| **Apertus 1.5 8B** | Jul 2026, Swiss AI | 8B / 70B | `swiss-ai/Apertus-v1.5-8B` | xIELU activation; *1.5 details unverified*. 8B BF16 fits. | Apache-2.0 (gated) |
| **EmbeddingGemma 2** | Oct 2026, Google | 270M text (+vision/audio towers) | `google/embeddinggemma-2` | Bidirectional Gemma-4 encoder, 24L d=512, 5:1 local(1024)/global, per-layer embeddings, mean pool, Matryoshka 768→128. Closest to our BERT path. | Apache-2.0 |
| **Nemotron 3.5 Embed-1B / Rerank-8B** | Oct 2026, NVIDIA | 1B / 8B | `nvidia/Nemotron-3.5-Embed-1B-BF16` | Bidirectional Ministral-3 tower, mean pool, MLP head. | OpenMDW-1.1 |
| **WeVisOne-VL 2B/4B** | Oct 2026, Tencent | 2.4B / 4.8B | `tencent/WeVisOne-VL-2B` | Qwen3-VL base + mask head: detection, grounding, OCR, segmentation in one VLM. Needs the Qwen3-VL vision tower first. | Apache-2.0 |
| **Qwen3-ASR-1.7B** | Jun 2026 (HF-native) | 0.3B encoder + 1.7B decoder | `Qwen/Qwen3-ASR-1.7B-hf` | Whisper-style conv + attention encoder, Qwen3 decoder. Our Conv1d + attention cover it. | Apache-2.0 |
| **Qwen-Image-2.1** | Sep 2026, Alibaba | 7B DiT | `Qwen/Qwen-Image-2.1` | 32 single-stream DiT layers, Qwen3-VL-8B text encoder, 64-ch VAE; *encoder/VAE unverified*. FP8 on 16 GB. | non-commercial |
| Krea 2 / Ideogram 4.0 | Jun 2026 | 12B / 9.3B DiT | — | Same DiT shape as above. | restrictive |
| Breeze TTS 2 | Aug 2026 | 3B | `BreezeBlue/Breeze-TTS-2` | Arch undisclosed. | non-commercial |

Too large for 16 GB (awareness only): Kolibri-1 78B-A3.5B (Aleph Alpha), GLM-5.3-Flash 320B-A18B, DeepSeek V4.1-Flash, Kimi K3, Mistral Large 4, Muse Spark 816B. No Phi-5, SmolLM4, OLMo 4, Gemma 5, new YOLO weights or SAM 4 found in the window (Gemma 4 and Granite 4.1 were April).

**Takeaways for the roadmap:** (1) the 2026 Qwen line is hybrid linear attention (Gated DeltaNet), a new op family, not a loader job; (2) sliding-window attention is now on the critical path for Meta, JetBrains, Google and Aleph Alpha models alike, so wiring `nn.SDPA`'s window mask into `models.Llama` unblocks the most; (3) a MoE router + expert dispatch is the other shared prerequisite; (4) EmbeddingGemma 2 is the cheapest new validated model (BERT path + window mask + per-layer embeddings).

## Implementation Order (active — 2026-10-27)

### Inference-first roadmap (ship in this order)

**A. Text-to-image:**
1. ~~**Stable Diffusion 1.5**~~ — ✅ done (2026-05-02).
2. SDXL base (same primitives, bigger UNet, dual text encoders).
3. Flux.1-dev / SD3 (deferred — MMDiT architecture, structurally different).

**B. LLM family fan-out (builds on `models.Llama` and per-arch variants):**
4. ~~**Llama-3 / 3.1 / 3.2**~~ — ✅ done (2026-10-27, ported from GoLight).
5. ~~**Phi-3 / 3.5**~~ — ✅ done (2026-10-27): fused qkv/gate_up split, LongRoPE, SentencePiece BPE.
6. ~~**Sliding-window attention in `models.Llama`** (+ per-layer local/global pattern)~~ done 2026-10-28; NoPE layers (Muse Glimmer) still open.
7. ~~**EmbeddingGemma 2**~~ — ✅ 2026-10-29 (`models.EmbeddingGemma2`, verified vs transformers; `/v1/embeddings`).
8. **Mistral 7B** (loader exists, needs a checkpoint) and ~~**Gemma 3**~~ done 2026-10-28 (1B text); Gemma 2 verified 2026-10-29.
9. ~~**MoE router + experts**~~ — ✅ 2026-10-28 (`nn.SparseMoE`, Qwen3-MoE); first real-size validation: Mellum 2.1 12B-A2.5B, then Qwen3.6-35B-A3B / Mixtral.
10. **Muse Glimmer 30B** — Llama-3 ops + SWA + NoPE global layers; needs Q4 weight path (see Phase 9).
11. **Gated DeltaNet + gated attention + MRoPE** — the Qwen3.5/3.6/3.8 family; a new op family (chunked linear attention).
12. **DeepSeek-V2 / V3** — MLA + DeepSeekMoE.
13. **GLM-4 / ChatGLM** — half-RoPE, MQA.
14. **Nemotron** — squared-ReLU MLP.
15. **MiMo** (Xiaomi).
16. **MiniMax (Text-01)** — lightning attention + MoE (last because attention path is novel).

**C. Quantization (Phase 9):**
17. GPTQ / AWQ loaders (4-bit weight checkpoints) — INT4 × FP16 GEMM via cuBLAS or a hand kernel; unlocks 7B at ~4 GB and 14B at ~8 GB.
18. INT8 GEMM via `cublasGemmEx(CUBLAS_COMPUTE_32I)` for SmoothQuant-style W8A8.

**D. Polish before training (Phase 10):**
19. CUDA graphs for the decode hot path — drops launch overhead, big tok/s win on small models.
20. Benchmark harness vs Candle / vLLM / llama.cpp.

**E. Training (Phase 8) — last in this rollout:**
21. AdamW grad-clipping + cosine + checkpoint save/load polish.
22. Mixed-precision training (BF16 weights, F32 master, loss scaling).
23. LoRA modules + fine-tune-Qwen example.

---

### Vision baseline (already shipped)
1. **ResNet-{18,34,50,101,152}** (torchvision style) — ✅ done
2. **MobileNet v2** — ✅ done
3. **VGG-{11,13,16,19}** — ✅ done
4. **ViT-B/16, L/16, H/14** — ✅ done
5. **YOLOv5 (n/s/m/l/x)** — ✅ done
6. **GPT-2 (small + pretrained)** — ✅ done

### Wave B — Pretrained weight loading for vision models
- torchvision ResNet-50 from HF Hub (`microsoft/resnet-50`)
- MobileNet v2 from `google/mobilenet_v2_1.0_224`
- ViT-B/16 from `google/vit-base-patch16-224`
- YOLOv5n from `keremberke/yolov5n-...` or Ultralytics PyTorch ckpt

### Wave C — Training recipes for new models
- ResNet-50 ImageNet-style training (synthetic 1k-class smoke test first)
- MobileNet v2 + grouped conv backward gradcheck
- ViT-B/16 fine-tune on a small classification set
- YOLOv5n train on a small detection benchmark (needs box loss + assigners)

### Wave D — Remaining classifiers
- SqueezeNet, GoogLeNet, Inception v3, DenseNet, MobileNet v3 (needs SE), EfficientNet (needs SE)
- ShuffleNet v2, RegNet, ConvNeXt
- Swin Transformer, MaxViT

### Wave E — Detection + Segmentation
- YOLOv8 / YOLOv11 (anchor-free)
- DETR (ViT primitives ready)
- SSD, RetinaNet, FCOS
- Faster R-CNN (needs RoI Align)
- DeepLab v3, FCN, LRASPP

### Wave F — Other LMs
- Llama-2 7B (same loader as Llama-3; needs the 7B checkpoint)
- Mistral 7B
- Phi-2 / Phi-3
- ~~BERT~~ ✅ / RoBERTa

### Wave G — Video + Optical Flow
- Conv3d primitive
- R3D-18, RAFT

### Wave H — Quantization
- INT8 tensor + dequant ops (groundwork in `quant/`)
- Quantized variants of ResNet, MobileNet, etc.

---

## Open-weight landscape, Jun–Oct 2026 (surveyed 2026-10-27)

Web survey of releases after the May checkpoint, filtered for a single 16 GB GPU. Items marked *[unverified]* rest on secondary sources only; verify the model card before implementing. The pre-2026 fan-out list above (Mistral, Gemma 2, DeepSeek-V2, GLM-4, …) is now historical: those families have all moved on.

| Model | Month | Size | Arch notes for us | License | Fits 16 GB |
|-------|-------|------|-------------------|---------|------------|
| **Qwen3.8-27B** (`Qwen/Qwen3.8-27B`) | Aug | 27B dense (+1B ViT) | Hybrid: 3× Gated DeltaNet + 1× gated attention per block, partial RoPE (rotary dim 64), MRoPE, MTP head, 262K ctx. Same arch as Qwen3.5/3.6 — one implementation covers the 2026 Qwen line | Apache-2.0 | Q4 only (~16 GB, tight) |
| **Muse Glimmer 30B** (`meta-models/Muse-Glimmer-30B`) | Aug | 29.6B dense + ViT | Llama successor: 52 layers, GQA 32/2, 3 local (SWA 2048, RoPE) : 1 global (NoPE), vocab 202K, 128K ctx, image input; DFlash block-diffusion drafter | Apache-2.0 | Q3–Q4 |
| **Mellum 2.1 12B-A2.5B** (`JetBrains/Mellum2.1-12B-A2.5B-Thinking`) | Oct | 12B MoE, 2.5B active | 64 experts top-8, SWA 1024 in 3 of 4 layers, GQA 32/4 | Apache-2.0 | ~7 GB Q4, 24 GB BF16 |
| **Jais 3** (`inception42/Jais-3-*`) | Oct | 0.9B / 3.7B / 7B / 32B, 36B-A4B MoE | custom `k2_horizon` arch, `trust_remote_code` *[arch unverified]* | Apache-2.0 | 0.9B–7B BF16 |
| **Apertus 1.5** (`swiss-ai/Apertus-v1.5-8B`) | Jul | 8B / 70B | xIELU activation, 262K ctx *[1.5 details unverified]* | Apache-2.0 (gated) | 8B BF16 |
| **EmbeddingGemma 2** (`google/embeddinggemma-2`) | Oct | 270M text (+170M vision, +300M audio) | Bidirectional Gemma-4 encoder: 24 layers, d=512, 5:1 local(1024):global, per-layer embeddings, mean pooling, Matryoshka 768→128 | Apache-2.0 | trivially |
| **Nemotron 3.5 Embed-1B / Rerank-8B** (`nvidia/Nemotron-3.5-*`) | Oct | 1B / 8B | Bidirectional Ministral-3 tower + Pixtral vision encoder, mean pool + MLP head | OpenMDW-1.1 | 1B yes, 8B BF16 yes |
| **WeVisOne-VL 2B/4B** (`tencent/WeVisOne-VL-2B`) | Oct | 2.4B / 4.8B | Qwen3-VL base + box-guided query decoder + mask head: detection, grounding, OCR, segmentation in one VLM | Apache-2.0 | yes |
| **Qwen3-ASR-1.7B** (`Qwen/Qwen3-ASR-1.7B-hf`) | Jun (HF-native) | 1.7B + 300M encoder | Whisper-style conv + attention audio encoder (24L, d=1024, GELU) + Qwen3-1.7B decoder; #1 open ASR | Apache-2.0 | yes |
| **Qwen-Image-2.1** (`Qwen/Qwen-Image-2.1`) | Sep | 7B DiT | 32 single-stream DiT layers, Qwen3-VL-8B text encoder, 64-ch VAE *[encoder/VAE from secondaries]* | Qwen Research (non-commercial) | FP8 |
| Krea 2 | Jun | 12B DiT | single-stream GQA DiT, Qwen3-VL encoder, Qwen-Image VAE | Krea 2 Community | FP8 Turbo ~12 GB |
| Breeze TTS 2 (`BreezeBlue/Breeze-TTS-2`) | Aug | 3B | arch undisclosed | non-commercial | yes |

Too large for this box, for awareness: Kolibri-1 (Aleph Alpha, 78B-A3.5B), GLM-5.3-Flash (320B-A18B), DeepSeek V4.1-Flash (552B), Kimi K3, Mistral Large 4, Muse Spark Open (816B). No Phi-5, SmolLM4, OLMo 4, Gemma 5, new Granite, YOLO27 weights, SAM 4 or open Whisper successor were found in the window (Gemma 4 and Granite 4.1 were April; YOLO26 January).

### Revised next-5 (replaces items 6–12 of the roadmap above)

1. ~~**EmbeddingGemma 2, text-only 270M**~~ — ✅ 2026-10-29 (see the table above). Was: cheapest win: bidirectional encoder on the BERT path; needs sliding-window mask (exists in `nn.SDPA`), per-layer embeddings, mean pooling. Validate like bge-small.
2. **Qwen3.8-27B at Q4** — one Gated-DeltaNet + gated-attention + MRoPE implementation unlocks Qwen3.5/3.6/3.8 and derivatives. New ops: chunked gated delta rule (linear attention), partial RoPE, 4-bit weight dequant-matmul (also needed by #3). The biggest single piece of work on the list.
3. **Muse Glimmer 30B at Q4** — plain dense GQA transformer: Llama-3 ops + sliding-window layers + NoPE global layers. Shares the 4-bit matmul with #2; Meta's first Apache-licensed line.
4. ~~**Qwen3-ASR-1.7B**~~ — done 2026-10-28 (see the verified table).
5. **Mellum 2.1 12B-A2.5B** — first MoE: top-8-of-64 router + SWA, small enough to validate MoE kernels at Q8 before Qwen3.6-35B-A3B class models.

Runner-ups: Nemotron-3.5-Embed-1B (after a Ministral-3 backbone), WeVisOne-VL-2B (after the Qwen3-VL vision tower), Qwen-Image-2.1 (DiT path after SD 1.5; non-commercial license). Mistral / Gemma 2 / Phi-4 / DeepSeek-V2 remain architecture-ready but are no longer the targets.

---

## Prerequisites for Full Coverage

| Capability | Needed By | Status |
|-----------|----------|--------|
| Depthwise / grouped Conv2d | MobileNet, ShuffleNet, EfficientNet, ResNeXt | ✅ done (Go-level group iteration; Wave 25) |
| Batch normalization (train + eval) | Almost all CNNs | ✅ native CUDA forward + backward (Wave 27) |
| Global / adaptive average pool | ResNet, MobileNet, etc. | ✅ `AdaptiveAvgPool2d` shipped |
| Dropout (autograd) | AlexNet, VGG, ViT, MobileNet | ✅ done |
| Patch embedding (Conv2d kernel=stride=patch) | ViT | ✅ done |
| Squeeze-and-Excitation | EfficientNet, MobileNet v3 | Not implemented (needs sigmoid·broadcast multiply) |
| Window attention | Swin | Not implemented |
| FPN / PANet | Faster R-CNN, RetinaNet, YOLO | ✅ PANet pattern in YOLOv5 |
| RoI Align / RoI Pool | Faster R-CNN, Mask R-CNN | Not implemented |
| Upsample (nearest) | YOLO PANet, segmentation | ✅ `ops.UpsampleNearest` |
| Bilinear upsample | DeepLab v3, segmentation | Not implemented |
| INT8 quantization | Quantized models | ✅ per-column weight quant + int8×F32 matmul (`quant/int8.go`, `ops.MatMulInt8`); no cuBLAS INT8 GEMM yet |
| Conv3d | Video models | ✅ `ops.Conv3d` (CPU + CUDA F32 forward) |
| GroupNorm / cross-attention / time embeddings | Stable Diffusion | ✅ done (SD 1.5) |
| FlashAttention | Long-context decode | ✅ `ops.FlashAttention` (CUDA tiled, 16× over SDPA at S=1024) |
| Sliding-window attention | Mistral, Gemma 2 | ✅ done 2026-10-28 (`LlamaConfig.SlidingWindow`, `SlidingLayers`; rotating KV cache) |
| RoPE scaling (llama3) | Llama-3.1+ | ✅ `models.RopeScaling`; LongRoPE (Phi-3 / Phi-4-mini) done |
| WebGPU device | non-NVIDIA GPUs | ✅ `-tags webgpu`; F32 only; GPT-2 verified |
| Tied LM head | GPT-2, GPT-J, many transformers | ✅ done (`GPT2Config.Tied`) |
| Sharded safetensors (`model.safetensors.index.json`) | Llama-2 7B+, GPT-2 large+, Qwen2.5-3B+ | ✅ done (`hub.DownloadModelSafetensors` + `safetensors.LoadFiles`); also reads HF Python cache layout (`refs/<rev>` + `snapshots/<sha>/`) |
| RoPE | Llama family | ✅ done (`ops.RoPE`, CPU + CUDA) |
| Grouped-query attention | Llama-2 70B, Mistral, Qwen2 | ✅ done (built into `models.Llama`) |
| KV cache | Efficient transformer generation | ✅ done (`models.KVCache`, concat-grown buffers) |
| Special-token-aware BPE | Chat-template prompts (`<|im_start|>` etc.) | ✅ done (`tokenizer.BPE.RegisterSpecial`, picked up from `added_tokens` in `tokenizer.json`) |
| `top-k` / `top-p` / temperature sampling | Diverse text generation | ✅ done (`models.SampleLogits` / `SampleOpts` — argmax when temp==0, top-k truncation, then nucleus, then weighted draw) |

---

## Quick reference — verified inference (RTX 5060 Ti)

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
| GPT-2 small on WebGPU | 1×16 | 124.44 M | ~2× slower than cuBLAS | `examples/25_gpt2_pretrained -device webgpu` (`-tags webgpu`) |
| Phi-3.5-mini-instruct (BF16) | 10-token chat + decode | 3 821.08 M | 164 ms prefill + 15 tok/s | `examples/35_qwen_inference -repo microsoft/Phi-3.5-mini-instruct -bf16` (7.6 GB GPU) |
| Qwen2-0.5B-Instruct **GGUF Q8_0 / Q4_K_M** | 15-token chat + decode | 494.03 M | 235 ms prefill + 17 tok/s | `examples/35_qwen_inference -gguf ~/.cache/light/gguf/qwen2-0.5b-instruct-q4_k_m.gguf` (0.4 GB + F32 embedding); logits within 0.05 (Q8_0) / 0.4 (Q4_K_M) of F32 |
| Llama-3.2-1B-Instruct **GGUF Q4_K_M** | 17-token chat + decode | 1 235.81 M | 501 ms prefill + 14 tok/s | `-gguf ~/.cache/light/gguf/llama-3.2-1b-instruct-q4_k_m.gguf -repo meta-llama/Llama-3.2-1B-Instruct`; same top-3 as F32, greedy identical |

**Validation bar (since the 2026-10-27 RoPE finding):** a model counts as verified only when its top-5 next-token logits match `scripts/verify/ref_transformers.py` on a ≥15-token chat-template prompt (`examples/35_qwen_inference -logits`), not just the top-1 of a short prompt. Qwen2-0.5B, Llama-3.2-1B (F32, 4 dp) and Phi-3.5-mini (BF16) pass this bar.
