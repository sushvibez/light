# Porting log: GoLight (accelerate/light) → Light (this repo)

Decision 2026-10-27: this repo is the main line. `workspace/accelerate/light`
("GoLight": generic `Tensor[T]`, WebGPU first, CUDA bolted on in June 2026)
is archived. Reason: with CUDA 13.1 reinstalled, this repo ran Qwen2-0.5B at
~31 tok/s and SD 1.5 correctly on the RTX 5060 Ti, while GoLight's CUDA path
was wrong on Qwen and ~300x slower. GoLight's reusable assets are being
ported here; its parity planning doc is kept at `docs/history/golight-parity.md`.

## State preserved

- The May-2026 "inference-only" refactor (deleted `autograd/`, `training/`
  and the training examples) was never committed. It is kept as
  `git stash@{0}` and as `../light-inference-only-wip-2026-05-08.patch`.
  `main` keeps autograd + training.

## Done on main (uncommitted working tree, 2026-10-27)

| Item | Files | Verified |
|------|-------|----------|
| Llama-3.x loader (`LoadLlama3FromHF`, `LoadCausalLMFromHF`, `EOSTokenIDs`, `BOSTokenID`) | `models/llama3_load.go` | Llama-3.2-1B-Instruct: "The capital of France is" → " Paris" (matches transformers top-1, logit 19.06); chat prompt "What is 2+2?" → "The answer is 4."; F32 29 tok/s, BF16 32 tok/s |
| Llama-3 RoPE frequency scaling (`rope_scaling.rope_type = llama3`) | `models/rope_scaling.go`, `ops.RoPEWithInvFreq`, `kernels/cpu/rope_scaled.go`, `kernels/cuda/rope_scaled.go`, `kernels/cuda/src/rope_scaled.cu` (+ `ptx/rope_scaled.ptx`) | inv-freq table matches transformers to 1e-6; CPU/CUDA F32 + BF16 parity tests |
| Example 35 generalised: loader by `model_type`, EOS/BOS from config | `examples/35_qwen_inference/main.go` | Qwen2-0.5B output unchanged |
| LayerNorm / RMSNorm take the autograd path when a gradient is wanted (training through norms had silently broken in May) | `nn/norm.go` | `TestGradCheckTransformerBlock` passes again; SD 1.5 BF16 still renders |
| `cudart.MemInfo` stub for `-tags nocuda` so examples 35/36/37 build without CUDA | `internal/cudart/meminfo_nocuda.go` | `go build -tags nocuda ./...` |
| WordPiece + Unigram tokenizers, `tokenizer.Load` dispatch, BERT post-processor | `tokenizer/{tokenizer,wordpiece,unigram,normalizer,postprocessor,specials,accents_table}.go` | exact match vs HF `tokenizers` on bge-small-en-v1.5 (10 sentences) and a synthetic Unigram |
| GoLight parity doc + verification harness | `docs/history/golight-parity.md`, `scripts/verify/` | |
| SDPA + CausalMask / SlidingWindowMask / ALiBi | `nn/sdpa.go` | SDPA == MultiHeadAttention to 1e-5; slopes match paper |
| FlashAttention (CPU online softmax; CUDA tiled kernels for D∈{32,64,128}, generic D≤256) | `ops/flash_attention.go`, `kernels/cuda/src/flash_attn.cu` | CPU↔CUDA and FA↔SDPA parity over S∈{1,7,64,257}×D∈{32,64,128}; S=1024 D=64: 0.84 ms vs 13.5 ms SDPA (16×), causal 4.0× |
| Pad / Flip / Roll / Sort / Lp Norm | `ops/shape_ops.go` (+ CUDA F32 kernels; Sort is a host round-trip on CUDA) | parity tests |
| Einsum (general CPU; 1–2 operand F32 lowered to batched matmul on CUDA, else host fallback) | `ops/einsum.go` | 15 notations |
| Conv3d, InstanceNorm2d | `ops/conv3d.go`, `ops/instancenorm.go`, `nn/instancenorm.go` | CPU/CUDA parity |
| Int8 per-column weight quant + int8×f32 matmul | `quant/int8.go`, `ops/matmul_int8.go`, `kernels/cuda/src/matmul_int8.cu` | rel-err-norm < 0.01 vs F32 |

| WebGPU as a third device (`-tags webgpu`, opt-in; `device.WebGPU`, `storage.NewWebGPU`, `tensor.To`, per-kind Contiguous/Copy hooks) with 16 WGSL shaders: elementwise + broadcast, unary, matmul (2-D with transB, batched), softmax/logsoftmax, layernorm/rmsnorm, reductions, index_select, concat, contiguous, conv2d, pools, batchnorm eval, casts | `device/webgpu*.go`, `storage/webgpu.go`, `tensor/{alloc,to,copy,contig_hook}*.go`, `internal/wgpudev/`, `kernels/webgpu/` | 108 parity tests vs CPU; GPT-2 greedy identical to CUDA/CPU (~2× slower than cuBLAS). Buffer limit: GoLight echoed the adapter's u64::MAX sentinel and got 256 MB; requesting min(binding limit, 2 GiB) is granted (2048 MiB alloc + bind). Beyond 2 GiB per tensor is a WebGPU ceiling. F64/F16/BF16, rope, conv1d, gather, cumsum, compare, groupnorm not registered on WebGPU yet. |

| BERT encoder (`BertModel`, `BertForMaskedLM`, `LoadBertFromHF`, CLS/mean pooling, L2 normalize) + exact-erf `ops.GeluErf` (CPU F32/F64, CUDA F32/F64) | `models/bert.go`, `models/bert_load.go`, `examples/40_bert_embeddings`, `scripts/verify/ref_bert.py` | `unsloth/bge-small-en-v1.5`: Go vs PyTorch max abs err 7.8e-7 (hidden) / 4.8e-7 (embedding) on CPU and CUDA; cosine matrix identical to 4 dp; WordPiece ids identical. MLM head shape-tested only (no MLM checkpoint cached). |
| CUDA driver-context fix: driver calls (module load, launch, sync) now bind the primary context on the calling thread | `internal/cudart/driver.go` | removed an intermittent `cuModuleLoadDataEx: invalid device context (201)` seen when many test binaries start together; 3 consecutive full-suite runs clean |

All four worktrees (`../wt-webgpu`, `../wt-ops`, `../wt-tokenizers`, `../wt-bert`)
are integrated into `main`'s working tree; their patches are kept beside the
repo as `../port-*.patch`. The worktrees can be removed once `main` is committed.

## Integration

Agents cannot commit in this repo (`.claude/settings.json` denies it), so each
worktree is a dirty tree on top of `main`. Integrate each as a patch:

```bash
cd ../wt-<name> && git add -A && git diff --cached > ../port-<name>.patch && git reset -q
cd ../light && git apply --3way ../port-<name>.patch
go build ./... && go test ./...
```

All four patches (`port-tokenizers`, `port-ops`, `port-webgpu`, `port-bert`)
have been applied to `main` (the ops loader hunk was merged by hand into
`kernels/cuda/loader.go`; duplicate hunks for `nn/norm.go`,
`meminfo_nocuda.go` and `tokenizer/*` were excluded). Verification after
integration: `go build` under default, `-tags nocuda` and `-tags webgpu`;
`go test ./...` (CUDA) clean three times in a row; `go test -tags webgpu`
clean; Qwen2-0.5B, Llama-3.2-1B (F32 + BF16), SD 1.5 BF16, ResNet-50
pretrained, BERT/bge-small and GPT-2 on WebGPU all produce the expected
outputs.

## Still to port from GoLight (drafts, no real-weight validation there)

T5, Whisper, Gemma 2, Falcon, CLIP, YOLO, UNet architecture code; the
`op × dtype × device` matrix test style; the `examples/verify` numpy
reference for Llama. None of these had HF-weight validation in GoLight.
