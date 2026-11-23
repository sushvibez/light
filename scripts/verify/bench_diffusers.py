#!/usr/bin/env python3
"""Full-generation timings with diffusers on the same card, for
docs/image-bench.md: the same prompt, size and steps as Light's
TestImageBenchCUDA, cold start, loads included. Each row is one process
phase: the pipeline loads with model CPU offload (one component on the
card at a time, like Light's sequential loads), the transformer from the
same FP8 / GGUF files Light uses where given (diffusers runs them at BF16
compute: FP8 through layerwise casting, GGUF dequantized per matmul).

FP8 rows load the whole transformer at FP8 (12 GB for FLUX.1, as the file
is) and then fp8_runtime() makes it runnable: diffusers' own
enable_layerwise_casting skips norm / embedding modules by name, which
then stay FP8 and fail in the first matmul or add. The biases and norm
scales pass through FP8 on the way, so these rows are timings, not
references.

    .venv/bin/python scripts/verify/bench_diffusers.py all
    .venv/bin/python scripts/verify/bench_diffusers.py "kind,repo,transformer,size,steps,guidance;..."
"""
import os, sys, time
import torch

prompt = "a watercolour painting of a red lighthouse on a rocky shore at dawn, soft light"
lib = os.path.join(os.environ.get("LIGHT_MODELS_DIR") or os.path.expanduser("~/.cache/light/models"), "checkpoints")
ALL = [
    ("zimage", "Tongyi-MAI/Z-Image-Turbo", "", 1024, 8, 0.0),
    ("flux2", "black-forest-labs/FLUX.2-klein-4B", "", 1024, 4, 0.0),
    ("flux2", "black-forest-labs/FLUX.2-klein-4B", f"{lib}/flux2-klein-4b-fp8", 1024, 4, 0.0),
    ("flux", "black-forest-labs/FLUX.1-schnell", f"{lib}/flux1-schnell-fp8.safetensors", 1024, 4, 0.0),
    ("flux", "black-forest-labs/FLUX.1-schnell", f"{lib}/flux1-schnell-Q4_K_S.gguf", 1024, 4, 0.0),
    ("flux", "black-forest-labs/FLUX.1-dev", f"{lib}/flux1-dev-fp8.safetensors", 1024, 28, 3.5),
    ("flux", "black-forest-labs/FLUX.1-dev", f"{lib}/flux1-dev-Q4_K_S.gguf", 1024, 28, 3.5),
]


def fp8_runtime(model):
    """Store every Linear / Conv / Embedding weight at FP8 and widen it to
    BF16 around its forward (no module skipped), and cast what has no such
    hook (norm scales, bare parameters, buffers) to BF16 once."""
    from diffusers.hooks import apply_layerwise_casting
    f8, bf = torch.float8_e4m3fn, torch.bfloat16
    apply_layerwise_casting(model, storage_dtype=f8, compute_dtype=bf, skip_modules_pattern=())
    for m in model.modules():
        reg = getattr(m, "_diffusers_hook", None)
        if reg is not None and reg.get_hook("layerwise_casting") is not None:
            continue
        for store in (m._parameters, m._buffers):
            for t in store.values():
                if t is not None and t.dtype == f8:
                    t.data = t.data.to(bf)
    return model


def chain(e):
    """The exception with its causes: diffusers reports a failed GGUF read
    as "Unable to load weights" and keeps the reason in the context."""
    parts, seen = [], set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        parts.append(f"{type(e).__name__}: {str(e)[:160]}")
        e = e.__cause__ or e.__context__
    return " <- ".join(parts)


spec = sys.argv[1] if len(sys.argv) > 1 else "all"
runs = ALL if spec == "all" else [(f[0], f[1], f[2], int(f[3]), int(f[4]), float(f[5])) for f in (r.split(",") for r in spec.split(";"))]

rows = []
for kind, repo, tf, size, steps, guidance in runs:
    name = repo.split("/")[-1]
    variant = "BF16"
    t0 = time.time()
    try:
        if kind == "flux":
            from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
            kw = {}
            if tf.endswith(".gguf"):
                variant = "Q4_K_S GGUF"
                import gguf  # noqa: F401  (diffusers needs it and hides its absence)
                kw["transformer"] = FluxTransformer2DModel.from_single_file(
                    tf, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16), torch_dtype=torch.bfloat16)
            elif tf:
                variant = "FP8"
                kw["transformer"] = fp8_runtime(FluxTransformer2DModel.from_single_file(
                    tf, config=repo, subfolder="transformer", torch_dtype=torch.float8_e4m3fn))
            else:
                variant = "BF16 (offload)"
            pipe = FluxPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16, **kw)
            seq = 256 if "schnell" in repo else 512
            call = lambda cb: pipe(prompt=prompt, height=size, width=size, num_inference_steps=steps, guidance_scale=guidance,
                                   max_sequence_length=seq, output_type="pt", callback_on_step_end=cb)
        elif kind == "flux2":
            from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
            kw = {}
            if tf:
                variant = "FP8"
                kw["transformer"] = fp8_runtime(Flux2Transformer2DModel.from_pretrained(tf, torch_dtype=torch.float8_e4m3fn))
            pipe = Flux2KleinPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16, **kw)
            call = lambda cb: pipe(prompt=prompt, height=size, width=size, num_inference_steps=steps, output_type="pt", callback_on_step_end=cb)
        elif kind == "zimage":
            from diffusers import ZImagePipeline
            pipe = ZImagePipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
            call = lambda cb: pipe(prompt=prompt, height=size, width=size, num_inference_steps=steps, guidance_scale=guidance,
                                   output_type="pt", callback_on_step_end=cb)
        else:
            raise ValueError(kind)
        pipe.enable_model_cpu_offload()
        marks = []
        def cb(p, i, t, kw):
            torch.cuda.synchronize()
            marks.append(time.time())
            return {}
        with torch.no_grad():
            call(cb)
        torch.cuda.synchronize()
        total = time.time() - t0
        per = ""
        if len(marks) > 1:
            per = "%.1f s" % ((marks[-1] - marks[0]) / (len(marks) - 1))
        row = f"| {name} | {variant} | {size}² | {steps} | {per} | {int(total // 60)}m{int(total % 60):02d}s |"
    except Exception as e:  # noqa: BLE001
        print("failed:", name, variant, chain(e), file=sys.stderr, flush=True)
        row = f"| {name} | {variant} | {size}² | {steps} | failed: {chain(e)[:200]} |"
    print(row, flush=True)
    rows.append(row)
    try:
        del pipe
    except NameError:
        pass
    torch.cuda.empty_cache()
print("\n| Model | Transformer | Size | Steps | Per step | Full generation |\n|---|---|---|---|---|---|")
print("\n".join(rows))
