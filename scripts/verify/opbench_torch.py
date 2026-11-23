#!/usr/bin/env python3
"""PyTorch timings of single ops at FLUX.1's 1024² geometry, for
docs/image-bench.md next to Light's TestOpBenchCUDA (same shapes, same
order). TF32 is off so F32 rows are true F32.

    .venv/bin/python scripts/verify/opbench_torch.py
"""
import time
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = "cuda"
rows = []

def bench(name, flops, nbytes, f):
    with torch.no_grad():
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        ds = []
        for _ in range(21):  # median of individually synchronized calls, as Light's bench
            t0 = time.perf_counter()
            f()
            torch.cuda.synchronize()
            ds.append((time.perf_counter() - t0) * 1e3)
    ms = sorted(ds)[len(ds) // 2]
    rate = f"{flops / (ms * 1e-3) / 1e12:.1f} TFLOPS" if flops else (f"{nbytes / (ms * 1e-3) / 1e9:.0f} GB/s" if nbytes else "")
    row = f"| {name} | {ms:.2f} ms | {rate} |"
    print(row, flush=True)
    rows.append(row)

M, K, H, Dh = 4096, 3072, 24, 128
x = torch.randn(M, K, device=dev) * 0.1
xb = x.bfloat16()
for N in (3072, 12288):
    w = torch.randn(N, K, device=dev) * 0.1
    wb, w8 = w.bfloat16(), w.to(torch.float8_e4m3fn)
    fl = 2 * M * K * N
    tag = f"({M} x {K}) · ({N} x {K})ᵀ"
    bench("GEMM F32 " + tag, fl, 0, lambda: F.linear(x, w))
    bench("GEMM BF16 in, F32 out (mixed) " + tag, fl, 0, lambda: F.linear(x.bfloat16(), wb).float())
    bench("GEMM BF16 in, BF16 out " + tag, fl, 0, lambda: F.linear(xb, wb))
    bench("FP8 weight → BF16 + GEMM " + tag, fl, 0, lambda: F.linear(xb, w8.bfloat16()))
S = 4096 + 256
q, k, v = (torch.randn(1, H, S, Dh, device=dev) * 0.1 for _ in range(3))
qb, kb, vb = q.bfloat16(), k.bfloat16(), v.bfloat16()
afl = 4 * H * S * S * Dh
atag = f"(1, {H}, {S}, {Dh})"
bench("Attention F32 " + atag, afl, 0, lambda: F.scaled_dot_product_attention(q, k, v))
bench("Attention BF16 " + atag, afl, 0, lambda: F.scaled_dot_product_attention(qb, kb, vb))
h = torch.randn(1, S, K, device=dev) * 0.1
by = S * K * 4 * 2
ntag = f"(1, {S}, {K})"
bench("LayerNorm F32 " + ntag, 0, by, lambda: F.layer_norm(h, (K,), eps=1e-6))
hq = torch.randn(1, S, H, Dh, device=dev) * 0.1
wq = torch.ones(Dh, device=dev)
bench(f"RMSNorm F32 (1, {S}, {H}, {Dh})", 0, by, lambda: F.rms_norm(hq, (Dh,), wq, 1e-6))
g = torch.randn(1, S, 4 * K, device=dev) * 0.1
bench(f"GELU(tanh) F32 (1, {S}, {4 * K})", 0, S * 4 * K * 8, lambda: F.gelu(g, approximate="tanh"))
bench("SiLU F32 " + ntag, 0, by, lambda: F.silu(h))
bench("Add F32 " + ntag, 0, S * K * 12, lambda: h + h)
sm = torch.randn(H, 1024, 1024, device=dev)
bench("Softmax F32 (24, 1024, 1024)", 0, H * 1024 * 1024 * 8, lambda: F.softmax(sm, -1))
img = torch.randn(1, 128, 512, 512, device=dev) * 0.1
cw = torch.randn(128, 128, 3, 3, device=dev) * 0.1
cfl = 2 * 128 * 128 * 9 * 512 * 512
bench("Conv2d 3x3 F32 (1, 128, 512, 512) → 128", cfl, 0, lambda: F.conv2d(img, cw, padding=1))
imgb, cwb = img.bfloat16(), cw.bfloat16()
bench("Conv2d 3x3 BF16 (1, 128, 512, 512) → 128", cfl, 0, lambda: F.conv2d(imgb, cwb, padding=1))
print("\n| Op | PyTorch | rate |\n|---|---|---|\n" + "\n".join(rows))
