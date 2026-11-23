// Candle timings of single ops at FLUX.1's 1024² geometry (same shapes
// and order as Light's ops.TestOpBenchCUDA and opbench_torch.py).
use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use std::time::Instant;

fn bench(dev: &Device, name: &str, flops: f64, bytes: f64, mut f: impl FnMut() -> Result<Tensor>) {
    let run = |f: &mut dyn FnMut() -> Result<Tensor>| -> Result<f64> {
        for _ in 0..3 {
            f()?;
        }
        dev.synchronize()?;
        let t0 = Instant::now();
        for _ in 0..10 {
            f()?;
        }
        dev.synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1e3 / 10.0)
    };
    match run(&mut f) {
        Ok(ms) => {
            let rate = if flops > 0.0 {
                format!("{:.1} TFLOPS", flops / (ms * 1e-3) / 1e12)
            } else if bytes > 0.0 {
                format!("{:.0} GB/s", bytes / (ms * 1e-3) / 1e9)
            } else {
                String::new()
            };
            println!("| {name} | {ms:.2} ms | {rate} |");
        }
        Err(e) => println!("| {name} | failed: {} | |", e.to_string().lines().next().unwrap_or("")),
    }
}

fn main() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let (m, k, h, dh) = (4096usize, 3072usize, 24usize, 128usize);
    let x = (Tensor::randn(0f32, 1f32, (m, k), &dev)? * 0.1)?;
    let xb = x.to_dtype(DType::BF16)?;
    for n in [3072usize, 12288] {
        let w = (Tensor::randn(0f32, 1f32, (n, k), &dev)? * 0.1)?;
        let wt = w.t()?;
        let wbt = w.to_dtype(DType::BF16)?.t()?;
        let fl = 2.0 * m as f64 * k as f64 * n as f64;
        let tag = format!("({m} x {k}) · ({n} x {k})ᵀ");
        bench(&dev, &format!("GEMM F32 {tag}"), fl, 0.0, || Ok(x.matmul(&wt)?));
        bench(&dev, &format!("GEMM BF16 in, F32 out (mixed) {tag}"), fl, 0.0, || {
            Ok(x.to_dtype(DType::BF16)?.matmul(&wbt)?.to_dtype(DType::F32)?)
        });
        bench(&dev, &format!("GEMM BF16 in, BF16 out {tag}"), fl, 0.0, || Ok(xb.matmul(&wbt)?));
    }
    let s = 4096 + 256;
    let q = (Tensor::randn(0f32, 1f32, (1, h, s, dh), &dev)? * 0.1)?;
    let kk = (Tensor::randn(0f32, 1f32, (1, h, s, dh), &dev)? * 0.1)?;
    let v = (Tensor::randn(0f32, 1f32, (1, h, s, dh), &dev)? * 0.1)?;
    let afl = 4.0 * h as f64 * (s * s) as f64 * dh as f64;
    let atag = format!("(1, {h}, {s}, {dh})");
    // What candle's flux model does: scores, softmax, values.
    let attn = |q: &Tensor, k: &Tensor, v: &Tensor| -> Result<Tensor> {
        let scale = 1.0 / (dh as f64).sqrt();
        let w = (q.matmul(&k.t()?)? * scale)?;
        Ok(candle_nn::ops::softmax_last_dim(&w)?.matmul(v)?)
    };
    bench(&dev, &format!("Attention F32 {atag}"), afl, 0.0, || attn(&q, &kk, &v));
    let (qb, kb, vb) = (q.to_dtype(DType::BF16)?, kk.to_dtype(DType::BF16)?, v.to_dtype(DType::BF16)?);
    bench(&dev, &format!("Attention BF16 {atag}"), afl, 0.0, || attn(&qb, &kb, &vb));
    let hx = (Tensor::randn(0f32, 1f32, (1, s, k), &dev)? * 0.1)?;
    let by = (s * k) as f64 * 8.0;
    let ntag = format!("(1, {s}, {k})");
    let (ones, zeros) = (Tensor::ones(k, DType::F32, &dev)?, Tensor::zeros(k, DType::F32, &dev)?);
    bench(&dev, &format!("LayerNorm F32 {ntag}"), 0.0, by, || Ok(candle_nn::ops::layer_norm(&hx, &ones, &zeros, 1e-6)?));
    let hq = (Tensor::randn(0f32, 1f32, (1, s, h, dh), &dev)? * 0.1)?;
    let wq = Tensor::ones(dh, DType::F32, &dev)?;
    bench(&dev, &format!("RMSNorm F32 (1, {s}, {h}, {dh})"), 0.0, by, || Ok(candle_nn::ops::rms_norm(&hq, &wq, 1e-6)?));
    let g = (Tensor::randn(0f32, 1f32, (1, s, 4 * k), &dev)? * 0.1)?;
    bench(&dev, &format!("GELU(tanh) F32 (1, {s}, {})", 4 * k), 0.0, (s * 4 * k) as f64 * 8.0, || Ok(g.gelu()?));
    bench(&dev, &format!("SiLU F32 {ntag}"), 0.0, by, || Ok(hx.silu()?));
    bench(&dev, &format!("Add F32 {ntag}"), 0.0, (s * k) as f64 * 12.0, || Ok((&hx + &hx)?));
    let sm = Tensor::randn(0f32, 1f32, (h, 1024, 1024), &dev)?;
    bench(&dev, "Softmax F32 (24, 1024, 1024)", 0.0, (h * 1024 * 1024) as f64 * 8.0, || Ok(candle_nn::ops::softmax(&sm, D::Minus1)?));
    let img = (Tensor::randn(0f32, 1f32, (1, 128, 512, 512), &dev)? * 0.1)?;
    let cw = (Tensor::randn(0f32, 1f32, (128, 128, 3, 3), &dev)? * 0.1)?;
    let cfl = 2.0 * 128.0 * 128.0 * 9.0 * 512.0 * 512.0;
    bench(&dev, "Conv2d 3x3 F32 (1, 128, 512, 512) → 128", cfl, 0.0, || Ok(img.conv2d(&cw, 1, 1, 1, 1)?));
    let (imgb, cwb) = (img.to_dtype(DType::BF16)?, cw.to_dtype(DType::BF16)?);
    bench(&dev, "Conv2d 3x3 BF16 (1, 128, 512, 512) → 128", cfl, 0.0, || Ok(imgb.conv2d(&cwb, 1, 1, 1, 1)?));
    Ok(())
}
