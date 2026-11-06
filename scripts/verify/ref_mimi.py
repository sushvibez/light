#!/usr/bin/env python3
"""Mimi codec reference (transformers MimiModel, F32) for models/mimi_test.go.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify/ref_mimi.py [repo] [cpu|cuda]

Resamples testdata/whisper/front_center_16k.wav to 24 kHz once (written to
testdata/mimi/front_center_24k.wav, the file the Go test reads), encodes it
with all 32 codebooks, prints the frame count and the first 8 codebooks'
codes, then decodes the first 8 codebooks and prints audio statistics.
"""
import os, sys, wave, numpy as np, torch
from scipy.signal import resample_poly
from transformers import MimiModel

repo = sys.argv[1] if len(sys.argv) > 1 else "kyutai/mimi"
dev = sys.argv[2] if len(sys.argv) > 2 else "cuda"
src, dst = "testdata/whisper/front_center_16k.wav", "testdata/mimi/front_center_24k.wav"
if not os.path.exists(dst):
    w = wave.open(src)
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    y = resample_poly(x, 3, 2)
    out = wave.open(dst, "wb"); out.setnchannels(1); out.setsampwidth(2); out.setframerate(24000)
    out.writeframes(np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16).tobytes()); out.close()
w = wave.open(dst)
x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
print("samples:", len(x), "first4:", [round(float(v), 6) for v in x[:4]])
model = MimiModel.from_pretrained(repo, attn_implementation="eager").to(dev).eval()
audio = torch.tensor(x, device=dev)[None, None]
with torch.no_grad():
    codes = model.encode(audio).audio_codes  # (1, 32, T)
    print("frames:", codes.shape[-1])
    for q in range(8):
        print(f"codes[{q}]:", codes[0, q].tolist())
    dec = model.decode(codes[:, :8]).audio_values[0, 0].float().cpu().numpy()
    print("decoded samples:", len(dec), "first8:", [round(float(v), 6) for v in dec[:8]])
    print(f"decoded rms={np.sqrt((dec**2).mean()):.6f} maxabs={np.abs(dec).max():.6f} at={int(np.abs(dec).argmax())}")
    # Full-codebook decode too (round trip quality).
    dec32 = model.decode(codes).audio_values[0, 0].float().cpu().numpy()
    n = min(len(dec32), len(x))
    print(f"decode32 rms={np.sqrt((dec32**2).mean()):.6f} snr_db={10*np.log10((x[:n]**2).mean()/((x[:n]-dec32[:n])**2).mean()):.2f}")
