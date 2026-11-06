#!/usr/bin/env python3
"""Whisper reference (transformers) for models/whisper_test.go.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify/ref_whisper.py [repo] [wav16k]

Prints sampled log-mel values, encoder output values, the raw first-step
decoder logits (prompt <|startoftranscript|><|en|><|transcribe|><|notimestamps|>)
and the greedy transcript, all in F32 on CPU.
"""
import sys, torch, numpy as np, soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

repo = sys.argv[1] if len(sys.argv) > 1 else "openai/whisper-tiny"
wav = sys.argv[2] if len(sys.argv) > 2 else "testdata/whisper/front_center_16k.wav"
audio, sr = sf.read(wav, dtype="float32")
assert sr == 16000
proc = WhisperProcessor.from_pretrained(repo)
model = WhisperForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.float32).eval()
feats = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features  # (1, 80, 3000)
mel = feats[0]
print("samples", len(audio), "mel", tuple(mel.shape))
def row(name, v): print(f"{name}: " + " ".join(f"{x:.5f}" for x in v))
row("mel[0][0:6]", mel[0, :6].tolist()); row("mel[40][100:106]", mel[40, 100:106].tolist()); row("mel[79][140:146]", mel[79, 140:146].tolist())
print(f"mel sum {float(mel.sum()):.3f} max {float(mel.max()):.5f} min {float(mel.min()):.5f}")
with torch.no_grad():
    enc = model.model.encoder(feats).last_hidden_state[0]  # (1500, 384)
    row("enc[0][0:6]", enc[0, :6].tolist()); row("enc[70][0:6]", enc[70, :6].tolist()); row("enc[1499][0:6]", enc[1499, :6].tolist())
    print(f"enc sum {float(enc.sum()):.3f} absmean {float(enc.abs().mean()):.5f}")
    prompt = torch.tensor([[50258, 50259, 50359, 50363]])
    logits = model(input_features=feats, decoder_input_ids=prompt).logits[0, -1]
    topv, topi = logits.topk(5)
    print("step0 top5:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  id={i:<6} {proc.tokenizer.decode([i])!r:<14} logit={v:.4f}")
    out = model.generate(feats, language="en", task="transcribe", do_sample=False, max_new_tokens=32)
    print("ids:", out[0].tolist())
    print("text:", repr(proc.tokenizer.decode(out[0], skip_special_tokens=True)))
