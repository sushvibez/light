#!/usr/bin/env python3
"""Qwen3-ASR reference (transformers >= 5.13) for models/qwen3_asr_test.go.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify/ref_qwen3_asr.py [repo] [wav16k]

Prints the mel filterbank samples, log-mel samples, the prompt token ids
produced by the processor (auto-language), audio-encoder and projector
samples, the first-step top-5 logits and the greedy transcript. F32 on CPU.
"""
import sys, torch, numpy as np, soundfile as sf
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-ASR-1.7B-hf"
wav = sys.argv[2] if len(sys.argv) > 2 else "testdata/whisper/front_center_16k.wav"
audio, sr = sf.read(wav, dtype="float32")
assert sr == 16000
proc = AutoProcessor.from_pretrained(repo)
fe = proc.feature_extractor
def row(name, v): print(f"{name}: " + " ".join(f"{x:.5f}" for x in v))
fb = fe.mel_filters  # (201, 128)
print("filters", fb.shape)
row("fb[:,0][0:6]", fb[:6, 0]); row("fb[:,10][8:14]", fb[8:14, 10]); row("fb[:,127][190:196]", fb[190:196, 127])
print(f"fb sum {float(fb.sum()):.6f}")
inputs = proc.apply_transcription_request(audio=audio, language=None, return_tensors="pt")
ids = inputs["input_ids"][0].tolist()
feats = inputs["input_features"]; mask = inputs["input_features_mask"]
print("samples", len(audio), "mel", tuple(feats.shape), "valid frames", int(mask.sum()))
mel = feats[0]
row("mel[0][0:6]", mel[0, :6].tolist()); row("mel[40][100:106]", mel[40, 100:106].tolist()); row("mel[127][140:146]", mel[127, 140:146].tolist())
print(f"mel sum {float(mel.sum()):.3f}")
n_audio = sum(1 for i in ids if i == proc.audio_token_id)
print("prompt len", len(ids), "audio tokens", n_audio)
print("ids:", ids[:12], "...", ids[-12:])
print("prompt text:", repr(proc.tokenizer.decode(ids)))
model = Qwen3ASRForConditionalGeneration.from_pretrained(repo, dtype=torch.float32).eval()
with torch.no_grad():
    af = model.get_audio_features(feats, mask, return_dict=True)
    enc = af.last_hidden_state; proj = af.pooler_output
    print("enc", tuple(enc.shape))
    row("enc[0][0:6]", enc[0, :6].tolist()); row("enc[10][0:6]", enc[10, :6].tolist()); row(f"enc[{enc.shape[0]-1}][0:6]", enc[-1, :6].tolist())
    print(f"enc sum {float(enc.sum()):.3f} absmean {float(enc.abs().mean()):.5f}")
    row("proj[0][0:6]", proj[0, :6].tolist()); row(f"proj[{proj.shape[0]-1}][0:6]", proj[-1, :6].tolist())
    logits = model(**inputs).logits[0, -1]
    topv, topi = logits.topk(5)
    print("step0 top5:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  id={i:<6} {proc.tokenizer.decode([i])!r:<14} logit={v:.4f}")
    out = model.generate(**inputs, do_sample=False, max_new_tokens=48)
    gen = out[0, len(ids):].tolist()
    print("gen ids:", gen)
    print("text:", repr(proc.tokenizer.decode(gen, skip_special_tokens=False)))
    print("parsed:", proc.batch_decode(out[:, len(ids):], skip_special_tokens=True))
