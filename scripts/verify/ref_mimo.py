#!/usr/bin/env python3
"""MiMo-7B reference logits for models/mimo_test.go, without remote code.

    .venv/bin/python scripts/verify/ref_mimo.py <repo_or_dir> [bf16|f32] [--ids 1,2,3 | --chat] [--cpu]
    .venv/bin/python scripts/verify/ref_mimo.py --make-tiny <dir>

XiaomiMiMo/MiMo-7B-* ship `modeling_mimo.py` (trust_remote_code), whose
MiMoForCausalLM is Qwen2ForCausalLM plus `model.mtp_layers` that the forward
pass never calls (they are for multi-token-prediction speculative decoding).
This script therefore runs transformers' own Qwen2ForCausalLM on the
checkpoint (a Qwen2Config built from its config.json): the same forward,
without executing downloaded code. The mtp_layers weights are reported as
unused and dropped.

--make-tiny writes a seeded tiny-random Qwen2 relabelled as `mimo`
(MiMoForCausalLM, num_nextn_predict_layers 1) with random
`model.mtp_layers.0.*` weights added, as in the real checkpoints, so the
loader must route `mimo` to Qwen2 and ignore the MTP weights.

--chat renders the validation-bar prompt (system + user) with the repo's
chat template and decodes 16 greedy tokens. Runs on CUDA when the weights
fit, else on the CPU (--cpu forces it).
"""
import json, os, sys, torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2TokenizerFast

if sys.argv[1] == "--make-tiny":
    from safetensors.torch import load_file, save_file
    out = sys.argv[2]
    cfg = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      rope_theta=640000.0, rms_norm_eps=1e-5, tie_word_embeddings=False,
                      bos_token_id=1, eos_token_id=2)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name:
                p.add_(0.2 * torch.randn_like(p))
            elif name.endswith(".bias"):
                p.add_(0.5 * torch.randn_like(p))
    m.save_pretrained(out, safe_serialization=True)
    path = os.path.join(out, "model.safetensors")
    sd = load_file(path)
    H, I = cfg.hidden_size, cfg.intermediate_size
    hd = H // cfg.num_attention_heads
    g = torch.Generator().manual_seed(1)
    r = lambda *s: 0.02 * torch.randn(*s, generator=g)
    mtp = "model.mtp_layers.0."
    for n in ("input_layernorm", "post_attention_layernorm", "token_layernorm", "hidden_layernorm", "final_layernorm"):
        sd[mtp + n + ".weight"] = torch.ones(H)
    sd[mtp + "input_proj.weight"] = r(H, 2 * H)
    sd[mtp + "self_attn.q_proj.weight"], sd[mtp + "self_attn.q_proj.bias"] = r(H, H), r(H)
    kv = cfg.num_key_value_heads * hd
    for n in ("k_proj", "v_proj"):
        sd[mtp + f"self_attn.{n}.weight"], sd[mtp + f"self_attn.{n}.bias"] = r(kv, H), r(kv)
    sd[mtp + "self_attn.o_proj.weight"] = r(H, H)
    sd[mtp + "mlp.gate_proj.weight"], sd[mtp + "mlp.up_proj.weight"] = r(I, H), r(I, H)
    sd[mtp + "mlp.down_proj.weight"] = r(H, I)
    save_file(sd, path, metadata={"format": "pt"})
    cj = os.path.join(out, "config.json")
    c = json.load(open(cj))
    c.update(model_type="mimo", architectures=["MiMoForCausalLM"], num_nextn_predict_layers=1)
    json.dump(c, open(cj, "w"), indent=2)
    print("wrote", out)
    sys.exit(0)

repo = sys.argv[1]
dt = torch.float32 if (len(sys.argv) > 2 and sys.argv[2] == "f32") else torch.bfloat16
chat = "--chat" in sys.argv
ids = None
if "--ids" in sys.argv:
    ids = [int(x) for x in sys.argv[sys.argv.index("--ids") + 1].split(",")]
if os.path.isdir(repo):
    cfg_path = os.path.join(repo, "config.json")
else:
    from huggingface_hub import hf_hub_download
    cfg_path = hf_hub_download(repo, "config.json")
raw = json.load(open(cfg_path))
for k in ("model_type", "architectures", "auto_map", "num_nextn_predict_layers"):
    raw.pop(k, None)
cfg = Qwen2Config(**raw)
# Not AutoTokenizer: it reads config.json's auto_map and asks to run the
# repo's configuration_mimo.py. The tokenizer files are plain Qwen2 ones.
tok = None if ids is not None else Qwen2TokenizerFast.from_pretrained(repo)
model = Qwen2ForCausalLM.from_pretrained(repo, config=cfg, dtype=dt, attn_implementation="eager").eval()
dev = "cpu"
if torch.cuda.is_available() and "--cpu" not in sys.argv:
    need = sum(p.numel() * p.element_size() for p in model.parameters())
    free, _ = torch.cuda.mem_get_info()
    if need < free - (2 << 30):
        dev = "cuda"
    else:
        print(f"weights {need / 2**30:.1f} GiB > free GPU {free / 2**30:.1f} GiB - 2: running on the CPU")
model = model.to(dev)
if chat:
    msgs = [{"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Explain in one sentence why the sky is blue."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
elif ids is None:
    ids = tok("The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is").input_ids
print("model class:", type(model).__name__, "(MiMo forward = Qwen2 forward)")
print("prompt ids:", ids)
x = torch.tensor([ids], device=dev)
with torch.no_grad():
    logits = model(input_ids=x).logits[0, -1].float()
    topv, topi = logits.topk(5)
    print(f"dtype={dt} device={dev} top5:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  id={i:<7} {(tok.decode([i]) if tok else '')!r:<14} logit={v:.4f}")
    out = model.generate(x, max_new_tokens=16 if chat else 8, do_sample=False)
    if tok:
        print("greedy:", repr(tok.decode(out[0, len(ids):], skip_special_tokens=False)))
    print("greedy ids:", out[0, len(ids):].tolist())
