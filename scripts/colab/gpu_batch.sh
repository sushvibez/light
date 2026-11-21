#!/usr/bin/env bash
# One-shot GPU batch for a fresh Colab VM (run over `colab ssh` / `colab
# console`, or from a notebook cell with `!bash ...`): installs Go, clones or
# updates the repo, points cgo at CUDA, smoke-tests a PTX launch, then runs
# the pending GPU checks group by group. Each group downloads only the
# checkpoints it needs and deletes them afterwards, so the local disk never
# holds more than one model. Logs go to $LOGS; a summary prints at the end.
#
#   bash scripts/colab/gpu_batch.sh                # all groups
#   bash scripts/colab/gpu_batch.sh core int4      # chosen groups
#   KEEP_MODELS=1 bash scripts/colab/gpu_batch.sh  # keep downloads
#
# Groups: core (CUDA parity of ops / nn / quant and the tiny-model CUDA
# branches; no downloads), int4 (real GPTQ / AWQ / act-order checkpoints,
# ~2 GB), mimo (MiMo-7B-RL MTP speculative decoding in the engine, 15.6 GB),
# moshi (per-stage frame profile + CPU profile, 8.2 GB + Mimi), speech
# (end-to-end speech-to-speech through the engine's OpenAI handler, same
# checkpoints; not in the default set), trellis (TRELLIS image to Gaussians / mesh against
# TRELLIS's own code), idol (three demo objects) and idolcheck (Idol's
# real weights against idol-sdk; after idol). NO_UPDATE=1 runs the
# checkout in $WORK as is.
# Everything fits an L4 (24 GB); an A100 is not needed, except flux2dev
# (FLUX.2-dev: the 32B transformer and the Mistral-3 24B encoder, 112 GB of
# BF16 weights): an H100 (80 GB), ~100 GB of downloads, HF_TOKEN (gated).
# Hugging Face downloads use $HF_TOKEN when set.
set -uo pipefail

# `colab run` stops a session that prints nothing for 30 s; downloads and
# the long tests are silent for minutes, so a heartbeat keeps it alive.
( while true; do sleep 20; printf '[%s heartbeat]\n' "$(date +%T)"; done ) &
HEARTBEAT=$!
trap 'kill $HEARTBEAT 2>/dev/null' EXIT

REPO_URL=${REPO_URL:-https://github.com/sushvibez/light}
BRANCH=${BRANCH:-main}
WORK=${WORK:-/content/light}
LOGS=${LOGS:-/content/logs}
HF_HUB=$HOME/.cache/huggingface/hub
SEL=("$@")
[ ${#SEL[@]} -eq 0 ] && SEL=(core int4 mimo moshi)
mkdir -p "$LOGS" "$HF_HUB"
STAMP=$(date +%Y%m%d-%H%M%S)
declare -a SUMMARY

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# --- Go ---------------------------------------------------------------------
say "Go"
want=$(curl -fsSL 'https://go.dev/VERSION?m=text' | head -1)
if [ "$(/usr/local/go/bin/go env GOVERSION 2>/dev/null)" != "$want" ]; then
	curl -fsSL "https://go.dev/dl/$want.linux-amd64.tar.gz" -o /tmp/go.tgz &&
		rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tgz && rm /tmp/go.tgz || exit 1
fi
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH CGO_ENABLED=1
go version || exit 1

# --- repo -------------------------------------------------------------------
say "repo"
if [ -n "${NO_UPDATE:-}" ] && [ -d "$WORK/.git" ]; then
	echo "NO_UPDATE: using $WORK as is"
elif [ -d "$WORK/.git" ]; then
	git -C "$WORK" fetch --depth 1 origin "$BRANCH" && git -C "$WORK" reset --hard FETCH_HEAD && git -C "$WORK" clean -fdq
else
	git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$WORK" || exit 1
fi
cd "$WORK" || exit 1
git log -1 --format='%h %s (%cd)'

# --- cgo: CUDA, cuBLAS, cuDNN -------------------------------------------------
# torch keeps the environment it came with (its own cuDNN, the driver path).
ORIG_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
say "cgo"
python3 -c 'import nvidia.cudnn' 2>/dev/null || ls /usr/include/cudnn.h >/dev/null 2>&1 ||
	pip install -q "nvidia-cudnn-cu$(/usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \([0-9]*\).*/\1/p')"
eval "$(python3 - <<'PY'
import glob, os, site, sys
H = ["cuda.h", "cuda_runtime.h", "cublas_v2.h", "cublasLt.h", "cudnn.h"]
L = ["cuda", "cudart", "cublas", "cublasLt", "cudnn"]
dirs = ["/usr/lib64-nvidia", "/usr/local/cuda/include", "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/include", "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/include", "/usr/include/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/usr/local/nvidia/lib64"]
for sp in sorted(set(site.getsitepackages() + [p for p in sys.path if p.endswith("-packages")])):
    dirs += sorted(glob.glob(f"{sp}/nvidia/*/include")) + sorted(glob.glob(f"{sp}/nvidia/*/lib"))
dirs = [d for d in dirs if os.path.isdir(d)]
hs, ls = {}, {}
for d in dirs:
    for h in H:
        if h not in hs and os.path.isfile(f"{d}/{h}"):
            hs[h] = d
    for l in L:
        hits = sorted(glob.glob(f"{d}/lib{l}.so*"))
        if l not in ls and hits:
            ls[l] = hits[0]
miss = [h for h in H if h not in hs] + [l for l in L if l not in ls]
if miss:
    print(f"echo 'missing: {miss}'; exit 1")
    sys.exit()
link = "/content/cudalinks"
os.makedirs(link, exist_ok=True)
for l, p in ls.items():
    q = f"{link}/lib{l}.so"
    if os.path.lexists(q):
        os.remove(q)
    os.symlink(p, q)
inc = list(dict.fromkeys(hs.values()))
lib = list(dict.fromkeys(os.path.dirname(p) for p in ls.values()))
print(f"export CGO_CFLAGS='{' '.join('-I' + d for d in inc)}'")
print(f"export CGO_LDFLAGS='{' '.join('-L' + d for d in [link] + lib)}'")
print(f"export LD_LIBRARY_PATH='{':'.join(lib)}:{os.environ.get('LD_LIBRARY_PATH', '')}'")
PY
)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# --- smoke test -------------------------------------------------------------
say "smoke test"
if ! go test -count=1 -run 'TestDeviceCount|TestLaunchAddF32' ./internal/cudart; then
	echo "PTX launch failed: rebuilding PTX with Colab's nvcc"
	bash scripts/build_ptx.sh && go test -count=1 -run 'TestLaunchAddF32' ./internal/cudart || exit 1
fi

# --- helpers ----------------------------------------------------------------
# fetch repo [glob ...]: download (all of it, or only the globs).
fetch() {
	local repo=$1
	shift
	python3 - "$repo" "$@" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo, pats = sys.argv[1], sys.argv[2:]
ignore = ["*.onnx", "onnx/*", "*.msgpack", "*.h5", "flax_model*", "tf_model*", "*.ot", "coreml/*", "original/*"]
snapshot_download(repo, allow_patterns=pats or None, ignore_patterns=None if pats else ignore,
                  token=os.environ.get("HF_TOKEN") or None)
print("fetched", repo)
PY
}
drop() { [ -n "${KEEP_MODELS:-}" ] || for r in "$@"; do rm -rf "$HF_HUB/models--${r//\//--}"; done; }

# run name pkgs regex [extra go test args]: heavy tests on, verbose, logged.
run() {
	local name=$1 pkgs=$2 re=$3
	shift 3
	local log="$LOGS/$STAMP-$name.log" t0=$SECONDS
	say "$name"
	LIGHT_HEAVY_TESTS=1 HF_HUB_OFFLINE=1 go test -count=1 -timeout 120m -v -run "$re" "$@" $pkgs 2>&1 | tee "$log" |
		grep -E '^(=== RUN|\s*--- |ok|FAIL|panic)|_test\.go:[0-9]+:'
	local st=${PIPESTATUS[0]}
	local pass fail skip
	pass=$(grep -c -- '--- PASS' "$log")
	fail=$(grep -c -- '--- FAIL' "$log")
	skip=$(grep -c -- '--- SKIP' "$log")
	SUMMARY+=("$(printf '%-8s %s  pass %d  fail %d  skip %d  %ds  %s' "$name" "$([ "$st" = 0 ] && echo OK || echo FAIL)" "$pass" "$fail" "$skip" $((SECONDS - t0)) "$log")")
}

# --- groups -----------------------------------------------------------------
for g in "${SEL[@]}"; do
	case $g in
	core)
		run core-ops "./ops ./nn ./quant" .
		run core-models ./models 'MiniMax|DeepseekV2Tiny|DeepseekRouting|MiMoTiny|MiMoMTPDraft|SpecGreedy|Int4MoETiny|MuseGlimmerGGUFTiny'
		;;
	int4)
		repos=(Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4 Qwen/Qwen2.5-0.5B-Instruct-AWQ TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ)
		for r in "${repos[@]}"; do fetch "$r"; done
		run int4 ./models 'Int4Checkpoints'
		drop "${repos[@]}"
		;;
	mimo)
		fetch XiaomiMiMo/MiMo-7B-RL
		run mimo ./engine 'TestGenerateMiMoSpeculativeBF16'
		drop XiaomiMiMo/MiMo-7B-RL
		;;
	moshi)
		fetch kyutai/moshiko-candle-q8 model.q8.gguf tokenizer_spm_32k_3.model
		fetch kyutai/mimi
		LIGHT_MOSHI_FRAMES=43 run moshi ./models 'TestMoshikoProfile' -cpuprofile "$LOGS/$STAMP-moshi.cpu.pprof"
		go tool pprof -top -nodecount=40 models.test "$LOGS/$STAMP-moshi.cpu.pprof" >"$LOGS/$STAMP-moshi.pprof.txt" 2>&1 ||
			go tool pprof -top -nodecount=40 "$LOGS/$STAMP-moshi.cpu.pprof" >"$LOGS/$STAMP-moshi.pprof.txt" 2>&1
		head -50 "$LOGS/$STAMP-moshi.pprof.txt"
		drop kyutai/moshiko-candle-q8 kyutai/mimi
		;;
	speech)
		fetch kyutai/moshiko-candle-q8 model.q8.gguf tokenizer_spm_32k_3.model
		fetch kyutai/mimi
		run speech ./engine 'TestOpenAISpeechToSpeech'
		drop kyutai/moshiko-candle-q8 kyutai/mimi
		;;
	imgbench | imgbench-sd | imgbench-klein)
		# Cost per image: build the CLI with CUDA and time real generations
		# (each twice: the first load is cold). Seconds x the runtime's
		# compute units per hour gives units per image.
		if [ ! -x "$WORK/light-bin" ]; then
			say "build light"
			go build -o "$WORK/light-bin" ./cmd/light || { SUMMARY+=("$g FAIL build"); continue; }
		fi
		bench() {
			local name=$1
			shift
			local log="$LOGS/$STAMP-$name.log" t0=$SECONDS
			say "$name"
			# flags before the positional repo / prompt: Go's flag parsing stops at the first positional
			"$WORK/light-bin" image -device gpu -o "$LOGS/$name.png" "$@" 2>&1 | tee "$log" | tail -4
			local st=${PIPESTATUS[0]}
			local times
			times=$(grep -oE '(loaded [^ ]+|generated) in [0-9.]+ s' "$log" | paste -sd ' ')
			SUMMARY+=("$(printf '%-16s %s  %s  wall %ds' "$name" "$([ "$st" = 0 ] && echo OK || echo FAIL)" "$times" $((SECONDS - t0)))")
		}
		if [ "$g" != imgbench-klein ]; then
			SD=stable-diffusion-v1-5/stable-diffusion-v1-5
			fetch $SD model_index.json 'scheduler/*' 'tokenizer/*' text_encoder/config.json text_encoder/model.safetensors \
				unet/config.json unet/diffusion_pytorch_model.safetensors vae/config.json vae/diffusion_pytorch_model.safetensors
			for i in 1 2; do bench "sd15-512-30s-$i" -steps 30 -seed $i $SD "a lighthouse on a rocky cliff at sunset, oil painting"; done
			drop $SD
		fi
		if [ "$g" != imgbench-sd ]; then
			KL=black-forest-labs/FLUX.2-klein-4B
			fetch $KL
			for i in 1 2; do bench "klein-512-$i" -width 512 -height 512 -seed $i $KL "a red fox sitting in a snowy forest at sunrise, photograph"; done
			for i in 1 2; do bench "klein-1024-$i" -width 1024 -height 1024 -seed $i $KL "a red fox sitting in a snowy forest at sunrise, photograph"; done
			drop $KL
		fi
		nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
		;;
	zperf)
		# Z-Image-Turbo 1024^2 with per-phase timings (LIGHT_TIMINGS): load,
		# encode, each step, VAE decode, the device synchronised at each mark.
		[ -x "$WORK/light-bin" ] || go build -o "$WORK/light-bin" ./cmd/light || { SUMMARY+=("$g FAIL build"); continue; }
		say "zperf kernels"
		go test -count=1 -run 'FlashAttentionBF16|ForwardMixed16|SiLUMulBF16' -bench 'FlashAttentionBF16' -benchtime 10x -v ./ops ./nn 2>&1 | grep -E '^(=== RUN|\s*--- |ok|FAIL|panic|Benchmark)|_test\.go:[0-9]+:'
		ZI=Tongyi-MAI/Z-Image-Turbo
		fetch $ZI
		say "zperf"
		LIGHT_TIMINGS=1 LIGHT_PROFILE_GPU=1 "$WORK/light-bin" image -device gpu -cpuprofile "$LOGS/zperf.cpu" -o "$LOGS/zperf.png" -width 1024 -height 1024 -seed 11 $ZI 			"A cozy reading nook by a large window on a rainy evening, warm lamp light, a ginger cat curled on a knitted blanket, shelves of old books, photorealistic" 2>&1 | tee "$LOGS/zperf.log"
		SUMMARY+=("zperf $(grep -c '^timing' "$LOGS/zperf.log") phases, $(grep -oE 'generated in [0-9.]+ s' "$LOGS/zperf.log")")
		;;
	imagebench)
		# docs/image-bench.md on one card: full generations at 1024² with
		# Light (TestImageBenchCUDA) and diffusers (bench_diffusers.py) from
		# the same files: BF16 repos, weight-only FP8 casts and city96's
		# Q4_K_S GGUFs (~110 GB of downloads, dropped afterwards). Tables
		# land in $LOGS/imagebench/.
		mkdir -p "$LOGS/imagebench" /content/light-models/checkpoints
		export LIGHT_MODELS_DIR=/content/light-models
		for r in Tongyi-MAI/Z-Image-Turbo black-forest-labs/FLUX.2-klein-4B black-forest-labs/FLUX.1-schnell black-forest-labs/FLUX.1-dev; do
			fetch "$r" "model_index.json" "scheduler/*" "tokenizer/*" "tokenizer_2/*" "text_encoder/*" "text_encoder_2/*" "transformer/*" "vae/*" || { SUMMARY+=("$g FAIL fetch $r"); continue 2; }
		done
		LD_LIBRARY_PATH=$ORIG_LD_LIBRARY_PATH python3 scripts/verify/make_flux_fp8_real.py black-forest-labs/FLUX.1-schnell /content/light-models/checkpoints/flux1-schnell-fp8.safetensors 2>&1 | tail -1
		LD_LIBRARY_PATH=$ORIG_LD_LIBRARY_PATH python3 scripts/verify/make_flux_fp8_real.py black-forest-labs/FLUX.1-dev /content/light-models/checkpoints/flux1-dev-fp8.safetensors 2>&1 | tail -1
		LD_LIBRARY_PATH=$ORIG_LD_LIBRARY_PATH python3 scripts/verify/make_flux2_fp8_real.py black-forest-labs/FLUX.2-klein-4B /content/light-models/checkpoints/flux2-klein-4b-fp8 2>&1 | tail -1
		python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download
for repo, f in [("city96/FLUX.1-schnell-gguf", "flux1-schnell-Q4_K_S.gguf"), ("city96/FLUX.1-dev-gguf", "flux1-dev-Q4_K_S.gguf")]:
    hf_hub_download(repo, f, local_dir="/content/light-models/checkpoints", token=os.environ.get("HF_TOKEN") or None)
    print("fetched", f)
PY
		python3 -c 'import gguf' 2>/dev/null || pip install -q gguf  # diffusers' GGUF loader needs it
		nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee "$LOGS/imagebench/gpu.txt"
		LIGHT_IMAGE_BENCH=all run imagebench-light ./diffusion 'TestImageBenchCUDA' -timeout 4h
		grep -E '^\s*image_bench_cuda_test.go:[0-9]+: \|' "$LOGS/$STAMP-imagebench-light.log" | sed 's/.*: //' > "$LOGS/imagebench/light.md"
		# The same with BF16 tensor-core compute (Mixed16), diffusers' precision.
		LIGHT_FLUX_MIXED16=1 LIGHT_IMAGE_BENCH=all run imagebench-light-mixed ./diffusion 'TestImageBenchCUDA' -timeout 4h
		grep -E '^\s*image_bench_cuda_test.go:[0-9]+: \|' "$LOGS/$STAMP-imagebench-light-mixed.log" | sed 's/.*: //' >> "$LOGS/imagebench/light.md"
		LD_LIBRARY_PATH=$ORIG_LD_LIBRARY_PATH python3 scripts/verify/bench_diffusers.py all 2>&1 | tee "$LOGS/$STAMP-imagebench-diffusers.log" | grep -E '^\|' | tee "$LOGS/imagebench/diffusers.md"
		say "image bench tables"
		echo "Light:"; cat "$LOGS/imagebench/light.md"; echo "diffusers:"; cat "$LOGS/imagebench/diffusers.md"
		drop Tongyi-MAI/Z-Image-Turbo black-forest-labs/FLUX.2-klein-4B black-forest-labs/FLUX.1-schnell black-forest-labs/FLUX.1-dev
		rm -rf /content/light-models
		;;
	candlebench)
		# Candle's FLUX example on the same card, for docs/image-bench.md:
		# a Rust toolchain and a CUDA build of candle-examples (~10 min),
		# then schnell at 1024², 4 steps: quantized (lmz/candle-flux's GGUF)
		# everywhere, BF16 (24 GB) and dev BF16 (28 steps) on a 40 GB card.
		# Each run twice; the second (downloads done, kernels warm) counts.
		mkdir -p "$LOGS/candlebench"
		if ! command -v cargo >/dev/null; then
			curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
		fi
		export PATH=$HOME/.cargo/bin:$PATH
		[ -d /content/candle ] || git clone -q --depth 1 https://github.com/huggingface/candle /content/candle
		( cd /content/candle && git log -1 --format='candle %h %cd' --date=short && cargo build --release --features cuda --example flux 2>&1 | tail -2 ) || { SUMMARY+=("$g FAIL build"); continue; }
		vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
		cbench() { # name args...: run twice, report the second's wall time
			local name=$1; shift
			local t0 t1
			for i in 1 2; do
				t0=$SECONDS
				( cd /content/candle && target/release/examples/flux "$@" ) > "$LOGS/$STAMP-candle-$name-$i.log" 2>&1 || { tail -5 "$LOGS/$STAMP-candle-$name-$i.log"; SUMMARY+=("candle $name FAIL"); return; }
				t1=$((SECONDS - t0))
				echo "candle $name run $i: ${t1}s"
			done
			printf '| %s | Candle | 1024² | %s | | %dm%02ds |\n' "$name" "$STEPS" $((t1 / 60)) $((t1 % 60)) | tee -a "$LOGS/candlebench/candle.md"
			cp /content/candle/out.jpg "$LOGS/candlebench/$name.jpg" 2>/dev/null
		}
		P="a watercolour painting of a red lighthouse on a rocky shore at dawn, soft light"
		STEPS=4 cbench schnell-q8 --model schnell --quantized --height 1024 --width 1024 --prompt "$P"
		if [ "${vram:-0}" -gt 36000 ]; then
			STEPS=4 cbench schnell-bf16 --model schnell --height 1024 --width 1024 --prompt "$P"
			STEPS=28 cbench dev-bf16 --model dev --height 1024 --width 1024 --prompt "$P"
		fi
		say "candle table"; cat "$LOGS/candlebench/candle.md"
		;;
	flux2dev)
		# FLUX.2-dev against diffusers (BF16, model CPU offload) at 512², 8
		# steps: Light with the transformer resident at BF16 (64 GB), then
		# as a weight-only FP8 cast (32 GB), both on the same seeded noise;
		# images and the reference land in $LOGS/flux2dev/. The 60 GB
		# single-file checkpoint is skipped.
		mkdir -p "$LOGS/flux2dev"
		fetch black-forest-labs/FLUX.2-dev "model_index.json" "scheduler/*" "tokenizer/*" "text_encoder/*" "transformer/*" "vae/*" || { SUMMARY+=("$g FAIL fetch"); continue; }
		LD_LIBRARY_PATH=$ORIG_LD_LIBRARY_PATH python3 scripts/verify/ref_real_pipe.py flux2dev "$LOGS/flux2dev/flux2dev.json" 512 8 2>&1 | tee "$LOGS/$STAMP-flux2dev-ref.log" | tail -3
		# On a 40 GB card the Mistral encoder runs on the CPU and the BF16
		# transformer streams; FP8 (32 GB) sits resident. (80 GB: all resident.)
		vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
		small=; [ "${vram:-0}" -lt 70000 ] && small=1   # 40 GB: encoder on the CPU, BF16 streamed
		tiny=; [ "${vram:-0}" -lt 36000 ] && tiny=1     # 24 GB: FP8 (32 GB) streamed too
		LIGHT_FLUX2_ENCODER_CPU=$small LIGHT_FLUX2_STREAM=$small LIGHT_KLEIN_REAL_REF="$LOGS/flux2dev/flux2dev.json" LIGHT_REAL_SAVE_DIR="$LOGS/flux2dev" run flux2dev-bf16 ./diffusion 'TestFlux2KleinRealCUDA'
		LD_LIBRARY_PATH=$ORIG_LD_LIBRARY_PATH python3 scripts/verify/make_flux2_fp8_real.py black-forest-labs/FLUX.2-dev /content/flux2-dev-fp8 2>&1 | tail -1
		mkdir -p "$LOGS/flux2dev/fp8"
		LIGHT_FLUX2_ENCODER_CPU=$small LIGHT_FLUX2_STREAM=$tiny LIGHT_KLEIN_REAL_REF="$LOGS/flux2dev/flux2dev.json" LIGHT_REAL_SAVE_DIR="$LOGS/flux2dev/fp8" LIGHT_FLUX2_TRANSFORMER=/content/flux2-dev-fp8 run flux2dev-fp8 ./diffusion 'TestFlux2KleinRealCUDA'
		drop black-forest-labs/FLUX.2-dev
		;;
	trellis)
		# TRELLIS image -> sparse structure -> structured latent -> Gaussians / mesh
		# on real weights: TRELLIS's own code with its real spconv /
		# xformers backends (pip-installed if missing) writes the reference
		# (~3.8 GB of downloads), then Light's CUDA build compares both
		# stages and writes light.ply / light_views.png next to ref.ply.
		# torch brings its own cuDNN: not the one LD_LIBRARY_PATH points cgo at.
		LD_LIBRARY_PATH=$ORIG_LD_LIBRARY_PATH python3 scripts/colab/trellis_check.py /content/trellis_work 2>&1 | tee "$LOGS/$STAMP-trellis-ref.log" | tail -8
		LIGHT_TRELLIS_DIR=/content/trellis_work run trellis ./diffusion 'TrellisReal|TrellisGaussiansReal|TrellisMeshReal'
		cp /content/trellis_work/light_views.png /content/trellis_work/ref.json "$LOGS/" 2>/dev/null
		;;
	idol)
		# speridlabs/idol on three of its demo images (black backgrounds cut
		# out by a flood fill: BiRefNet is not ported yet), 50 steps, 262,144
		# Gaussians, PLY + turntable views in $LOGS/idol/. DINOv3 is gated: it
		# needs HF_TOKEN or the token file (hf auth login).
		[ -x "$WORK/light-bin" ] || go build -o "$WORK/light-bin" ./cmd/light || { SUMMARY+=("$g FAIL build"); continue; }
		mkdir -p "$LOGS/idol"
		python3 - "$LOGS/idol" <<'PY'
import sys, urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
out = sys.argv[1]
for name in ["fig-fox", "food-avocado", "fig-robot-cute"]:
    src = f"{out}/{name}.webp"
    urllib.request.urlretrieve(f"https://raw.githubusercontent.com/speridlabs/idol-sdk/main/demo/examples/{name}.webp", src)
    im = Image.open(src).convert("RGB")
    m = im.copy()
    for xy in [(0, 0), (m.width - 1, 0), (0, m.height - 1), (m.width - 1, m.height - 1)]:
        ImageDraw.floodfill(m, xy, (255, 0, 255), thresh=18)
    mask = (np.array(m) == [255, 0, 255]).all(-1)
    a = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))
    im.putalpha(a)
    im.save(f"{out}/{name}.png")
    print("cut", name, f"{(np.array(a) > 127).mean():.2f} object")
PY
		for ex in fig-fox food-avocado fig-robot-cute; do
			say "idol $ex"
			t0=$SECONDS
			"$WORK/light-bin" idol -device gpu -o "$LOGS/idol/$ex.ply" "$LOGS/idol/$ex.png" 2>&1 | tr '
' '
' | grep -v '^step' | tee "$LOGS/idol-$ex.log"
			SUMMARY+=("idol $ex: $(grep -oE '(loaded idol|generated [0-9]+ Gaussians) in [0-9.]+ s' "$LOGS/idol-$ex.log" | paste -sd ' ')  wall $((SECONDS - t0))s")
		done
		;;
	idolcheck)
		# Idol on the real weights against idol-sdk: scripts/colab/idol_check.py
		# records the SDK's FP16 networks as shipped and in F32 for fig-fox
		# (~10 min; torch keeps the original LD_LIBRARY_PATH), then the
		# models / diffusion TestIdol*RealMatchesSDK compare Light's CUDA
		# path. Run after the idol group (its cut-out fig-fox.png, downloads).
		[ -f "$LOGS/idol/fig-fox.png" ] || { SUMMARY+=("idolcheck SKIP (run the idol group first)"); continue; }
		[ -d /content/idol-sdk ] || git clone -q --depth 1 https://github.com/speridlabs/idol-sdk /content/idol-sdk
		pip install -q tensordict plyfile 2>&1 | tail -1
		LD_LIBRARY_PATH=$ORIG_LD_LIBRARY_PATH python3 scripts/colab/idol_check.py /content/idol-sdk "$LOGS/idol/fig-fox.png" /content/idol_ref 2>&1 |
			grep -v 'it/s\]$' | tee "$LOGS/$STAMP-idol-ref.log" | tail -12
		W=$(ls -d "$HF_HUB"/models--speridlabs--idol/snapshots/*/ | head -1)
		LIGHT_IDOL_REF=/content/idol_ref LIGHT_IDOL_DIR=$W run idolcheck './models ./diffusion' 'TestIdolRealMatchesSDK|TestIdolConditionRealMatchesSDK' -p 1
		;;
	examples)
		# Example images for scratch/examples: 1024^2 generations kept in
		# $LOGS/examples/<model>/ (run with colab run --keep, then colab download).
		# FLUX 3 Image Klein is gated: it runs only when HF_TOKEN is set.
		if [ ! -x "$WORK/light-bin" ]; then
			say "build light"
			go build -o "$WORK/light-bin" ./cmd/light || { SUMMARY+=("$g FAIL build"); continue; }
		fi
		gen() {
			local model=$1 name=$2
			shift 2
			mkdir -p "$LOGS/examples/$model"
			local log="$LOGS/$STAMP-$model-$name.log" t0=$SECONDS
			say "$model/$name"
			"$WORK/light-bin" image -device gpu -o "$LOGS/examples/$model/$name.png" "$@" 2>&1 | tee "$log" | tail -4
			local st=${PIPESTATUS[0]}
			local times
			times=$(grep -oE '(loaded [^ ]+|generated) in [0-9.]+ s' "$log" | paste -sd ' ')
			SUMMARY+=("$(printf '%-28s %s  %s  wall %ds' "$model/$name" "$([ "$st" = 0 ] && echo OK || echo FAIL)" "$times" $((SECONDS - t0)))")
		}
		ZI=Tongyi-MAI/Z-Image-Turbo
		fetch $ZI
		gen z-image-turbo reading-nook -width 1024 -height 1024 -seed 11 $ZI \
			"A cozy reading nook by a large window on a rainy evening, warm lamp light, a ginger cat curled on a knitted blanket, shelves of old books, photorealistic"
		gen z-image-turbo ramen-stall -width 1024 -height 1024 -seed 12 $ZI \
			"A ramen stall in a Tokyo alley at night, steam rising from bowls, glowing paper lanterns and neon signs reflected on wet cobblestones, cinematic photograph"
		drop $ZI
		if [ -n "${HF_TOKEN:-}" ]; then
			F3=black-forest-labs/flux-3-image-klein
			fetch $F3 model_index.json modular_model_index.json 'transformer/*' 'vae/*' 'text_encoder/*'
			gen flux-3-image-klein knitted-iceberg -width 1024 -height 1024 -seed 21 $F3 \
				'A chunky knitted diorama: three white cable-knit icebergs with tiny pale blue houses, a cream banner embroidered "KNITTED ICEBERG VILLAGE", a giant magenta knitted whale tail bursting from a yarn sea'
			gen flux-3-image-klein lighthouse-keeper -width 1024 -height 1024 -seed 22 $F3 \
				"An elderly lighthouse keeper in a yellow raincoat stands on wet rocks at dusk, holding a brass lantern; behind him a white lighthouse beam cuts through sea spray, and storm clouds glow orange near the horizon. Photographic, 35mm, shallow depth of field."
			drop $F3
		else
			SUMMARY+=("flux-3-image-klein          SKIP  (no HF_TOKEN for the gated repo)")
		fi
		;;
	*) echo "unknown group $g" ;;
	esac
done

say "summary"
printf '%s\n' "${SUMMARY[@]}" | tee "$LOGS/$STAMP-summary.txt"
