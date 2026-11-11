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
# checkpoints; not in the default set).
# Everything fits an L4 (24 GB); an A100 is not needed.
# Hugging Face downloads use $HF_TOKEN when set.
set -uo pipefail

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
if [ -d "$WORK/.git" ]; then
	git -C "$WORK" fetch --depth 1 origin "$BRANCH" && git -C "$WORK" reset --hard FETCH_HEAD && git -C "$WORK" clean -fdq
else
	git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$WORK" || exit 1
fi
cd "$WORK" || exit 1
git log -1 --format='%h %s (%cd)'

# --- cgo: CUDA, cuBLAS, cuDNN -------------------------------------------------
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
	imgbench)
		# Cost per image: build the CLI with CUDA and time real generations
		# (each twice: the first load is cold). Seconds × the runtime's
		# compute units per hour gives units per image.
		say "build light"
		go build -o "$WORK/light-bin" ./cmd/light || { SUMMARY+=("imgbench FAIL build"); continue; }
		bench() {
			local name=$1
			shift
			local log="$LOGS/$STAMP-$name.log" t0=$SECONDS
			say "$name"
			"$WORK/light-bin" image -device gpu "$@" -o "$LOGS/$name.png" 2>&1 | tee "$log" | tail -4
			local st=${PIPESTATUS[0]}
			local times
			times=$(grep -oE '(loaded [^ ]+|generated) in [0-9.]+ s' "$log" | paste -sd ' ')
			SUMMARY+=("$(printf '%-16s %s  %s  wall %ds' "$name" "$([ "$st" = 0 ] && echo OK || echo FAIL)" "$times" $((SECONDS - t0)))")
		}
		SD=stable-diffusion-v1-5/stable-diffusion-v1-5
		fetch $SD model_index.json 'scheduler/*' 'tokenizer/*' text_encoder/config.json text_encoder/model.safetensors \
			unet/config.json unet/diffusion_pytorch_model.safetensors vae/config.json vae/diffusion_pytorch_model.safetensors
		for i in 1 2; do bench "sd15-512-30s-$i" -steps 30 -seed $i $SD "a lighthouse on a rocky cliff at sunset, oil painting"; done
		drop $SD
		KL=black-forest-labs/FLUX.2-klein-4B
		fetch $KL
		for i in 1 2; do bench "klein-512-$i" -width 512 -height 512 -seed $i $KL "a red fox sitting in a snowy forest at sunrise, photograph"; done
		for i in 1 2; do bench "klein-1024-$i" -width 1024 -height 1024 -seed $i $KL "a red fox sitting in a snowy forest at sunrise, photograph"; done
		drop $KL
		nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
		;;
	*) echo "unknown group $g" ;;
	esac
done

say "summary"
printf '%s\n' "${SUMMARY[@]}" | tee "$LOGS/$STAMP-summary.txt"
