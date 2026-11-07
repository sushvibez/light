#!/usr/bin/env bash
# Compile every .cu file under kernels/cuda/src/ to a PTX file in kernels/cuda/ptx/.
#
# PTX is embedded into the Go binary via //go:embed and JIT'd to local SM by
# the CUDA driver at first use (cached under ~/.nv/ComputeCache).
#
# Usage:
#   scripts/build_ptx.sh           # builds compute_80 baseline
#
# Override baseline via LIGHT_PTX_ARCH env var (e.g. compute_90 for Hopper-only).
#
# The committed PTX is built with scripts/nvrtc_ptx.py --all (NVRTC 13.0,
# PTX ISA 9.0, loads on CUDA 13.0+ drivers). This script uses whatever nvcc
# is installed, so a newer toolkit raises the ISA and with it the minimum
# driver; use it for local rebuilds (e.g. the Colab notebook does when the
# driver is older than the committed PTX needs).

set -euo pipefail

CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
NVCC="${CUDA_PATH}/bin/nvcc"
ARCH="${LIGHT_PTX_ARCH:-compute_80}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/kernels/cuda/src"
OUT_DIR="${REPO_ROOT}/kernels/cuda/ptx"

mkdir -p "${OUT_DIR}"

FLAGS="-I${SRC_DIR} -I${CUDA_PATH}/include --std=c++17 -use_fast_math --ptx -arch=${ARCH}"

built=0
for cu in "${SRC_DIR}"/*.cu; do
  [ -f "${cu}" ] || continue
  name="$(basename "${cu}" .cu)"
  out="${OUT_DIR}/${name}.ptx"
  echo "nvcc -> ${out}"
  "${NVCC}" ${FLAGS} "${cu}" -o "${out}"
  built=$((built + 1))
done

echo "built ${built} PTX file(s) at arch=${ARCH}"
