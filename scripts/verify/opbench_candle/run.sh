#!/usr/bin/env bash
# Candle timings of single ops at FLUX.1's 1024² geometry, next to Light's
# TestOpBenchCUDA and opbench_torch.py:  CANDLE_DIR=~/src/candle bash run.sh
set -euo pipefail
cd "$(dirname "$0")"
: "${CANDLE_DIR:?set CANDLE_DIR to a candle checkout}"
sed "s#@CANDLE_DIR@#$CANDLE_DIR#g" Cargo.toml.in > Cargo.toml
cargo run --release 2>&1 | grep -E '^\||error|panicked' 
