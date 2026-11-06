#!/usr/bin/env bash
# Run per-package coverage check against the project threshold.
#
# Usage:
#   scripts/coverage.sh               # CPU build
#   scripts/coverage.sh cuda          # CUDA build
#
# Exits non-zero on any package below threshold (default 80).

set -euo pipefail

TAGS="${1:-}"
THRESHOLD="${LIGHT_COV_THRESHOLD:-80.0}"

cd "$(dirname "$0")/.."

go run ./internal/coverage_check/cmd \
  -threshold "${THRESHOLD}" \
  -tags "${TAGS}" \
  -pkgs "./..."
