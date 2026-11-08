#!/usr/bin/env bash
# Fetch the pinned wgpu-native release (the WebGPU backend's run-time
# library; Light loads it without cgo) into ~/.cache/light/wgpu/, where
# internal/wgpudev looks for it ($LIGHT_WGPU_LIB or a copy next to the
# executable also work). Same pinned version as github.com/sushvibez/gfx.
set -euo pipefail

VERSION=v29.0.0.0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${LIGHT_WGPU_DIR:-$HOME/.cache/light/wgpu}"

# The host by default. WGPU_OS / WGPU_ARCH fetch for somewhere else —
# packaging a Windows build on Linux needs the Windows DLL, and there is
# nothing host-specific about downloading it. A cross fetch lands in its
# own directory so it cannot overwrite the one this machine runs on.
case "${WGPU_OS:-$(uname -s)}" in
  Linux|linux)  os=linux ;;
  Darwin|macos) os=macos ;;
  MINGW*|MSYS*|CYGWIN*|windows) os=windows ;;
  *) echo "unsupported OS: ${WGPU_OS:-$(uname -s)}" >&2; exit 1 ;;
esac

case "${WGPU_ARCH:-$(uname -m)}" in
  x86_64|amd64) arch=x86_64 ;;
  aarch64|arm64) arch=aarch64 ;;
  *) echo "unsupported arch: ${WGPU_ARCH:-$(uname -m)}" >&2; exit 1 ;;
esac

if [ -n "${WGPU_OS:-}" ] || [ -n "${WGPU_ARCH:-}" ]; then
  DEST="$DEST-$os-$arch"
fi

# v29 renamed the Windows assets to carry an ABI infix; use the MSVC build.
# Linux/macOS keep the plain wgpu-$os-$arch-release.zip spelling.
if [ "$os" = "windows" ]; then
  asset="wgpu-$os-$arch-msvc-release.zip"
else
  asset="wgpu-$os-$arch-release.zip"
fi
url="https://github.com/gfx-rs/wgpu-native/releases/download/$VERSION/$asset"

rm -rf "$DEST"
mkdir -p "$DEST"
echo "fetching $url"
curl -sfL -o "$DEST/$asset" "$url"
unzip -o -q "$DEST/$asset" -d "$DEST"
rm "$DEST/$asset"
# v22+ release zips nest the library under lib/; flatten so the loader finds
# build/wgpu/libwgpu_native.{so,dll,a} at the top level (as LibPaths expects).
if [ -d "$DEST/lib" ]; then
  cp -f "$DEST/lib/"* "$DEST/" 2>/dev/null || true
fi
echo "$VERSION" > "$DEST/commit-sha"
echo "wgpu-native $VERSION ready in $DEST"
