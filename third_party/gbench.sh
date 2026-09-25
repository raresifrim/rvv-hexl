#!/usr/bin/env bash
# third_party/gbench.sh <CXX> <install-prefix>
#
# Builds Google Benchmark from source for bench/hexl when no system package is
# available. Prefer the distro package where it exists:
#     Ubuntu/Debian:  sudo apt install libbenchmark-dev
#     macOS:          brew install google-benchmark
# The Makefile picks up third_party/install/gbench automatically if present.
set -euo pipefail

CXX_BIN="${1:-g++}"
PREFIX="${2:?install prefix required}"
TAG="${GBENCH_TAG:-v1.9.1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src/benchmark"

mkdir -p "$HERE/src"
if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/google/benchmark.git "$SRC"
fi
cmake -S "$SRC" -B "$SRC/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$(command -v "$CXX_BIN")" \
  -DBENCHMARK_ENABLE_TESTING=OFF \
  -DBENCHMARK_ENABLE_GTEST_TESTS=OFF \
  -DBENCHMARK_ENABLE_WERROR=OFF \
  -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build "$SRC/build" -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
cmake --install "$SRC/build"
echo "Google Benchmark $TAG -> $PREFIX"
