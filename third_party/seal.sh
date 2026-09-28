#!/usr/bin/env bash
# third_party/seal.sh — build / test one Microsoft SEAL configuration.
#
#   seal.sh build     configure + build + install the selected configuration
#   seal.sh check     test/seal/seal-smoke.cpp against it (+ SEAL's sealtest if built)
#
# Driven by `make seal` / `make seal-check`, which export the selection:
#   WITH_RVV_HEXL    ON | OFF      ON: -DSEAL_USE_INTEL_HEXL=ON against rvv-hexl at HEXL_PREFIX
#   SEAL_DIR         <dir>         build tree in <dir>/build, logs in <dir>/*.log
#   SEAL_PREFIX      <dir>/install install prefix
#   SEAL_TAG         v4.1.2        the SEAL release the IPCEI suite uses
#   SEAL_TESTS       ON | OFF      also build SEAL's own sealtest (needs libgtest-dev)
#   STRICT           1             check: treat a reached rvv-hexl stub as a failure
#   ISA_FLAGS CXX CC JOBS
#
# How this maps onto upstream: SEAL's HEXL switch is a plain CMake option,
# -DSEAL_USE_INTEL_HEXL=ON, which does find_package(HEXL 1.2.4) and links HEXL::hexl.
# rvv-hexl's HEXLConfig.cmake (make rvv-hexl-install) satisfies it; no source changes.
#
# Both configurations (stock and rvv-hexl) are built with IDENTICAL options, so the
# backend is the only difference:
#   * SEAL_BUILD_DEPS=OFF: with ON, SEAL downloads Intel HEXL 1.2.5 itself (x86-only)
#     instead of looking for ours. As a consequence SEAL's optional dependencies must be
#     found or disabled: Microsoft GSL (API sugar only), zlib and Zstandard (compressed
#     serialisation, which the IPCEI suite also turns off) are disabled.
#   * static libseal (SEAL's default and what the IPCEI suite links), examples on.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC_DIR="$HERE/src"

SEAL_TAG="${SEAL_TAG:-v4.1.2}"
SEAL_DIR="${SEAL_DIR:?set by the Makefile (build/seal/<ISA>/<name>)}"
SEAL_PREFIX="${SEAL_PREFIX:-$SEAL_DIR/install}"
WITH_RVV_HEXL="${WITH_RVV_HEXL:-OFF}"
SEAL_TESTS="${SEAL_TESTS:-OFF}"
ISA_FLAGS="${ISA_FLAGS:-}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
CXX="${CXX:-g++}"
CC="${CC:-gcc}"

SEAL_SRC="$SRC_DIR/SEAL-$SEAL_TAG"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }

# run_logged <logfile> <what> <cmd...>: quiet on success, tail of the log on failure.
run_logged() {
  local log="$1" what="$2"; shift 2
  if ! "$@" >"$log" 2>&1; then
    echo "ERROR: $what failed. Last 40 lines of $log:" >&2
    tail -40 "$log" >&2
    return 1
  fi
}

fetch_seal() {
  mkdir -p "$SRC_DIR"
  if [ ! -d "$SEAL_SRC" ]; then
    step "clone SEAL $SEAL_TAG"
    git clone --depth 1 --branch "$SEAL_TAG" https://github.com/microsoft/SEAL.git "$SEAL_SRC"
  fi
}

# the installed config.h says which backend this SEAL was built with
seal_uses_hexl() {
  grep -q '^#define SEAL_USE_INTEL_HEXL' "$SEAL_PREFIX"/include/SEAL-*/seal/util/config.h 2>/dev/null
}

build_seal() {
  local variant="$1"; shift
  local bdir="$SEAL_DIR/build"
  mkdir -p "$SEAL_DIR"
  step "SEAL $SEAL_TAG $variant: configure (ISA '$ISA_FLAGS', CXX=$CXX)"
  run_logged "$SEAL_DIR/configure.log" "configure $variant" \
    cmake -S "$SEAL_SRC" -B "$bdir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$(command -v "$CC")" \
      -DCMAKE_CXX_COMPILER="$(command -v "$CXX")" \
      -DCMAKE_C_FLAGS="$ISA_FLAGS" \
      -DCMAKE_CXX_FLAGS="$ISA_FLAGS" \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DSEAL_BUILD_DEPS=OFF \
      -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF -DSEAL_USE_ZSTD=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DSEAL_BUILD_EXAMPLES=ON \
      -DSEAL_BUILD_TESTS="$SEAL_TESTS" \
      -DSEAL_BUILD_BENCH=OFF \
      -DCMAKE_INSTALL_PREFIX="$SEAL_PREFIX" \
      "$@"
  step "SEAL $variant: build (-j$JOBS)"
  run_logged "$SEAL_DIR/build.log" "build $variant" cmake --build "$bdir" -j "$JOBS"
  run_logged "$SEAL_DIR/install.log" "install $variant" cmake --install "$bdir"
  echo "SEAL $variant -> $SEAL_PREFIX"
  echo "  examples: $bdir/bin/sealexamples"
  [ "$SEAL_TESTS" = ON ] && echo "  tests:    $bdir/bin/sealtest"
  return 0
}

case "${1:-}" in
  build)
    fetch_seal
    if [ "$WITH_RVV_HEXL" = ON ]; then
      HEXL_PREFIX="${HEXL_PREFIX:?HEXL_PREFIX must point at an rvv-hexl install}"
      [ -f "$HEXL_PREFIX/lib/cmake/hexl-1.2.6/HEXLConfig.cmake" ] \
        || { echo "ERROR: no HEXLConfig.cmake under $HEXL_PREFIX (make rvv-hexl-install)" >&2; exit 1; }
      build_seal "+ rvv-hexl" \
        -DSEAL_USE_INTEL_HEXL=ON \
        -DHEXL_DIR="$HEXL_PREFIX/lib/cmake/hexl-1.2.6"
      grep -q "HEXL: using rvv-hexl" "$SEAL_DIR/configure.log" \
        || { echo "ERROR: SEAL did not pick up rvv-hexl (see $SEAL_DIR/configure.log)" >&2; exit 1; }
    else
      build_seal "stock" -DSEAL_USE_INTEL_HEXL=OFF
    fi
    ;;
  check)
    inc=$(ls -d "$SEAL_PREFIX"/include/SEAL-* 2>/dev/null | head -1)
    lib=$(ls "$SEAL_PREFIX"/lib/libseal-*.a 2>/dev/null | head -1)
    [ -n "$inc" ] && [ -n "$lib" ] || { echo "no SEAL install at $SEAL_PREFIX (make seal ...)" >&2; exit 1; }
    hexl_args=()
    if seal_uses_hexl; then
      HEXL_PREFIX="${HEXL_PREFIX:?HEXL_PREFIX must point at the rvv-hexl install this SEAL links}"
      # static rvv-hexl: the smoke binary must not depend on LD_LIBRARY_PATH
      hexl_args=(-I"$HEXL_PREFIX/include" "$HEXL_PREFIX/lib/libhexl.a")
    fi
    smoke="$SEAL_DIR/seal-smoke"
    step "$(basename "$SEAL_DIR"): build test/seal/seal-smoke.cpp"
    "$CXX" -std=c++17 -O2 $ISA_FLAGS -I"$inc" "$ROOT/test/seal/seal-smoke.cpp" -o "$smoke" \
      "$lib" ${hexl_args[@]+"${hexl_args[@]}"} -lpthread
    step "$(basename "$SEAL_DIR"): seal-smoke"
    rc=0
    "$smoke" || rc=$?
    # same rule as `make rvv-hexl-test`: reaching a stub is TODO, not a failure (STRICT=1 fails)
    if [ "$rc" = 2 ] && [ -z "${STRICT:-}" ]; then
      echo "(an rvv-hexl stub was reached: TODO, not a failure; STRICT=1 makes it fail)"
      rc=0
    fi
    if [ -x "$SEAL_DIR/build/bin/sealtest" ]; then
      step "$(basename "$SEAL_DIR"): sealtest"
      "$SEAL_DIR/build/bin/sealtest" --gtest_brief=1 || rc=1
    fi
    exit $rc
    ;;
  *)
    sed -n '2,15p' "$0"; exit 1 ;;
esac
