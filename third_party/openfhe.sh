#!/usr/bin/env bash
# third_party/openfhe.sh — build / test one OpenFHE configuration.
#
#   openfhe.sh build     configure + build + install the selected configuration
#   openfhe.sh check     run OpenFHE's own unit tests (core/pke/binfhe) on it
#
# Driven by `make openfhe` / `make openfhe-check`, which export the selection:
#   NATIVE_SIZE      64 | 32              OpenFHE's native integer width
#   WITH_RVV_HEXL    ON | OFF             ON: stage the openfhe-hexl overlay and link
#                                         it against rvv-hexl at HEXL_PREFIX
#   OPENFHE_DIR      <dir>                build tree in <dir>/build, logs in <dir>/*.log
#   OPENFHE_PREFIX   <dir>/install        install prefix
#   ISA_FLAGS CXX CC JOBS OPENFHE_TAG OPENFHE_HEXL_TAG
#   OPENFHE_BENCHMARKS / OPENFHE_UNITTESTS   ON | OFF
#
# How this maps onto upstream: WITH_RVV_HEXL=ON is exactly what openfhe-configurator
# does for an "openfhe-hexl" build (stage the overlay onto openfhe-development, then
# cmake -DWITH_INTEL_HEXL=ON), plus -DINTEL_HEXL_PREBUILT=ON -DINTEL_HEXL_HINT_DIR=
# so OpenFHE's find_package(HEXL 1.2.6) picks up rvv-hexl instead of downloading
# Intel HEXL from GitHub.
#
# Same conventions as the IPCEI suite (ZKP+FHE Research/cpp_benches/common):
#   * WITH_NATIVEOPT=OFF + ISA flags injected by hand (OpenFHE's switch hardcodes
#     -march=native, which GCC rejects on RISC-V).
#   * Shared libraries only, for every configuration: the benches are byte-identical
#     across configurations and only RPATH / LD_LIBRARY_PATH decides which OpenFHE
#     they load.
#   * rvv-hexl builds work at NATIVE_SIZE=64 and 32. Upstream's HEXL HAL casts every
#     coefficient vector to uint64_t* (a 32-bit build would compile and be WRONG), so
#     after staging, patches/openfhe-hexl-wordsize.py rewrites those casts to OpenFHE's
#     BasicInteger (identical code at 64; rvv-hexl's uint32_t API at 32) and fixes the
#     overlay's riscv64 MultD branch, which does not compile at NATIVE_SIZE=32.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$HERE/src"

OPENFHE_TAG="${OPENFHE_TAG:-v1.5.1}"
OPENFHE_HEXL_TAG="${OPENFHE_HEXL_TAG:-v1.5.1.0}"
OPENFHE_DIR="${OPENFHE_DIR:?set by the Makefile (build/openfhe/<ISA>/<name>)}"
OPENFHE_PREFIX="${OPENFHE_PREFIX:-$OPENFHE_DIR/install}"
NATIVE_SIZE="${NATIVE_SIZE:-64}"
WITH_RVV_HEXL="${WITH_RVV_HEXL:-OFF}"
ISA_FLAGS="${ISA_FLAGS:-}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
CXX="${CXX:-g++}"
CC="${CC:-gcc}"
OPENFHE_BENCHMARKS="${OPENFHE_BENCHMARKS:-ON}"
OPENFHE_UNITTESTS="${OPENFHE_UNITTESTS:-$WITH_RVV_HEXL}"

STOCK_SRC="$SRC_DIR/openfhe-development-$OPENFHE_TAG"
OVERLAY_SRC="$SRC_DIR/openfhe-hexl-$OPENFHE_HEXL_TAG"
RVVHEXL_SRC="$SRC_DIR/openfhe-rvvhexl-$OPENFHE_TAG"

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

fetch_stock() {
  mkdir -p "$SRC_DIR"
  if [ ! -d "$STOCK_SRC" ]; then
    step "clone openfhe-development $OPENFHE_TAG"
    git clone --depth 1 --branch "$OPENFHE_TAG" \
      https://github.com/openfheorg/openfhe-development.git "$STOCK_SRC"
  fi
  patch_multd "$STOCK_SRC/src/core/include/math/hal/intnat/ubintnat.h"
}

# Same downstream patch as ZKP+FHE Research/cpp_benches/common/build_openfhe.sh:
# MultD()'s preprocessor chain covers riscv64 only via HAVE_INT128, which CMake turns
# off at NATIVE_SIZE=32, so the stock32 build hits `#error Architecture not supported`.
# Adds an explicit riscv64 arm. Idempotent; inert on other architectures.
patch_multd() {
  python3 - "$1" <<'PATCH_EOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
s = p.read_text()
if "defined(__riscv) && __riscv_xlen == 64" in s or "#elif __riscv" in s:
    print("[openfhe] MultD riscv64 arm present")
    sys.exit(0)
anchor = "#elif defined(__arm__) || defined(__powerpc__)  // 32 bit processor"
if s.count(anchor) != 1:
    print("[openfhe] WARNING: MultD anchor not found; NATIVE_SIZE=32 may not build on riscv64",
          file=sys.stderr)
    sys.exit(0)
arm = """#elif defined(__riscv) && __riscv_xlen == 64
            // Added downstream (rvv-hexl): riscv64 otherwise relies on HAVE_INT128,
            // which CMake disables at NATIVE_SIZE=32.
            __uint128_t c{static_cast<__uint128_t>(a) * b};
            res.hi = static_cast<uint64_t>(c >> 64);
            res.lo = static_cast<uint64_t>(c);
"""
p.write_text(s.replace(anchor, arm + anchor))
print("[openfhe] patched MultD(): added riscv64 __int128 arm")
PATCH_EOF
}

# The openfhe-hexl overlay is "staged" onto a pristine checkout, exactly like
# openfhe-hexl/scripts/stage-openfhe-development-hexl.sh does.
stage_rvvhexl() {
  fetch_stock
  if [ ! -d "$OVERLAY_SRC" ]; then
    step "clone openfhe-hexl $OPENFHE_HEXL_TAG"
    git clone --depth 1 --branch "$OPENFHE_HEXL_TAG" \
      https://github.com/openfheorg/openfhe-hexl.git "$OVERLAY_SRC"
  fi
  if [ ! -f "$RVVHEXL_SRC/.staged" ]; then
    step "stage openfhe-hexl overlay -> $RVVHEXL_SRC"
    rm -rf "$RVVHEXL_SRC"
    cp -R "$STOCK_SRC" "$RVVHEXL_SRC"   # keeps .git: OpenFHE's CMake fetches submodules
    for f in CMakeLists.txt CMakeLists.User.txt OpenFHEConfig.cmake.in; do
      cp "$OVERLAY_SRC/$f" "$RVVHEXL_SRC/"
    done
    for d in benchmark configure src third-party; do
      cp -R "$OVERLAY_SRC/$d" "$RVVHEXL_SRC/"
    done
    grep -q WITH_INTEL_HEXL "$RVVHEXL_SRC/CMakeLists.txt" \
      || { echo "ERROR: staged CMakeLists.txt has no WITH_INTEL_HEXL" >&2; exit 1; }
    touch "$RVVHEXL_SRC/.staged"
  fi
  # idempotent: safe on an already staged (and patched) tree
  python3 "$HERE/patches/openfhe-hexl-wordsize.py" "$RVVHEXL_SRC"
}

# build_openfhe <src> <label> [extra cmake args...]
build_openfhe() {
  local src="$1" variant="$2"; shift 2
  local bdir="$OPENFHE_DIR/build" prefix="$OPENFHE_PREFIX" nsize="$NATIVE_SIZE"
  local onoff_bench="$OPENFHE_BENCHMARKS" onoff_ut="$OPENFHE_UNITTESTS"
  mkdir -p "$OPENFHE_DIR"

  # macOS/clang (laptop development only): Homebrew libomp must be wired by hand.
  local omp_args=()
  if [ "$(uname -s)" = Darwin ] && command -v brew >/dev/null 2>&1; then
    local lomp; lomp="$(brew --prefix)/opt/libomp"
    omp_args=(-DOpenMP_CXX_FLAGS="-Xclang -fopenmp -I$lomp/include"
              -DOpenMP_C_FLAGS="-Xclang -fopenmp -I$lomp/include"
              -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_C_LIB_NAMES=omp
              -DOpenMP_omp_LIBRARY="$lomp/lib/libomp.dylib")
  fi

  step "OpenFHE $variant: configure (NATIVE_SIZE=$nsize, ISA '$ISA_FLAGS', CXX=$CXX)"
  run_logged "$OPENFHE_DIR/configure.log" "configure $variant" \
    cmake -S "$src" -B "$bdir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$(command -v "$CC")" \
      -DCMAKE_CXX_COMPILER="$(command -v "$CXX")" \
      -DCMAKE_C_FLAGS="$ISA_FLAGS" \
      -DCMAKE_CXX_FLAGS="$ISA_FLAGS" \
      -DWITH_NATIVEOPT=OFF \
      -DWITH_OPENMP=ON \
      -DNATIVE_SIZE="$nsize" \
      -DBUILD_SHARED=ON -DBUILD_STATIC=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_BENCHMARKS="$onoff_bench" \
      -DBUILD_UNITTESTS="$onoff_ut" \
      -DCMAKE_INSTALL_PREFIX="$prefix" \
      ${omp_args[@]+"${omp_args[@]}"} \
      "$@"
  step "OpenFHE $variant: build (-j$JOBS; long)"
  run_logged "$OPENFHE_DIR/build.log" "build $variant" \
    cmake --build "$bdir" -j "$JOBS"
  run_logged "$OPENFHE_DIR/install.log" "install $variant" \
    cmake --install "$bdir"
  echo "OpenFHE $variant -> $prefix"
  [ "$onoff_bench" = ON ] && echo "  upstream benchmarks: $bdir/bin/benchmark/"
  [ "$onoff_ut" = ON ] && echo "  unit tests:          $bdir/unittest/"
  return 0
}

case "${1:-}" in
  build)
    if [ "$WITH_RVV_HEXL" = ON ]; then
      case "$NATIVE_SIZE" in 64|32) ;; *) echo "ERROR: WITH_RVV_HEXL=ON supports NATIVE_SIZE=64 or 32" >&2; exit 1 ;; esac
      HEXL_PREFIX="${HEXL_PREFIX:?HEXL_PREFIX must point at an rvv-hexl install}"
      [ -f "$HEXL_PREFIX/lib/cmake/hexl-1.2.6/HEXLConfig.cmake" ] \
        || { echo "ERROR: no HEXLConfig.cmake under $HEXL_PREFIX (make rvv-hexl-install)" >&2; exit 1; }
      stage_rvvhexl
      build_openfhe "$RVVHEXL_SRC" "NATIVE_SIZE=$NATIVE_SIZE + rvv-hexl" \
        -DWITH_INTEL_HEXL=ON \
        -DINTEL_HEXL_PREBUILT=ON \
        -DINTEL_HEXL_HINT_DIR="$HEXL_PREFIX"
    else
      fetch_stock
      build_openfhe "$STOCK_SRC" "NATIVE_SIZE=$NATIVE_SIZE stock"
    fi
    ;;
  check)
    ut="$OPENFHE_DIR/build/unittest"
    [ -d "$ut" ] || { echo "no unit tests in $ut (build with OPENFHE_UNITTESTS=ON)" >&2; exit 1; }
    rc=0
    for t in core_tests pke_tests binfhe_tests; do
      step "$(basename "$OPENFHE_DIR"): $t"
      "$ut/$t" --gtest_brief=1 || rc=1
    done
    exit $rc
    ;;
  *)
    sed -n '2,17p' "$0"; exit 1 ;;
esac
