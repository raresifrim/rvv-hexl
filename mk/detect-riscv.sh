#!/usr/bin/env bash
# mk/detect-riscv.sh — decide, once per toolchain/board, whether rvv-hexl can be built
# with RVV and which -march strings to use. Called by mk/config.mk, which caches the
# result in build/config/riscv-<key>.mk (`make reconfigure` forgets it).
#
# Output (stdout, make syntax):
#   RVV_DETECT_ISA          rvv | scalar      what will actually be built
#   RVV_DETECT_MARCH        -march for ISA=rvv
#   RVV_DETECT_SCALAR_MARCH -march for ISA=scalar (= the rvv march minus V)
#   RVV_DETECT_HOST         extensions common to all harts (native builds)
#   RVV_DETECT_NOTE         why a fallback happened (empty if none)
#
# Inputs (environment):
#   CXX            compiler to test
#   MODE           native (building on the board: check /proc/cpuinfo) | cross
#   ISA_REQUEST    auto (default) | rvv | scalar
#   USER_MARCH / USER_SCALAR_MARCH   explicit RISCV_MARCH / RISCV_SCALAR_MARCH, if given
#   CPUINFO        default /proc/cpuinfo (override to simulate another board)
#
# Checks, in order:
#   1. board (native only): every hart must report V. The -march is then limited to
#      the extensions ALL harts report (big.LITTLE boards such as the K3 can differ,
#      e.g. only X100 has H), because GCC may emit any -march extension anywhere,
#      including in the scalar code. A too-wide -march is a SIGILL, not a slow path.
#   2. toolchain: the compiler accepts the -march and compiles a probe that uses the
#      RVV C intrinsics v1.0 API the port relies on (__riscv_v_intrinsic >= 1000000,
#      policy suffixes, tuple segment loads, narrowing/widening, vmulhu, ...).
#   Any failure with ISA_REQUEST != scalar falls back to scalar, with a note.
set -uo pipefail

CXX="${CXX:-g++}"
MODE="${MODE:-cross}"
ISA_REQUEST="${ISA_REQUEST:-auto}"
USER_MARCH="${USER_MARCH:-}"
USER_SCALAR_MARCH="${USER_SCALAR_MARCH:-}"
CPUINFO="${CPUINFO:-/proc/cpuinfo}"

# RVA23U64 extensions that change code generation (the rest are profile guarantees
# Linux does not list in cpuinfo). A board reporting all of them gets -march=rva23u64.
RVA23_CODEGEN="v zba zbb zbs zicond zfa zfhmin zcb zcmop zimop zawrs zvbb zvfhmin zicbom zicboz zihintntl zihintpause zkt zvkt"
# rva23u64 minus V: the matching scalar baseline
RVA23_SCALAR="rv64imafdc_zicsr_zifencei_zba_zbb_zbs_zicond_zfa_zfhmin_zcb_zcmop_zimop_zawrs_zicbom_zicboz_zihintntl_zihintpause_zkt"

say() { echo "[rvv-detect] $*" >&2; }

accepts() {  # accepts <march>
  echo 'int x;' | "$CXX" -march="$1" -x c++ -c -o /dev/null - >/dev/null 2>&1
}

intrinsics_ok() {  # intrinsics_ok <march>
  "$CXX" -march="$1" -O1 -x c++ -c -o /dev/null - >/dev/null 2>&1 <<'EOF'
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#if !defined(__riscv_v_intrinsic) || __riscv_v_intrinsic < 1000000
#error "RVV C intrinsics v1.0 required"
#endif
void rvv_probe(uint64_t* p, uint32_t* q, size_t n) {
  size_t vl = __riscv_vsetvl_e64m1(n);
  vuint64m1_t a = __riscv_vle64_v_u64m1(p, vl);
  vuint64m1_t b = __riscv_vmulhu_vv_u64m1(a, a, vl);
  b = __riscv_vmul_vx_u64m1(b, 3, vl);
  b = __riscv_vminu_vv_u64m1(b, __riscv_vsub_vx_u64m1(b, 7, vl), vl);
  vbool64_t m = __riscv_vmsgeu_vx_u64m1_b64(b, 5, vl);
  b = __riscv_vadd_vx_u64m1_mu(m, b, b, 1, vl);
  b = __riscv_vmerge_vvm_u64m1(a, b, m, vl);
  __riscv_vse64_v_u64m1(p, b, vl);
  size_t vl32 = __riscv_vsetvl_e32m1(n);
  vuint32m1_t x = __riscv_vncvt_x_x_w_u32m1(__riscv_vle64_v_u64m2(p, vl32), vl32);
  x = __riscv_vmulhu_vx_u32m1(x, 9u, vl32);
  vuint32m1x2_t t = __riscv_vlseg2e32_v_u32m1x2(q, vl32);
  x = __riscv_vadd_vv_u32m1(x, __riscv_vget_v_u32m1x2_u32m1(t, 0), vl32);
  x = __riscv_vadd_vv_u32m1(x, __riscv_vlse32_v_u32m1(q, 8, vl32), vl32);
  __riscv_vse64_v_u64m2(p, __riscv_vzext_vf2_u64m2(x, vl32), vl32);
  (void)__riscv_vsetvlmax_e8m1();
}
EOF
}

# strip V: single letter 'v' and every zv*/zve*/zvl* extension
strip_v() {
  local base="${1%%_*}" rest=""
  [ "$base" != "$1" ] && rest="${1#*_}"
  base="${base//v/}"
  local out="$base" e
  for e in ${rest//_/ }; do case "$e" in zv*) ;; *) out="${out}_$e" ;; esac; done
  echo "$out"
}

# ---- 1. what the board has (intersection over all harts) ------------------------
HOST_LETTERS="" HOST_EXTS="" HAVE_HOST=0
if [ "$MODE" = native ] && [ -r "$CPUINFO" ] && grep -q '^isa[[:space:]]*:' "$CPUINFO"; then
  HAVE_HOST=1
  read -r HOST_LETTERS HOST_EXTS < <(grep '^isa[[:space:]]*:' "$CPUINFO" | sed 's/^[^:]*:[[:space:]]*//' | awk '
    { n++; s = $0; sub(/^rv(32|64)/, "", s)
      split(s, parts, "_"); letters = parts[1]
      for (i = 1; i <= length(letters); i++) L[substr(letters, i, 1)]++
      for (i = 2; i in parts; i++) if (parts[i] != "") E[parts[i]]++ }
    END {
      ls = ""; split("imafdcv", order, "")
      for (i = 1; i <= 7; i++) if (L[order[i]] == n) ls = ls order[i]
      es = ""
      for (e in E) if (E[e] == n && e ~ /^z/ && e !~ /^zve/ && e !~ /^zvl/) es = es " " e
      print ls, es }')
  HOST_EXTS="$(echo $HOST_EXTS | tr ' ' '\n' | sort | tr '\n' ' ' | sed 's/ $//')"
fi

host_has() {  # host_has <ext>  (single letter or z-extension)
  case "$1" in
    ?) case "$HOST_LETTERS" in *"$1"*) return 0 ;; esac; return 1 ;;
    *) case " $HOST_EXTS " in *" $1 "*) return 0 ;; esac; return 1 ;;
  esac
}

# -march built from exactly what the board reports (V optional), minus what the
# compiler does not know. zicsr/zifencei are always present on Linux-capable cores.
host_march() {  # host_march with_v|no_v
  local letters="$HOST_LETTERS" base exts="" e
  [ "$1" = no_v ] && letters="${letters//v/}"
  base="rv64${letters}_zicsr_zifencei"
  for e in $HOST_EXTS; do
    case "$e" in zicsr|zifencei) continue ;; esac
    [ "$1" = no_v ] && case "$e" in zv*) continue ;; esac
    exts="${exts}_$e"
  done
  if accepts "$base$exts"; then echo "$base$exts"; return; fi
  local kept=""  # slow path: keep only what this compiler understands
  for e in ${exts//_/ }; do accepts "${base}_$e" && kept="${kept}_$e"; done
  echo "$base$kept"
}

NOTE=""
fallback() { NOTE="$1"; say "falling back to ISA=scalar: $1"; }

# ---- 2. vector march ---------------------------------------------------------------
MARCH=""
if [ "$ISA_REQUEST" != scalar ]; then
  if [ "$HAVE_HOST" = 1 ] && ! host_has v; then
    fallback "this board has no V extension (cpuinfo isa: rv64$HOST_LETTERS)"
  elif [ -n "$USER_MARCH" ]; then
    MARCH="$USER_MARCH"
  elif [ "$HAVE_HOST" = 1 ]; then
    missing=""
    for e in $RVA23_CODEGEN; do host_has "$e" || missing="$missing $e"; done
    if [ -z "$missing" ] && accepts rva23u64; then
      MARCH=rva23u64
    else
      MARCH="$(host_march with_v)"
      [ -n "$missing" ] && say "board is not RVA23 (missing:$missing); using its own extensions"
    fi
  else
    for m in rva23u64 rv64gcv; do accepts "$m" && MARCH="$m" && break; done
  fi

  if [ -z "$NOTE" ]; then
    if [ -z "$MARCH" ] || ! accepts "$MARCH"; then
      fallback "$CXX accepts no vector -march${MARCH:+ (tried $MARCH)}"
      MARCH=""
    elif ! intrinsics_ok "$MARCH"; then
      fallback "$CXX with -march=$MARCH cannot compile the RVV intrinsics v1.0 API (GCC >= 14 or LLVM >= 17 needed)"
    fi
  fi
fi

# ---- 3. scalar march = vector march minus V --------------------------------------
if [ -n "$USER_SCALAR_MARCH" ]; then
  SMARCH="$USER_SCALAR_MARCH"
elif [ "$MARCH" = rva23u64 ] || { [ -z "$MARCH" ] && [ "$HAVE_HOST" = 0 ] && accepts "$RVA23_SCALAR"; }; then
  SMARCH="$RVA23_SCALAR"
elif [ "$HAVE_HOST" = 1 ]; then
  missing=""
  for e in $RVA23_CODEGEN; do [ "$e" = v ] || case "$e" in zv*) ;; *) host_has "$e" || missing="$missing $e" ;; esac; done
  if [ -z "$missing" ] && accepts "$RVA23_SCALAR"; then SMARCH="$RVA23_SCALAR"; else SMARCH="$(host_march no_v)"; fi
elif [ -n "$MARCH" ]; then
  SMARCH="$(strip_v "$MARCH")"
else
  SMARCH=rv64gc
fi
accepts "$SMARCH" || SMARCH=rv64gc

ISA=rvv
{ [ "$ISA_REQUEST" = scalar ] || [ -n "$NOTE" ] || [ -z "$MARCH" ]; } && ISA=scalar
if [ "$ISA_REQUEST" != scalar ] && [ "$ISA" = rvv ]; then
  say "RVV enabled: -march=$MARCH  (scalar baseline: -march=$SMARCH)"
fi

echo "RVV_DETECT_ISA := $ISA"
echo "RVV_DETECT_MARCH := $MARCH"
echo "RVV_DETECT_SCALAR_MARCH := $SMARCH"
echo "RVV_DETECT_HOST := $( [ "$HAVE_HOST" = 1 ] && echo "rv64$HOST_LETTERS $HOST_EXTS" )"
echo "RVV_DETECT_NOTE := $NOTE"
