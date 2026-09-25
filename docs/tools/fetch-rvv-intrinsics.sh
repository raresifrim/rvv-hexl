#!/usr/bin/env bash
# docs/tools/fetch-rvv-intrinsics.sh — assemble the RVA23 RVV C-intrinsics reference in
# docs/rvv_intrinsics/ (git-ignored) and build a searchable index of it.
#
# Source: github.com/riscv-non-isa/rvv-intrinsic-doc
#   * tag v1.0-ratified                 base V intrinsics (incl. Zvfh / Zvfhmin float16)
#   * main @ $RVV_INTRINSICS_EXTRA_REF   the ratified extensions the tag does not cover yet:
#       auto-generated/vector-crypto/    Zvbb (⊃ Zvkb), Zvbc, Zvkg, Zvkned, Zvknh[ab], Zvksed, Zvksh
#       auto-generated/bfloat16/         Zvfbfmin, Zvfbfwma (chapters 00-05 only)
#     Draft extensions on main that are NOT part of RVA23 are left out on purpose:
#     Zvfbfa, Zvzip, Zvabd, Zvdot*, Zvfofp8min, Zvq*dot*, Zvfbdot*, ... .
#
# Then runs gen-rvv-intrinsics-index.py, which writes
#   docs/rvv_intrinsics/INDEX.md                 per-extension summary (RVA23 status, K3, GCC)
#   docs/rvv_intrinsics/rva23-intrinsics.tsv     one line per intrinsic, grep-able
#
# Usage: docs/tools/fetch-rvv-intrinsics.sh            (normally via `make rvv-intrinsics-doc`)
# Env:   PROBE_CC=<riscv gcc>   compiler used to mark which intrinsics it implements
#                               (default: native gcc on riscv64, else riscv64-*-gcc in PATH)
#        FORCE=1                re-extract the extra extensions and rebuild the index
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$(cd "$HERE/.." && pwd)/rvv_intrinsics"
URL=https://github.com/riscv-non-isa/rvv-intrinsic-doc.git
TAG="${RVV_INTRINSICS_TAG:-v1.0-ratified}"
# Pinned for reproducibility: main as of 2026-08-24.
EXTRA_REF="${RVV_INTRINSICS_EXTRA_REF:-f204a64633237425f0b2b9addf65c71e164e570f}"

# ---- 1. base: the ratified v1.0 intrinsics -------------------------------------
if [ ! -d "$DEST/.git" ]; then
  git -c advice.detachedHead=false clone --depth 1 --branch "$TAG" "$URL" "$DEST"
fi

# ---- 2. extra ratified extensions from main (extracted, index untouched) ---------
STAMP="$DEST/.rva23-extras"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$EXTRA_REF" ] || [ "${FORCE:-0}" = 1 ]; then
  echo "[rvv-intrinsics] adding vector-crypto + bfloat16 from main @ ${EXTRA_REF:0:9}"
  git -C "$DEST" fetch -q --depth 1 origin "$EXTRA_REF"
  rm -rf "$DEST/auto-generated/vector-crypto" "$DEST/auto-generated/bfloat16"
  git -C "$DEST" archive FETCH_HEAD \
      auto-generated/vector-crypto \
      auto-generated/bfloat16/intrinsic_funcs \
      auto-generated/bfloat16/overloaded_intrinsic_funcs \
      auto-generated/bfloat16/policy_funcs/intrinsic_funcs \
      auto-generated/bfloat16/policy_funcs/overloaded_intrinsic_funcs \
      doc/vector-bfloat16-spec.adoc \
      vector_crypto_notes.adoc \
    | tar -x -C "$DEST"
  # drop the draft (non-RVA23) bf16 chapters: 06 = Zvfbfa, 07 = Zvzip
  find "$DEST/auto-generated/bfloat16" -name '0[6-9]_*.adoc' -delete
  echo "$EXTRA_REF" > "$STAMP"
fi

# ---- 3. which compiler to ask ----------------------------------------------------
if [ -z "${PROBE_CC:-}" ]; then
  if [ "$(uname -m)" = riscv64 ]; then
    for c in gcc-16 gcc-15 gcc-14 gcc; do command -v "$c" >/dev/null && PROBE_CC="$c" && break; done
  else
    for c in riscv64-unknown-linux-gnu-gcc riscv64-linux-gnu-gcc riscv64-unknown-elf-gcc \
             /opt/homebrew/opt/riscv-gnu-toolchain/bin/riscv64-unknown-elf-gcc; do
      command -v "$c" >/dev/null && PROBE_CC="$c" && break
    done
  fi
fi

python3 "$HERE/gen-rvv-intrinsics-index.py" "$DEST" ${PROBE_CC:+--cc "$PROBE_CC"}
