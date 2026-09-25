#!/usr/bin/env python3
"""Make the staged openfhe-hexl overlay (v1.5.1.0) independent of NATIVE_SIZE.

    openfhe-hexl-wordsize.py <staged openfhe-development tree>

Applied by third_party/openfhe.sh right after staging the overlay, for every
rvv-hexl build. Idempotent; fails loudly if upstream changes shape.

1. Word size. The overlay's HEXL HAL reinterprets every coefficient vector as
   uint64_t* before calling HEXL (60 casts in 7 files). With NATIVE_SIZE=32
   OpenFHE stores uint32_t, so those casts read 8N bytes of a 4N-byte buffer:
   it compiles and computes garbage. The patch casts to BasicInteger* instead
   (OpenFHE's own word typedef, math/hal/basicint.h: uint64_t at 64, uint32_t
   at 32). At NATIVE_SIZE=64 the patched code is identical to upstream; at 32
   overload resolution picks rvv-hexl's uint32_t API (HEXL_RVV_HAS_32BIT_API).

2. RISC-V MultD. ubintnathexl.h's `#elif __riscv` branch uses U128BITS, which
   only exists when CMake sets HAVE_INT128, and CMake clears it at
   NATIVE_SIZE=32, so the overlay does not even compile there on riscv64. Same
   bug as stock OpenFHE's ubintnat.h (patched by openfhe.sh for the stock
   builds); fixed the same way, with the compiler's unsigned __int128.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1])
MARK = "rvv-hexl: word-size-generic"

CAST_FILES = [
    "src/core/include/math/hal/intnat-hexl/mubintvecnathexl.h",
    "src/core/lib/math/hal/intnat-hexl/mubintvecnathexl.cpp",
    "src/core/include/math/hal/intnat-hexl/transformnathexl-impl.h",
    "src/core/include/lattice/hal/hexl/hexlpoly.h",
    "src/core/include/lattice/hal/hexl/hexlpoly-impl.h",
    "src/core/include/lattice/hal/hexl/hexldcrtpoly.h",
    "src/core/include/lattice/hal/hexl/hexldcrtpoly-impl.h",
]
EXPECTED_CASTS = 60

CAST = re.compile(r"reinterpret_cast<(const )?uint64_t\*>")
# the declared type on the same line: "uint64_t* op1 =" / "const uint64_t* input ="
DECL = re.compile(r"^(\s*)(const )?uint64_t\*(\s+\w+\s*=\s*reinterpret_cast<)")


def patch_casts():
    found = 0
    for rel in CAST_FILES:
        p = ROOT / rel
        s = p.read_text()
        found += len(CAST.findall(s))
    if found == 0 and all(MARK in (ROOT / rel).read_text() for rel in CAST_FILES):
        print("[openfhe-hexl] word-size patch already applied")
        return
    if found != EXPECTED_CASTS:
        sys.exit(f"[openfhe-hexl] ERROR: expected {EXPECTED_CASTS} uint64_t* casts, found "
                 f"{found}: upstream changed, review {__file__}")
    for rel in CAST_FILES:
        p = ROOT / rel
        out = []
        for line in p.read_text().splitlines(keepends=True):
            if CAST.search(line):
                line = DECL.sub(lambda m: f"{m.group(1)}{m.group(2) or ''}BasicInteger*{m.group(3)}", line)
                line = CAST.sub(lambda m: f"reinterpret_cast<{m.group(1) or ''}BasicInteger*>", line)
            out.append(line)
        text = "".join(out)
        text = f"// {MARK} (patched by rvv-hexl/third_party/patches/openfhe-hexl-wordsize.py)\n" + text
        p.write_text(text)
    left = sum(len(CAST.findall((ROOT / rel).read_text())) for rel in CAST_FILES)
    if left:
        sys.exit(f"[openfhe-hexl] ERROR: {left} uint64_t* casts left after patching")
    print(f"[openfhe-hexl] patched {found} uint64_t* casts -> BasicInteger* in {len(CAST_FILES)} files")


def patch_multd():
    p = ROOT / "src/core/include/math/hal/intnat-hexl/ubintnathexl.h"
    s = p.read_text()
    old = ("#elif __riscv\n"
           "        U128BITS wres(0), wa(a), wb(b);\n")
    new = ("#elif __riscv\n"
           "        // rvv-hexl: U128BITS only exists with HAVE_INT128 (not at NATIVE_SIZE=32)\n"
           "        unsigned __int128 wres(0), wa(a), wb(b);\n")
    if new in s:
        print("[openfhe-hexl] MultD riscv64 fix already applied")
        return
    if s.count(old) != 1:
        sys.exit("[openfhe-hexl] ERROR: MultD riscv branch not found: upstream changed, review "
                 + __file__)
    p.write_text(s.replace(old, new))
    print("[openfhe-hexl] patched MultD(): riscv64 branch without U128BITS")


patch_casts()
patch_multd()
