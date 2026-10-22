#!/usr/bin/env python3
# Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
# All rights reserved.
#
"""Route OpenFHE's modular additions to HEXL (openfhe-hexl v1.5.1.0).

    openfhe-hexl-add.py <staged openfhe-development tree>

Applied by third_party/openfhe.sh after openfhe-hexl-wordsize.py, for every
rvv-hexl build. Idempotent; fails loudly if upstream changes shape.

The overlay ships with HEXL_ADD_ENABLE 0 (math/math-hal.h, since its v1.1.1
update, no reason given), so OpenFHE keeps its own loops for vector and
polynomial additions (NativeVector ModAdd / ModAddEq / ModAddNoCheckEq,
HexlPoly operator+= and AddILElementOne, HexlDCRTPoly Plus) and only the
multiplications reach HEXL. Both paths have the same contract (inputs already
reduced mod q: OpenFHE's own loop is ModAddFastEq), so this changes speed, not
results. Measured on the K3 X100 (N=4096, eltwise benchmark, 2026-10-14):
OpenFHE's own vector add 2.86 ns/element, rvv-hexl's RVV EltwiseAddMod 0.66.
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
HAL = ROOT / "src/core/include/math/math-hal.h"
OLD = "#define HEXL_ADD_ENABLE 0\n"
NEW = "#define HEXL_ADD_ENABLE 1  // rvv-hexl: patches/openfhe-hexl-add.py\n"

s = HAL.read_text()
if NEW in s:
    print("[openfhe-hexl] HEXL_ADD_ENABLE already 1")
elif s.count(OLD) == 1:
    HAL.write_text(s.replace(OLD, NEW))
    print("[openfhe-hexl] HEXL_ADD_ENABLE 0 -> 1: additions go to HEXL")
else:
    sys.exit(f"[openfhe-hexl] ERROR: '{OLD.strip()}' not found once in {HAL}: upstream changed, review")
