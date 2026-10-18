#!/usr/bin/env python3
# Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
# SPDX-License-Identifier: Apache-2.0
#
"""Make binfhe's external products accumulate with NativePoly::ModMulAddEq (OpenFHE v1.5.1).

    openfhe-binfhe-multadd.py <staged openfhe-development tree>

Applied by third_party/openfhe.sh to the rvv-hexl tree only (stock builds stay
stock), after openfhe-hexl-multadd.py, which adds ModMulAddEq. Idempotent; fails
loudly if upstream changes shape.

The blind-rotation accumulators (src/binfhe/lib/rgsw-acc-{cggi,dm,lmkcdey}.cpp)
accumulate digit x key products, and CGGI also tmp x monomial, as
    X += (A * B);   or   X += (A *= B);
i.e. a multiplication into a temporary polynomial, then an addition. Each
becomes X.ModMulAddEq(A, B): one fused call (EltwiseMultAddMod) when
HEXL_MULADD_ENABLE is 1, the same two HEXL calls as before when it is 0. Same
result: all operands are reduced polynomials in EVALUATION format; the "*="
forms only modified A in place as a last use. Key generation (result[i][1]
+= ...) is left alone.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1])
MARK = "rvv-hexl: patches/openfhe-binfhe-multadd.py"
FILES = {  # file -> number of accumulation lines expected
    "src/binfhe/lib/rgsw-acc-cggi.cpp": 8,     # AddToAccCGGI: 4 digit sums + 4 monomial products
    "src/binfhe/lib/rgsw-acc-dm.cpp": 2,       # AddToAccDM
    "src/binfhe/lib/rgsw-acc-lmkcdey.cpp": 4,  # AddToAccLMKCDEY, Automorphism
}
# "<indent>X += (A * B);" or "... *= ..." with X the accumulator or CGGI's tmp
LINE = re.compile(r"^(\s*)(tmp|acc->GetElements\(\)\[[01]\]) \+= \((.+?) \*=? (.+?)\);$", re.M)

for rel, want in FILES.items():
    p = ROOT / rel
    s = p.read_text()
    if MARK in s:
        print(f"[openfhe-binfhe] {rel}: multiply-add patch already applied")
        continue
    found = len(LINE.findall(s))
    if found != want:
        sys.exit(f"[openfhe-binfhe] ERROR: {rel}: expected {want} accumulation lines, found {found}: "
                 f"upstream changed, review {__file__}")
    s = LINE.sub(lambda m: f"{m.group(1)}{m.group(2)}.ModMulAddEq({m.group(3)}, {m.group(4)});", s)
    p.write_text(f"// {MARK}: accumulations use ModMulAddEq\n" + s)
    print(f"[openfhe-binfhe] {rel}: {found} accumulations -> ModMulAddEq")
