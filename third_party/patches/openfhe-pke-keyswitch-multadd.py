#!/usr/bin/env python3
# Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
# SPDX-License-Identifier: Apache-2.0
#
"""Make OpenFHE's hybrid key switching accumulate with ModMulAddEq (OpenFHE v1.5.1).

    openfhe-pke-keyswitch-multadd.py <staged openfhe-development tree>

Applied by third_party/openfhe.sh to the rvv-hexl tree only (stock builds stay
stock), after openfhe-hexl-multadd.py, which adds ModMulAddEq. Idempotent; fails
loudly if upstream changes shape.

KeySwitchHYBRID::EvalFastKeySwitchCoreExt (every relinearization and rotation in
BFV, BGV and CKKS) accumulates the digit x key products limb by limb as
    elements[0].SetElementAtIndex(i, elements[0].GetElementAtIndex(i) + cji * bji);
a multiplication into a temporary polynomial, an addition into a second one, then
a copy back. Each line becomes elements[.].GetAllElements()[i].ModMulAddEq(cji, .):
one fused call (EltwiseMultAddMod) when HEXL_MULADD_ENABLE is 1, the two HEXL calls
in place when it is 0. Same result: all operands are reduced polynomials in
EVALUATION format, over the same modulus.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1])
REL = "src/pke/lib/keyswitch/keyswitch-hybrid.cpp"
MARK = "rvv-hexl: patches/openfhe-pke-keyswitch-multadd.py"
LINE = re.compile(r"^(\s*)elements\[([01])\]\.SetElementAtIndex\(i, elements\[\2\]\.GetElementAtIndex\(i\) \+ cji \* ([ab]ji)\);$", re.M)

p = ROOT / REL
s = p.read_text()
if MARK in s:
    print(f"[openfhe-pke] {REL}: multiply-add patch already applied")
    sys.exit(0)
found = len(LINE.findall(s))
if found != 2:
    sys.exit(f"[openfhe-pke] ERROR: {REL}: expected 2 accumulation lines, found {found}: "
             f"upstream changed, review {__file__}")
s = LINE.sub(lambda m: f"{m.group(1)}elements[{m.group(2)}].GetAllElements()[i].ModMulAddEq(cji, {m.group(3)});", s)
p.write_text(f"// {MARK}: key-switch accumulation uses ModMulAddEq\n" + s)
print(f"[openfhe-pke] {REL}: {found} accumulations -> ModMulAddEq")
