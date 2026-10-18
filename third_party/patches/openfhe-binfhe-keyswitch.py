#!/usr/bin/env python3
# Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
# SPDX-License-Identifier: Apache-2.0
#
"""Vectorize binfhe's LWE key switching through NativeVector (OpenFHE v1.5.1).

    openfhe-binfhe-keyswitch.py <staged openfhe-development tree>

Applied by third_party/openfhe.sh to the rvv-hexl tree only (stock builds stay
stock), after openfhe-hexl-sub.py. Idempotent; fails loudly if upstream changes.

LWEEncryptionScheme::KeySwitch (src/binfhe/lib/lwe-pke.cpp) subtracts N x
digitCount key rows of length n from the accumulator, element by element:
    for (k < n) a[k].ModSubFastEq(refAj[k], Q);
This patch makes each row one vector call, a.ModSubEq(refAj), which on the
rvv-hexl build is intel::hexl::EltwiseSubMod (openfhe-hexl-sub.py). Same
result: the rows come from DiscreteUniformGenerator(qKS).GenerateVector(n)
(modulus qKS = Q, length n, values < Q), so ModSubEq's parameter check holds
and both paths compute (a - row) mod Q on reduced inputs.
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
MARK = "rvv-hexl: patches/openfhe-binfhe-keyswitch.py"
REL = "src/binfhe/lib/lwe-pke.cpp"
OLD = """            auto& refAj = refA[a0][j];
            for (uint32_t k = 0; k < n; ++k)
                a[k].ModSubFastEq(refAj[k], Q);
"""
NEW = f"""            a.ModSubEq(refA[a0][j]);  // {MARK}: one vector subtraction per key row
"""

p = ROOT / REL
s = p.read_text()
if MARK in s:
    print("[openfhe-binfhe] key-switch patch already applied")
elif s.count(OLD) == 1:
    p.write_text(s.replace(OLD, NEW))
    print("[openfhe-binfhe] KeySwitch: per-element ModSubFastEq loop -> NativeVector::ModSubEq")
else:
    sys.exit(f"[openfhe-binfhe] ERROR: KeySwitch inner loop not found once in {REL}: upstream changed, review {__file__}")
