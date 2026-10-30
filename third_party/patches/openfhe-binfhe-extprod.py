#!/usr/bin/env python3
# Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
# All rights reserved.
#
"""Make binfhe's blind-rotation external products one intel::hexl::EltwiseExternalProductMod call (OpenFHE v1.5.1).

    openfhe-binfhe-extprod.py <staged openfhe-development tree>

Applied by third_party/openfhe.sh to the rvv-hexl tree only (stock builds stay
stock), after openfhe-binfhe-multadd.py and openfhe-binfhe-decompose.py.
Idempotent; fails loudly if upstream changes shape.

After the gadget decomposition and the forward NTTs, every blind-rotation step
multiplies the digit polynomials by the bootstrapping key and sums them per
accumulator polynomial: one EltwiseMultMod plus EltwiseMultAddMod per digit and
output (and, in CGGI, one more with the monomial). With rvv-hexl each of the four
sites becomes one EltwiseExternalProductMod call (rvv-hexl extension,
HEXL_RVV_HAS_EXTERNAL_PRODUCT), which reads every digit and key polynomial once:
    AddToAccCGGI    acc_j += X^a sum_i dct_i ek1[i][j] + X^-a sum_i dct_i ek2[i][j]
    AddToAccDM      acc_j  = sum_i dct_i ek[i][j]
    AddToAccLMKCDEY acc_j  = sum_i dct_i ek[i][j]
    Automorphism    acc_j += sum_d dcta_d ak[d][j]   (acc_0 zeroed just before)
OpenFHE's key layout ev[i][j] is the kernel's key[2 i + j]. All operands are
reduced polynomials in EVALUATION format; the accumulator aliases none of them.

Switch HEXL_EXTPROD_ENABLE in math/math-hal.h, from OPENFHE_HEXL_EXTPROD
(default 1): 1 calls rvv-hexl, 0 keeps OpenFHE's loops (the comparison
baseline: ModMulAddEq per digit, see openfhe-binfhe-multadd.py).
"""
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1])
MARK = "rvv-hexl: patches/openfhe-binfhe-extprod.py"
VALUE = os.environ.get("OPENFHE_HEXL_EXTPROD", "1")
if VALUE not in ("0", "1"):
    sys.exit(f"[openfhe-binfhe] ERROR: OPENFHE_HEXL_EXTPROD must be 0 or 1, got {VALUE!r}")

HAL = "src/core/include/math/math-hal.h"


def fail(msg):
    sys.exit(f"[openfhe-binfhe] ERROR: {msg}: upstream changed, review {__file__}")


def block(comment, digits, k, key1, mono1=None, key2=None, mono2=None, accumulate="false"):
    """The code inserted before OpenFHE's loops: one rvv-hexl call, then return."""
    keys = [("k1", key1)] + ([("k2", key2)] if key2 else [])
    decl = ", ".join(f"{name}(2 * {k})" for name, _ in keys)
    fill = "".join(f"\n                {name}[2 * i + j] = reinterpret_cast<const BasicInteger*>(&{src}[i][j][0]);"
                   for name, src in keys)

    def ptr(p):
        return f"reinterpret_cast<const BasicInteger*>(&{p}[0])" if p else "nullptr"

    return f"""#if HEXL_EXTPROD_ENABLE && defined(HEXL_RVV_HAS_EXTERNAL_PRODUCT)
    {{  // {MARK}: {comment}, one call
        std::vector<const BasicInteger*> d({k}), {decl};
        for (uint32_t i{{0}}; i < {k}; ++i) {{
            d[i] = reinterpret_cast<const BasicInteger*>(&{digits}[i][0]);
            for (uint32_t j{{0}}; j < 2; ++j) {{{fill}
            }}
        }}
        auto& av = acc->GetElements();
        intel::hexl::EltwiseExternalProductMod(reinterpret_cast<BasicInteger*>(&av[0][0]),
                                               reinterpret_cast<BasicInteger*>(&av[1][0]), d.data(), {k}, k1.data(),
                                               {ptr(mono1)}, {"k2.data()" if key2 else "nullptr"}, {ptr(mono2)},
                                               params->GetN(), params->GetQ().ConvertToInt<uint64_t>(), {accumulate});
        return;
    }}
#endif
"""


# file -> [(anchor, block)]: each block goes right before its anchor
SITES = {
    "src/binfhe/lib/rgsw-acc-cggi.cpp": [
        ("    const std::vector<std::vector<NativePoly>>& ev1(ek1->GetElements());\n",
         block("acc_j += X^a sum dct ek1 + X^-a sum dct ek2", "dct", "digitsG2", "ek1->GetElements()",
               mono1="monomial", key2="ek2->GetElements()", mono2="monomialNeg", accumulate="true")),
    ],
    "src/binfhe/lib/rgsw-acc-dm.cpp": [
        ("    const std::vector<std::vector<NativePoly>>& ev = ek->GetElements();\n",
         block("acc_j = sum dct ek", "dct", "digitsG2", "ek->GetElements()")),
    ],
    "src/binfhe/lib/rgsw-acc-lmkcdey.cpp": [
        ("    // acc = dct * ek (matrix product);\n    const auto& ev        = ek->GetElements();\n",
         block("acc_j = sum dct ek", "dct", "digitsG2", "ek->GetElements()")),
        ("    // acc = dct * input (matrix product);\n    const auto& ev = ak->GetElements();\n",
         block("acc_j += sum dcta ak (acc_0 zeroed above)", "dcta", "digitsG", "ak->GetElements()",
               accumulate="true")),
    ],
}

# the switch
p = ROOT / HAL
s = p.read_text()
if MARK in s:
    s2 = re.sub(r"#define HEXL_EXTPROD_ENABLE [01]", f"#define HEXL_EXTPROD_ENABLE {VALUE}", s)
    if s2 != s:
        p.write_text(s2)
        print(f"[openfhe-binfhe] HEXL_EXTPROD_ENABLE -> {VALUE}")
else:
    anchor = "#define HEXL_MUL_ENABLE 1\n"
    if s.count(anchor) != 1:
        fail(f"{HAL}: HEXL_MUL_ENABLE anchor")
    p.write_text(s.replace(anchor, anchor + f"#define HEXL_EXTPROD_ENABLE {VALUE}  // {MARK}\n"))

# the call sites
for rel, edits in SITES.items():
    p = ROOT / rel
    s = p.read_text()
    if MARK in s:
        print(f"[openfhe-binfhe] {rel}: external-product patch already applied (HEXL_EXTPROD_ENABLE {VALUE})")
        continue
    for anchor, code in edits:
        if s.count(anchor) != 1:
            fail(f"{rel}: anchor {anchor.strip()!r} found {s.count(anchor)} times")
        s = s.replace(anchor, code + anchor)
    p.write_text(f"// {MARK}: the external products call rvv-hexl\n" + s)
    print(f"[openfhe-binfhe] {rel}: {len(edits)} external product(s) -> intel::hexl::EltwiseExternalProductMod "
          f"(HEXL_EXTPROD_ENABLE {VALUE})")
