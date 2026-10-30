#!/usr/bin/env python3
# Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
# All rights reserved.
#
"""Make OpenFHE's key-switching inner products one EltwiseDotProductMod call per limb (OpenFHE v1.5.1).

    openfhe-pke-keyswitch-dot.py <staged openfhe-development tree>

Applied by third_party/openfhe.sh to the rvv-hexl tree only (stock builds stay
stock), after openfhe-pke-keyswitch-multadd.py. Idempotent; fails loudly if
upstream changes shape. Both techniques (every relinearization and rotation in
BFV, BGV and CKKS):

- KeySwitchBV::EvalFastKeySwitchCore: for every limb j of Ql,
      res0_j = sum_i digit_i,j * b_i,j,   res1_j = sum_i digit_i,j * a_i,j
- KeySwitchHYBRID::EvalFastKeySwitchCoreExt: the same for every limb i of QlP
  over the dnum digits, with the key limb idx = i (+ delta for the P limbs).

openfhe-pke-keyswitch-multadd.py runs them as one ModMulAddEq per digit, limb
and result. With rvv-hexl each limb becomes one EltwiseDotProductMod call
(rvv-hexl extension, HEXL_RVV_HAS_DOT_PRODUCT): M = 2 results, key[2 i] = b,
key[2 i + 1] = a, input_mod_factor 1 (all operands reduced, EVALUATION format),
overwriting the zero-initialised results; every limb once, OpenMP over the limbs
(HYBRID's loop is turned around: limbs outside, digits inside the call).

Switch HEXL_DOTPROD_ENABLE in math/math-hal.h, from OPENFHE_HEXL_DOTPROD
(default 1): 1 calls rvv-hexl, 0 keeps the ModMulAddEq loops (the comparison
baseline).
"""
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1])
MARK = "rvv-hexl: patches/openfhe-pke-keyswitch-dot.py"
MULTADD = "rvv-hexl: patches/openfhe-pke-keyswitch-multadd.py"
VALUE = os.environ.get("OPENFHE_HEXL_DOTPROD", "1")
if VALUE not in ("0", "1"):
    sys.exit(f"[openfhe-pke] ERROR: OPENFHE_HEXL_DOTPROD must be 0 or 1, got {VALUE!r}")

HAL = "src/core/include/math/math-hal.h"
BV = "src/pke/lib/keyswitch/keyswitch-bv.cpp"
HYBRID = "src/pke/lib/keyswitch/keyswitch-hybrid.cpp"
GUARD = "#if HEXL_DOTPROD_ENABLE && defined(HEXL_RVV_HAS_DOT_PRODUCT)"


def fail(msg):
    sys.exit(f"[openfhe-pke] ERROR: {msg}: upstream changed, review {__file__}")


def ptr(expr, const=True):
    return f"reinterpret_cast<{'const ' if const else ''}BasicInteger*>(&{expr}[0])"


def limb_call(limbs, limb, key_limb, out0, out1, ret):
    """One dot product per limb over the `limit` digits, OpenMP over the limbs, then return."""
    return f"""{GUARD}
    {{  // {MARK}: one dot product per limb, both results
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit({limbs}))
        for (uint32_t {limb} = 0; {limb} < {limbs}; ++{limb}) {{
            std::vector<const BasicInteger*> d(limit), k(2 * limit);
            for (uint32_t t = 0; t < limit; ++t) {{
                d[t]         = {ptr(f"(*digits)[t].GetElementAtIndex({limb})")};
                k[2 * t]     = {ptr(f"bv[t].GetElementAtIndex({key_limb})")};
                k[2 * t + 1] = {ptr(f"av[t].GetElementAtIndex({key_limb})")};
            }}
            auto& r0           = {out0};
            BasicInteger* r[2] = {{{ptr("r0", False)}, {ptr(out1, False)}}};
            intel::hexl::EltwiseDotProductMod(r, 2, d.data(), limit, k.data(), nullptr, r0.GetRingDimension(),
                                              r0.GetModulus().ConvertToInt<uint64_t>(), 1, false);
        }}
        return {ret};
    }}
#endif
"""


SITES = {
    BV: ("    std::vector<DCRTPoly> res{DCRTPoly(paramsQl, Format::EVALUATION, true), "
         "DCRTPoly(paramsQl, Format::EVALUATION, true)};\n",
         limb_call("sizeQl", "j", "j", "res[0].GetAllElements()[j]", "res[1].GetAllElements()[j]",
                   "std::make_shared<std::vector<DCRTPoly>>(std::move(res))")),
    HYBRID: ("    auto& elements = (*result);\n",
             limb_call("sizeQlP", "i", "(i >= sizeQl) ? i + delta : i", "elements[0].GetAllElements()[i]",
                       "elements[1].GetAllElements()[i]", "result")),
}

# the switch
p = ROOT / HAL
s = p.read_text()
if MARK in s:
    s2 = re.sub(r"#define HEXL_DOTPROD_ENABLE [01]", f"#define HEXL_DOTPROD_ENABLE {VALUE}", s)
    if s2 != s:
        p.write_text(s2)
        print(f"[openfhe-pke] HEXL_DOTPROD_ENABLE -> {VALUE}")
else:
    anchor = "#define HEXL_MUL_ENABLE 1\n"
    if s.count(anchor) != 1:
        fail(f"{HAL}: HEXL_MUL_ENABLE anchor")
    p.write_text(s.replace(anchor, anchor + f"#define HEXL_DOTPROD_ENABLE {VALUE}  // {MARK}\n"))

# the two cores: the block goes right after the zero-initialised results
for rel, (anchor, code) in SITES.items():
    p = ROOT / rel
    s = p.read_text()
    if MARK in s:
        print(f"[openfhe-pke] {rel}: dot-product patch already applied (HEXL_DOTPROD_ENABLE {VALUE})")
        continue
    if MULTADD not in s:
        fail(f"{rel}: openfhe-pke-keyswitch-multadd.py must be applied first")
    if s.count(anchor) != 1:
        fail(f"{rel}: anchor {anchor.strip()!r} found {s.count(anchor)} times")
    s = s.replace(anchor, anchor + code)
    p.write_text(f"// {MARK}: the key-switching inner products call rvv-hexl\n" + s)
    print(f"[openfhe-pke] {rel}: key switching -> intel::hexl::EltwiseDotProductMod (HEXL_DOTPROD_ENABLE {VALUE})")
