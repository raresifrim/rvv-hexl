#!/usr/bin/env python3
# Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
# All rights reserved.
#
"""Make binfhe's gadget decomposition call intel::hexl::SignedDigitDecompose (OpenFHE v1.5.1).

    openfhe-binfhe-decompose.py <staged openfhe-development tree>

Applied by third_party/openfhe.sh to the rvv-hexl tree only (stock builds stay
stock). Idempotent; fails loudly if upstream changes shape.

RingGSWAccumulator::SignedDigitDecompose (src/binfhe/lib/rgsw-acc.cpp) splits each
coefficient of the accumulator (two polynomials, or one ring element) into signed
base-2^g digits, coefficient by coefficient, each digit added into its own output
polynomial, the lowest digit dropped. With rvv-hexl both overloads become one
SignedDigitDecompose call per input polynomial (rvv-hexl extension,
HEXL_RVV_HAS_SIGNED_DIGIT_DECOMPOSE), writing the digit polynomials directly
(the ciphertext overload keeps OpenFHE's interleaving: digit d of polynomial c in
output[2d + c]).

The kernel stores; OpenFHE adds into its outputs. Every caller passes freshly
zero-initialised polynomials (AddToAccCGGI, AddToAccDM, AddToAccLMKCDEY and
LMKCDEY's Automorphism: NativePoly(..., Format::COEFFICIENT, true)), which this
script checks, so the values are the same.

Switch HEXL_DECOMPOSE_ENABLE in math/math-hal.h, from OPENFHE_HEXL_DECOMPOSE
(default 1): 1 calls rvv-hexl, 0 keeps OpenFHE's loop (the comparison baseline).
"""
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1])
MARK = "rvv-hexl: patches/openfhe-binfhe-decompose.py"
VALUE = os.environ.get("OPENFHE_HEXL_DECOMPOSE", "1")
if VALUE not in ("0", "1"):
    sys.exit(f"[openfhe-binfhe] ERROR: OPENFHE_HEXL_DECOMPOSE must be 0 or 1, got {VALUE!r}")

HAL = "src/core/include/math/math-hal.h"
ACC = "src/binfhe/lib/rgsw-acc.cpp"
CALLERS = ["src/binfhe/lib/rgsw-acc-cggi.cpp", "src/binfhe/lib/rgsw-acc-dm.cpp",
           "src/binfhe/lib/rgsw-acc-lmkcdey.cpp"]
ZEROED = re.compile(r"std::vector<NativePoly> \w+\(\w+, NativePoly\(params->GetPolyParams\(\), "
                    r"Format::COEFFICIENT, true\)\);\n\s*SignedDigitDecompose\(")


def fail(msg):
    sys.exit(f"[openfhe-binfhe] ERROR: {msg}: upstream changed, review {__file__}")


# the callers must hand over zero-initialised outputs (4 call sites)
zeroed = sum(len(ZEROED.findall((ROOT / c).read_text())) for c in CALLERS)
calls = sum((ROOT / c).read_text().count("SignedDigitDecompose(") for c in CALLERS)
if zeroed != 4 or calls != 4:
    fail(f"expected 4 SignedDigitDecompose calls on zeroed outputs, found {calls} calls, {zeroed} zeroed")

# the switch
p = ROOT / HAL
s = p.read_text()
if MARK in s:
    s2 = re.sub(r"#define HEXL_DECOMPOSE_ENABLE [01]", f"#define HEXL_DECOMPOSE_ENABLE {VALUE}", s)
    if s2 != s:
        p.write_text(s2)
        print(f"[openfhe-binfhe] HEXL_DECOMPOSE_ENABLE -> {VALUE}")
else:
    anchor = "#define HEXL_MUL_ENABLE 1\n"
    if s.count(anchor) != 1:
        fail(f"{HAL}: HEXL_MUL_ENABLE anchor")
    p.write_text(s.replace(anchor, anchor + f"#define HEXL_DECOMPOSE_ENABLE {VALUE}  // {MARK}\n"))

# the two overloads
p = ROOT / ACC
s = p.read_text()
if MARK in s:
    print(f"[openfhe-binfhe] {ACC}: decomposition patch already applied (HEXL_DECOMPOSE_ENABLE {VALUE})")
    sys.exit(0)

CT = ("                                              const std::vector<NativePoly>& input,\n"
      "                                              std::vector<NativePoly>& output) const {\n")
RING = ("                                              const NativePoly& input, std::vector<NativePoly>& output) const {\n")
for anchor, name in ((CT, "ciphertext overload"), (RING, "ring-element overload")):
    if s.count(anchor) != 1:
        fail(f"{ACC}: {name} signature")

CT_BODY = f"""#if HEXL_DECOMPOSE_ENABLE && defined(HEXL_RVV_HAS_SIGNED_DIGIT_DECOMPOSE)
    {{  // {MARK}: one call per polynomial, digit d of polynomial c into output[2d + c]
        const uint64_t Q{{params->GetQ().ConvertToInt<uint64_t>()}};
        const uint64_t gBits{{static_cast<uint64_t>(__builtin_ctz(params->GetBaseG()))}};
        const uint32_t digits{{params->GetDigitsG() - 1}};  // approximate: the first digit is dropped
        const uint32_t N{{params->GetN()}};
        std::vector<BasicInteger*> dst(digits);
        for (uint32_t c{{0}}; c < 2; ++c) {{
            for (uint32_t d{{0}}; d < digits; ++d)
                dst[d] = reinterpret_cast<BasicInteger*>(&output[2 * d + c][0]);
            intel::hexl::SignedDigitDecompose(dst.data(), reinterpret_cast<const BasicInteger*>(&input[c][0]), N, Q,
                                              gBits, digits);
        }}
        return;
    }}
#endif
"""
RING_BODY = f"""#if HEXL_DECOMPOSE_ENABLE && defined(HEXL_RVV_HAS_SIGNED_DIGIT_DECOMPOSE)
    {{  // {MARK}: one call, digit d into output[d]
        const uint64_t Q{{params->GetQ().ConvertToInt<uint64_t>()}};
        const uint64_t gBits{{static_cast<uint64_t>(__builtin_ctz(params->GetBaseG()))}};
        const uint32_t digits{{params->GetDigitsG() - 1}};  // approximate: the first digit is dropped
        std::vector<BasicInteger*> dst(digits);
        for (uint32_t d{{0}}; d < digits; ++d)
            dst[d] = reinterpret_cast<BasicInteger*>(&output[d][0]);
        intel::hexl::SignedDigitDecompose(dst.data(), reinterpret_cast<const BasicInteger*>(&input[0]),
                                          params->GetN(), Q, gBits, digits);
        return;
    }}
#endif
"""
s = s.replace(CT, CT + CT_BODY).replace(RING, RING + RING_BODY)
p.write_text(f"// {MARK}: SignedDigitDecompose calls rvv-hexl\n" + s)
print(f"[openfhe-binfhe] {ACC}: SignedDigitDecompose -> intel::hexl (HEXL_DECOMPOSE_ENABLE {VALUE})")
