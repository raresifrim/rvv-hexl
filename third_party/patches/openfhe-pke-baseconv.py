#!/usr/bin/env python3
# Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
# All rights reserved.
#
"""Make OpenFHE's approximate RNS base conversion call intel::hexl::BaseConvert (OpenFHE v1.5.1).

    openfhe-pke-baseconv.py <staged openfhe-development tree>

Applied by third_party/openfhe.sh to the rvv-hexl tree only (stock builds stay
stock), after the openfhe-hexl overlay is staged. Idempotent; fails loudly if
upstream changes shape.

HexlDCRTPolyImpl::ApproxSwitchCRTBasis (the overlay's copy of
DCRTPoly::ApproxSwitchCRTBasis, src/core/include/lattice/hal/hexl/hexldcrtpoly-impl.h)
converts a polynomial from one RNS basis to another, coefficient by coefficient:
128-bit products summed, one 128-bit Barrett reduction per output limb, scalar.
HYBRID key switching calls it in ModUp (ApproxModUp: each digit to the other
limbs and P) and in ModDown (ApproxModDown: P back to Q), so every CKKS (and
BFV / BGV HYBRID) relinearization and rotation. With rvv-hexl the whole
conversion becomes one BaseConvert call (rvv-hexl extension,
HEXL_RVV_HAS_BASE_CONVERT), bit for bit the same result. Only the branch that
upstream takes at NATIVE_SIZE=64 without WITH_REDUCED_NOISE is replaced.

Switch HEXL_BASECONV_ENABLE in math/math-hal.h, from OPENFHE_HEXL_BASECONV
(default 1): 1 calls rvv-hexl, 0 keeps OpenFHE's loop (the comparison
baseline).
"""
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1])
MARK = "rvv-hexl: patches/openfhe-pke-baseconv.py"
VALUE = os.environ.get("OPENFHE_HEXL_BASECONV", "1")
if VALUE not in ("0", "1"):
    sys.exit(f"[openfhe-pke] ERROR: OPENFHE_HEXL_BASECONV must be 0 or 1, got {VALUE!r}")

HAL = "src/core/include/math/math-hal.h"
IMPL = "src/core/include/lattice/hal/hexl/hexldcrtpoly-impl.h"


def fail(msg):
    sys.exit(f"[openfhe-pke] ERROR: {msg}: upstream changed, review {__file__}")


# the switch
p = ROOT / HAL
s = p.read_text()
if MARK in s:
    s2 = re.sub(r"#define HEXL_BASECONV_ENABLE [01]", f"#define HEXL_BASECONV_ENABLE {VALUE}", s)
    if s2 != s:
        p.write_text(s2)
        print(f"[openfhe-pke] HEXL_BASECONV_ENABLE -> {VALUE}")
else:
    anchor = "#define HEXL_MUL_ENABLE 1\n"
    if s.count(anchor) != 1:
        fail(f"{HAL}: HEXL_MUL_ENABLE anchor")
    p.write_text(s.replace(anchor, anchor + f"#define HEXL_BASECONV_ENABLE {VALUE}  // {MARK}\n"))

# the conversion
p = ROOT / IMPL
s = p.read_text()
if MARK in s:
    print(f"[openfhe-pke] {IMPL}: base conversion patch already applied (HEXL_BASECONV_ENABLE {VALUE})")
    sys.exit(0)

ANCHOR = """HexlDCRTPolyImpl<VecType> HexlDCRTPolyImpl<VecType>::ApproxSwitchCRTBasis(
    const std::shared_ptr<Params>& paramsQ, const std::shared_ptr<Params>& paramsP,
    const std::vector<NativeInteger>& QHatInvModq, const std::vector<NativeInteger>& QHatInvModqPrecon,
    const std::vector<std::vector<NativeInteger>>& QHatModp, const std::vector<DoubleNativeInt>& modpBarrettMu) const {
    HexlDCRTPolyImpl<VecType> ans(paramsP, m_format, true);
    uint32_t sizeQ = (m_vectors.size() > paramsQ->GetParams().size()) ? paramsQ->GetParams().size() : m_vectors.size();
    uint32_t sizeP = ans.m_vectors.size();
"""
if s.count(ANCHOR) != 1:
    fail(f"{IMPL}: ApproxSwitchCRTBasis prologue")

BODY = f"""#if HEXL_BASECONV_ENABLE && defined(HEXL_RVV_HAS_BASE_CONVERT) && (NATIVEINT == 64) && !defined(WITH_REDUCED_NOISE)
    {{  // {MARK}: the whole conversion in one call
        std::vector<uint64_t> q(sizeQ), p(sizeP), qHatInv(sizeQ), qHatModp(static_cast<size_t>(sizeQ) * sizeP);
        std::vector<const uint64_t*> in(sizeQ);
        std::vector<uint64_t*> out(sizeP);
        for (uint32_t i = 0; i < sizeQ; ++i) {{
            q[i]       = m_vectors[i].GetModulus().template ConvertToInt<uint64_t>();
            qHatInv[i] = QHatInvModq[i].ConvertToInt<uint64_t>();
            in[i]      = reinterpret_cast<const uint64_t*>(&m_vectors[i][0]);
            for (uint32_t j = 0; j < sizeP; ++j)
                qHatModp[static_cast<size_t>(i) * sizeP + j] = QHatModp[i][j].ConvertToInt<uint64_t>();
        }}
        for (uint32_t j = 0; j < sizeP; ++j) {{
            p[j]   = ans.m_vectors[j].GetModulus().template ConvertToInt<uint64_t>();
            out[j] = reinterpret_cast<uint64_t*>(&ans.m_vectors[j][0]);
        }}
        intel::hexl::BaseConvert(out.data(), in.data(), m_params->GetRingDimension(), q.data(), sizeQ, p.data(), sizeP,
                                 qHatInv.data(), qHatModp.data());
        return ans;
    }}
#endif
"""
s = s.replace(ANCHOR, ANCHOR + BODY)
p.write_text(f"// {MARK}: ApproxSwitchCRTBasis calls rvv-hexl\n" + s)
print(f"[openfhe-pke] {IMPL}: ApproxSwitchCRTBasis -> intel::hexl::BaseConvert (HEXL_BASECONV_ENABLE {VALUE})")
