#!/usr/bin/env python3
# Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
# All rights reserved.
#
"""Make OpenFHE's exact RNS base conversion call intel::hexl::BaseConvertExact (OpenFHE v1.5.1).

    openfhe-pke-baseconv-exact.py <staged openfhe-development tree>

Applied by third_party/openfhe.sh to the rvv-hexl tree only (stock builds stay
stock), after the openfhe-hexl overlay is staged. Idempotent; fails loudly if
upstream changes shape.

HexlDCRTPolyImpl::SwitchCRTBasis (the overlay's copy of DCRTPoly::SwitchCRTBasis,
src/core/include/lattice/hal/hexl/hexldcrtpoly-impl.h) is the HPS exact base
extension of BFV multiplication (ExpandCRTBasis, ExpandCRTBasisReverseOrder, and
the Pl -> Ql step of the leveled variants): per coefficient the y_i = x_i Q_i^-1
mod q_i, the double nu = 0.5 + sum y_i / q_i, alpha = floor(nu), then per output
limb a 128-bit sum and Barrett reduction minus alpha Q mod p_j, all scalar. With
rvv-hexl the whole conversion becomes one BaseConvertExact call (rvv-hexl
extension, HEXL_RVV_HAS_BASE_CONVERT_EXACT), the same arithmetic: nu summed in
OpenFHE's order, the same alpha row. NATIVE_SIZE=64 only (the kernel's storage);
SwitchCRTBasis's QHatModp is indexed [j][i], the kernel's q_hat_mod_p [i * K + j].

Switch HEXL_BASECONVEXACT_ENABLE in math/math-hal.h, from
OPENFHE_HEXL_BASECONV_EXACT (default 1): 1 calls rvv-hexl, 0 keeps OpenFHE's loop
(the comparison baseline).
"""
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1])
MARK = "rvv-hexl: patches/openfhe-pke-baseconv-exact.py"
VALUE = os.environ.get("OPENFHE_HEXL_BASECONV_EXACT", "1")
if VALUE not in ("0", "1"):
    sys.exit(f"[openfhe-pke] ERROR: OPENFHE_HEXL_BASECONV_EXACT must be 0 or 1, got {VALUE!r}")

HAL = "src/core/include/math/math-hal.h"
IMPL = "src/core/include/lattice/hal/hexl/hexldcrtpoly-impl.h"


def fail(msg):
    sys.exit(f"[openfhe-pke] ERROR: {msg}: upstream changed, review {__file__}")


# the switch
p = ROOT / HAL
s = p.read_text()
if MARK in s:
    s2 = re.sub(r"#define HEXL_BASECONVEXACT_ENABLE [01]", f"#define HEXL_BASECONVEXACT_ENABLE {VALUE}", s)
    if s2 != s:
        p.write_text(s2)
        print(f"[openfhe-pke] HEXL_BASECONVEXACT_ENABLE -> {VALUE}")
else:
    anchor = "#define HEXL_MUL_ENABLE 1\n"
    if s.count(anchor) != 1:
        fail(f"{HAL}: HEXL_MUL_ENABLE anchor")
    p.write_text(s.replace(anchor, anchor + f"#define HEXL_BASECONVEXACT_ENABLE {VALUE}  // {MARK}\n"))

# the conversion
p = ROOT / IMPL
s = p.read_text()
if MARK in s:
    print(f"[openfhe-pke] {IMPL}: exact base conversion patch already applied (HEXL_BASECONVEXACT_ENABLE {VALUE})")
    sys.exit(0)

ANCHOR = """HexlDCRTPolyImpl<VecType> HexlDCRTPolyImpl<VecType>::SwitchCRTBasis(
    const std::shared_ptr<Params>& paramsP, const std::vector<NativeInteger>& QHatInvModq,
    const std::vector<NativeInteger>& QHatInvModqPrecon, const std::vector<std::vector<NativeInteger>>& QHatModp,
    const std::vector<std::vector<NativeInteger>>& alphaQModp, const std::vector<DoubleNativeInt>& modpBarrettMu,
    const std::vector<double>& qInv) const {
    uint32_t sizeQ = m_vectors.size();
    uint32_t sizeP = paramsP->GetParams().size();
"""
if s.count(ANCHOR) != 1:
    fail(f"{IMPL}: SwitchCRTBasis prologue")

BODY = f"""#if HEXL_BASECONVEXACT_ENABLE && defined(HEXL_RVV_HAS_BASE_CONVERT_EXACT) && (NATIVEINT == 64)
    {{  // {MARK}: the whole conversion in one call
        HexlDCRTPolyImpl<VecType> ans(paramsP, m_format, true);
        std::vector<uint64_t> q(sizeQ), p(sizeP), qHatInv(sizeQ), qHatModp(static_cast<size_t>(sizeQ) * sizeP),
            alphaQ(static_cast<size_t>(sizeQ + 1) * sizeP);
        std::vector<const uint64_t*> in(sizeQ);
        std::vector<uint64_t*> out(sizeP);
        for (uint32_t i = 0; i < sizeQ; ++i) {{
            q[i]       = m_vectors[i].GetModulus().template ConvertToInt<uint64_t>();
            qHatInv[i] = QHatInvModq[i].ConvertToInt<uint64_t>();
            in[i]      = reinterpret_cast<const uint64_t*>(&m_vectors[i][0]);
            for (uint32_t j = 0; j < sizeP; ++j)
                qHatModp[static_cast<size_t>(i) * sizeP + j] = QHatModp[j][i].ConvertToInt<uint64_t>();
        }}
        for (uint32_t j = 0; j < sizeP; ++j) {{
            p[j]   = ans.m_vectors[j].GetModulus().template ConvertToInt<uint64_t>();
            out[j] = reinterpret_cast<uint64_t*>(&ans.m_vectors[j][0]);
        }}
        for (uint32_t a = 0; a <= sizeQ; ++a)
            for (uint32_t j = 0; j < sizeP; ++j)
                alphaQ[static_cast<size_t>(a) * sizeP + j] = alphaQModp[a][j].ConvertToInt<uint64_t>();
        intel::hexl::BaseConvertExact(out.data(), in.data(), m_params->GetRingDimension(), q.data(), sizeQ, p.data(),
                                      sizeP, qHatInv.data(), qHatModp.data(), qInv.data(), alphaQ.data());
        return ans;
    }}
#endif
"""
s = s.replace(ANCHOR, ANCHOR + BODY)
p.write_text(f"// {MARK}: SwitchCRTBasis calls rvv-hexl\n" + s)
print(f"[openfhe-pke] {IMPL}: SwitchCRTBasis -> intel::hexl::BaseConvertExact (HEXL_BASECONVEXACT_ENABLE {VALUE})")
