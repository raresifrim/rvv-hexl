#!/usr/bin/env python3
"""Give OpenFHE a fused vector multiply-add, backed by rvv-hexl's EltwiseMultAddMod.

    OPENFHE_HEXL_MULADD=0|1 openfhe-hexl-multadd.py <staged openfhe-development tree>

Applied by third_party/openfhe.sh after the add / sub patches, for every rvv-hexl
build. Idempotent (re-running with another OPENFHE_HEXL_MULADD only updates the
switch); fails loudly if upstream changes shape.

Adds NativeVector::ModMulAddEq(a, b) and NativePoly (HexlPolyImpl)::ModMulAddEq(a, b):
this = (this + a * b) mod q, element-wise, all operands reduced mod q. It is what
patches/openfhe-binfhe-multadd.py makes binfhe's external product call instead of
"acc += (x * y)" (a multiplication into a temporary polynomial, then an addition).

Switch HEXL_MULADD_ENABLE in math/math-hal.h, from OPENFHE_HEXL_MULADD (default 1):
  1  one intel::hexl::EltwiseMultAddMod call (rvv-hexl extension,
     HEXL_RVV_HAS_MULT_ADD_MOD): one pass, no temporary;
  0  EltwiseMultMod into a temporary, then EltwiseAddMod: the same two calls as
     the code it replaces (the comparison baseline).
Per call the fused kernels are 1.05-1.33x faster (bench-hexl BM_EltwiseMultAddMod
vs BM_EltwiseMultModThenAddMod, 2026-10-14); end to end, IPCEI TFHE benches against
0 (2026-10-15): X100 RVV 1.03-1.06x, X100 scalar 1.06-1.17x, A100 RVV 1.01-1.04x,
A100 scalar 0.98-1.01x.
"""
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1])
MARK = "rvv-hexl: patches/openfhe-hexl-multadd.py"
VALUE = os.environ.get("OPENFHE_HEXL_MULADD", "1")
if VALUE not in ("0", "1"):
    sys.exit(f"[openfhe-hexl] ERROR: OPENFHE_HEXL_MULADD must be 0 or 1, got {VALUE!r}")

HAL = "src/core/include/math/math-hal.h"
VEC_H = "src/core/include/math/hal/intnat-hexl/mubintvecnathexl.h"
VEC_CPP = "src/core/lib/math/hal/intnat-hexl/mubintvecnathexl.cpp"
POLY_H = "src/core/include/lattice/hal/hexl/hexlpoly.h"

EDITS = {
    HAL: [("#define HEXL_MUL_ENABLE 1\n",
           f"#define HEXL_MUL_ENABLE 1\n#define HEXL_MULADD_ENABLE {VALUE}  // {MARK}\n")],
    VEC_H: [("    NativeVectorT& ModMulEq(const NativeVectorT& b);\n",
             "    NativeVectorT& ModMulEq(const NativeVectorT& b);\n"
             f"    // this = (this + a * b) mod q, element-wise; {MARK}\n"
             "    NativeVectorT& ModMulAddEq(const NativeVectorT& a, const NativeVectorT& b);\n")],
    VEC_CPP: [("template <class IntegerType>\nNativeVectorT<IntegerType>& NativeVectorT<IntegerType>::MultAccEqNoCheck(",
               f"""template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::ModMulAddEq(const NativeVectorT& a,
                                                                const NativeVectorT& b) {{  // {MARK}
    if (m_data.size() != a.m_data.size() || m_data.size() != b.m_data.size() || m_modulus != a.m_modulus ||
        m_modulus != b.m_modulus)
        OPENFHE_THROW("ModMulAddEq called on NativeVectorT's with different parameters.");
    if constexpr (std::is_same_v<IntegerType, NativeInteger>) {{
        BasicInteger* acc       = reinterpret_cast<BasicInteger*>(&m_data[0]);
        const BasicInteger* op1 = reinterpret_cast<const BasicInteger*>(&a.m_data[0]);
        const BasicInteger* op2 = reinterpret_cast<const BasicInteger*>(&b.m_data[0]);
    #if HEXL_MULADD_ENABLE && defined(HEXL_RVV_HAS_MULT_ADD_MOD)
        intel::hexl::EltwiseMultAddMod(acc, op1, op2, acc, m_data.size(), m_modulus.m_value, 1);
    #else
        std::vector<BasicInteger> prod(m_data.size());  // the two calls of "acc += (a * b)"
        intel::hexl::EltwiseMultMod(prod.data(), op1, op2, m_data.size(), m_modulus.m_value, 1);
        intel::hexl::EltwiseAddMod(acc, acc, prod.data(), m_data.size(), m_modulus.m_value);
    #endif
    }}
    else {{
        for (size_t i = 0; i < m_data.size(); ++i)
            m_data[i].ModAddFastEq(a.m_data[i].ModMul(b.m_data[i], m_modulus), m_modulus);
    }}
    return *this;
}}

template <class IntegerType>
NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::MultAccEqNoCheck(""")],
    POLY_H: [("    HexlPolyImpl& MultAccEqNoCheck(const HexlPolyImpl& V, const Integer& I) override {\n",
              f"""    // this = this + a * b, element-wise (EVALUATION format); {MARK}
    HexlPolyImpl& ModMulAddEq(const HexlPolyImpl& a, const HexlPolyImpl& b) {{
        if constexpr (std::is_same_v<VecType, NativeVector>) {{
            if (!m_values)
                m_values = std::make_unique<VecType>(m_params->GetRingDimension(), m_params->GetModulus());
            m_values->ModMulAddEq(*a.m_values, *b.m_values);
            return *this;
        }}
        else {{
            return *this += a.Times(b);
        }}
    }}

    HexlPolyImpl& MultAccEqNoCheck(const HexlPolyImpl& V, const Integer& I) override {{
""")],
}

applied = 0
for rel, edits in EDITS.items():
    p = ROOT / rel
    s = p.read_text()
    if MARK in s:
        if rel == HAL:  # only the switch can change on a re-run
            s2 = re.sub(r"#define HEXL_MULADD_ENABLE [01]", f"#define HEXL_MULADD_ENABLE {VALUE}", s)
            if s2 != s:
                p.write_text(s2)
                print(f"[openfhe-hexl] HEXL_MULADD_ENABLE -> {VALUE}")
        continue
    for old, new in edits:
        if s.count(old) != 1:
            sys.exit(f"[openfhe-hexl] ERROR: {rel}: expected text found {s.count(old)} times (not 1): "
                     f"upstream changed, review {__file__}\n--- expected:\n{old}")
        s = s.replace(old, new)
    p.write_text(s)
    applied += len(edits)
print(f"[openfhe-hexl] ModMulAddEq added ({applied} edits), HEXL_MULADD_ENABLE {VALUE}" if applied
      else f"[openfhe-hexl] ModMulAddEq already applied (HEXL_MULADD_ENABLE {VALUE})")
