#!/usr/bin/env python3
# Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
# SPDX-License-Identifier: Apache-2.0
#
"""Route OpenFHE's modular subtractions to HEXL (openfhe-hexl v1.5.1.0).

    openfhe-hexl-sub.py <staged openfhe-development tree>

Applied by third_party/openfhe.sh after openfhe-hexl-wordsize.py (it writes
BasicInteger* casts, which that patch expects to find already converted), for
every rvv-hexl build. Idempotent; fails loudly if upstream changes shape.

The overlay never routes subtraction: its NativeVector ModSub / ModSubEq
(vector and scalar) are OpenFHE's own loops, and no openfhe-hexl commit ever
called intel::hexl::EltwiseSubMod, although HEXL has it since v1.1.0. Every
polynomial-level subtraction ends in those four methods (HexlPoly Minus /
operator-= / unary minus / Negate, which the overlay computes as 0 -= x, and
HexlDCRTPoly Minus / operator-= / Negate per tower), so this patch routes all
of them. Same contract on both paths (inputs already reduced mod q: the loops
use ModSubFastEq). Adds the switch HEXL_SUB_ENABLE next to HEXL_ADD_ENABLE /
HEXL_MUL_ENABLE in math/math-hal.h, set to 1. Not routed: the element-wise
ModSubFast steps inside the RNS basis conversions (HexlDCRTPoly
SwitchCRTBasis, FastBaseConv*, ...).
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
MARK = "rvv-hexl: patches/openfhe-hexl-sub.py"

HAL = "src/core/include/math/math-hal.h"
HAL_OLD = "#define HEXL_MUL_ENABLE 1\n"
HAL_NEW = HAL_OLD + f"#define HEXL_SUB_ENABLE 1  // {MARK}\n"

VEC = "src/core/lib/math/hal/intnat-hexl/mubintvecnathexl.cpp"
VEC_EDITS = [
    # ModSub(scalar)
    ("""NativeVectorT<IntegerType> NativeVectorT<IntegerType>::ModSub(const IntegerType& b) const {
    auto mv{m_modulus};
    auto bv{b};
    auto ans(*this);
    if (bv.m_value >= mv.m_value)
        bv.ModEq(mv);
    for (size_t i = 0; i < ans.m_data.size(); ++i)
        ans[i].ModSubFastEq(bv, mv);
    return ans;
}
""", f"""NativeVectorT<IntegerType> NativeVectorT<IntegerType>::ModSub(const IntegerType& b) const {{
    auto mv{{m_modulus}};
    auto bv{{b}};
    if (bv.m_value >= mv.m_value)
        bv.ModEq(mv);
    if constexpr (HEXL_SUB_ENABLE && std::is_same_v<IntegerType, NativeInteger>) {{  // {MARK}
        NativeVectorT ans(m_data.size(), m_modulus);
        BasicInteger* res       = reinterpret_cast<BasicInteger*>(&ans.m_data[0]);
        const BasicInteger* op1 = reinterpret_cast<const BasicInteger*>(&m_data[0]);
        intel::hexl::EltwiseSubMod(res, op1, bv.m_value, m_data.size(), mv.m_value);
        return ans;
    }}
    else {{
        auto ans(*this);
        for (size_t i = 0; i < ans.m_data.size(); ++i)
            ans[i].ModSubFastEq(bv, mv);
        return ans;
    }}
}}
"""),
    # ModSubEq(scalar)
    ("""NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::ModSubEq(const IntegerType& b) {
    auto mv{m_modulus};
    auto bv{b};
    if (bv.m_value >= mv.m_value)
        bv.ModEq(mv);
    for (size_t i = 0; i < m_data.size(); ++i)
        m_data[i].ModSubFastEq(bv, mv);
    return *this;
}
""", f"""NativeVectorT<IntegerType>& NativeVectorT<IntegerType>::ModSubEq(const IntegerType& b) {{
    auto mv{{m_modulus}};
    auto bv{{b}};
    if (bv.m_value >= mv.m_value)
        bv.ModEq(mv);
    if constexpr (HEXL_SUB_ENABLE && std::is_same_v<IntegerType, NativeInteger>) {{  // {MARK}
        BasicInteger* op1 = reinterpret_cast<BasicInteger*>(&m_data[0]);
        intel::hexl::EltwiseSubMod(op1, op1, bv.m_value, m_data.size(), mv.m_value);
    }}
    else {{
        for (size_t i = 0; i < m_data.size(); ++i)
            m_data[i].ModSubFastEq(bv, mv);
    }}
    return *this;
}}
"""),
    # ModSub(vector)
    ("""    auto mv{m_modulus};
    auto ans(*this);
    for (size_t i = 0; i < ans.m_data.size(); ++i)
        ans[i].ModSubFastEq(b[i], mv);
    return ans;
}
""", f"""    if constexpr (HEXL_SUB_ENABLE && std::is_same_v<IntegerType, NativeInteger>) {{  // {MARK}
        NativeVectorT ans(m_data.size(), m_modulus);
        BasicInteger* res       = reinterpret_cast<BasicInteger*>(&ans.m_data[0]);
        const BasicInteger* op1 = reinterpret_cast<const BasicInteger*>(&m_data[0]);
        const BasicInteger* op2 = reinterpret_cast<const BasicInteger*>(&b.m_data[0]);
        intel::hexl::EltwiseSubMod(res, op1, op2, m_data.size(), m_modulus.m_value);
        return ans;
    }}
    else {{
        auto mv{{m_modulus}};
        auto ans(*this);
        for (size_t i = 0; i < ans.m_data.size(); ++i)
            ans[i].ModSubFastEq(b[i], mv);
        return ans;
    }}
}}
"""),
    # ModSubEq(vector)
    ("""        OPENFHE_THROW("ModSubEq called on NativeVectorT's with different parameters.");
    for (size_t i = 0; i < m_data.size(); ++i)
        m_data[i].ModSubFastEq(b[i], m_modulus);
    return *this;
}
""", f"""        OPENFHE_THROW("ModSubEq called on NativeVectorT's with different parameters.");
    if constexpr (HEXL_SUB_ENABLE && std::is_same_v<IntegerType, NativeInteger>) {{  // {MARK}
        BasicInteger* op1       = reinterpret_cast<BasicInteger*>(&m_data[0]);
        const BasicInteger* op2 = reinterpret_cast<const BasicInteger*>(&b.m_data[0]);
        intel::hexl::EltwiseSubMod(op1, op1, op2, m_data.size(), m_modulus.m_value);
    }}
    else {{
        for (size_t i = 0; i < m_data.size(); ++i)
            m_data[i].ModSubFastEq(b[i], m_modulus);
    }}
    return *this;
}}
"""),
]


def patch(rel, edits):
    p = ROOT / rel
    s = p.read_text()
    if MARK in s:
        return 0
    for old, new in edits:
        if s.count(old) != 1:
            sys.exit(f"[openfhe-hexl] ERROR: {rel}: expected text found {s.count(old)} times "
                     f"(not 1): upstream changed, review {__file__}\n--- expected:\n{old}")
        s = s.replace(old, new)
    p.write_text(s)
    return len(edits)


n = patch(HAL, [(HAL_OLD, HAL_NEW)]) + patch(VEC, VEC_EDITS)
print("[openfhe-hexl] HEXL_SUB_ENABLE already applied" if n == 0
      else f"[openfhe-hexl] HEXL_SUB_ENABLE 1: ModSub / ModSubEq (vector, scalar) -> EltwiseSubMod ({n} edits)")
