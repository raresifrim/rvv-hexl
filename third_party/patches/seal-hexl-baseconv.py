#!/usr/bin/env python3
# Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
# SPDX-License-Identifier: Apache-2.0
#
"""Make SEAL's RNS base conversion use EltwiseReduceMod / EltwiseFMAMod (SEAL v4.1.2).

    seal-hexl-baseconv.py <SEAL source tree>

Applied by third_party/seal.sh before every build. Idempotent; fails loudly if
upstream changes shape. The new code is compiled only with SEAL_USE_INTEL_HEXL, so
the stock build is unchanged; -DSEAL_RVV_HEXL_BASECONV=0 selects the original code
in an HEXL build too (seal.sh: SEAL_RVV_HEXL_BASECONV=0).

BaseConverter::fast_convert_array (BEHZ: fastbconv, fastbconv_sk, fast_floor,
sm_mrq, ...) computes, for every output modulus p_k and coefficient j,
    out_k[j] = sum_i (in_i[j] * w_i mod q_i) * M[k][i]  mod p_k
SEAL does it coefficient by coefficient: a coefficient-major temporary, then one
128-bit dot product (dot_product_mod) per coefficient and output modulus. Here it
runs limb by limb on whole vectors: in_i is reduced mod q_i (its range is not
assumed) and multiplied by w_i (EltwiseReduceMod, EltwiseFMAMod), then each output
limb accumulates temp_i * M[k][i] with EltwiseFMAMod. temp_i < q_i enters
EltwiseFMAMod with input_mod_factor = the smallest of 1, 2, 4, 8 covering q_i / p_k,
or is first reduced mod p_k when q_i > 8 p_k (e.g. m_tilde = 2^32). Same result:
every partial sum is reduced mod p_k.
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
REL = "native/src/seal/util/rns.cpp"
MARK = "rvv-hexl: patches/seal-hexl-baseconv.py"

p = ROOT / REL
s = p.read_text()
if MARK in s:
    print(f"[seal] {REL}: base-conversion patch already applied")
    sys.exit(0)

START = "        void BaseConverter::fast_convert_array(ConstRNSIter in, RNSIter out, MemoryPoolHandle pool) const\n        {\n"
if s.count(START) != 1:
    sys.exit(f"[seal] ERROR: {REL}: fast_convert_array not found once; review {__file__}")
a = s.index(START) + len(START)

body = f"""#if defined(SEAL_USE_INTEL_HEXL) && SEAL_RVV_HEXL_BASECONV
            // {MARK}: limb by limb on whole vectors (see the patch)
            {{
                const size_t ibase_size = ibase_.size();
                const size_t obase_size = obase_.size();
                const size_t count = in.poly_modulus_degree();
                auto temp(allocate_uint(count * ibase_size, pool));
                auto scratch(allocate_uint(count, pool));
                for (size_t i = 0; i < ibase_size; i++)
                {{
                    const uint64_t qi = ibase_.base()[i].value();
                    uint64_t *t = temp.get() + i * count;
                    intel::hexl::EltwiseReduceMod(t, in[i].ptr(), count, qi, qi, 1);
                    const uint64_t w = ibase_.inv_punctured_prod_mod_base_array()[i].operand;
                    if (w != 1)
                    {{
                        intel::hexl::EltwiseFMAMod(t, t, w, nullptr, count, qi, 1);
                    }}
                }}
                for (size_t k = 0; k < obase_size; k++)
                {{
                    const uint64_t pk = obase_.base()[k].value();
                    uint64_t *o = out[k].ptr();
                    for (size_t i = 0; i < ibase_size; i++)
                    {{
                        const uint64_t qi = ibase_.base()[i].value();
                        const uint64_t *src = temp.get() + i * count;
                        uint64_t factor = 1;
                        if (qi > pk)
                        {{
                            const uint64_t f = (qi + pk - 1) / pk;
                            factor = f <= 2 ? 2 : f <= 4 ? 4 : f <= 8 ? 8 : 0;
                            if (!factor)
                            {{
                                intel::hexl::EltwiseReduceMod(scratch.get(), src, count, pk, pk, 1);
                                src = scratch.get();
                                factor = 1;
                            }}
                        }}
                        intel::hexl::EltwiseFMAMod(o, src, base_change_matrix_[k][i], i ? o : nullptr, count, pk, factor);
                    }}
                }}
                return;
            }}
#endif
"""
s = s[:a] + body + s[a:]

head = '#include "seal/util/uintarithsmallmod.h"\n'
if s.count(head) != 1:
    sys.exit(f"[seal] ERROR: {REL}: include anchor not found once; review {__file__}")
s = s.replace(head, head + f"""#ifdef SEAL_USE_INTEL_HEXL
#include "hexl/hexl.hpp"  // {MARK}
#endif
#ifndef SEAL_RVV_HEXL_BASECONV
#define SEAL_RVV_HEXL_BASECONV 1
#endif
""", 1)
p.write_text(s)
print(f"[seal] {REL}: fast_convert_array -> EltwiseReduceMod / EltwiseFMAMod")
