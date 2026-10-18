#!/usr/bin/env python3
# Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
# SPDX-License-Identifier: Apache-2.0
#
"""Make SEAL's key switching accumulate with EltwiseMultAddMod (SEAL v4.1.2).

    seal-hexl-keyswitch.py <SEAL source tree>

Applied by third_party/seal.sh before every build. Idempotent; fails loudly if
upstream changes shape. The new code is compiled only with SEAL_USE_INTEL_HEXL and
the rvv-hexl extension (HEXL_RVV_HAS_MULT_ADD_MOD), so the stock build is unchanged;
-DSEAL_RVV_HEXL_KEYSWITCH=0 selects the original code in an rvv-hexl build too
(seal.sh: SEAL_RVV_HEXL_KEYSWITCH=0).

Evaluator::switch_key_inplace multiplies every decomposition digit (lazy NTT output,
in [0, 4q)) by the matching key polynomial and accumulates the products of all
digits in 128-bit form, reducing every SEAL_MULTIPLY_ACCUMULATE_USER_MOD_MAX digits
and once at the end. With rvv-hexl each (digit, key component) pair becomes one
EltwiseMultAddMod call that accumulates the reduced product straight into the
destination t_poly_prod: no 128-bit accumulator, no final reduction pass. Same
result: every step is (digit * key + acc) mod q, and acc is fully reduced.
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
REL = "native/src/seal/evaluator.cpp"
MARK = "rvv-hexl: patches/seal-hexl-keyswitch.py"

p = ROOT / REL
s = p.read_text()
if MARK in s:
    print(f"[seal] {REL}: key-switch patch already applied")
    sys.exit(0)


def cut(text, start, end):
    a = text.find(start)
    if a < 0 or text.count(start) != 1:
        sys.exit(f"[seal] ERROR: {REL}: anchor not found once: {start.strip()[:60]!r}; review {__file__}")
    b = text.find(end, a)
    if b < 0:
        sys.exit(f"[seal] ERROR: {REL}: end anchor not found: {end.strip()[:60]!r}; review {__file__}")
    return a, b


# the lazy accumulation, from its allocation to the end of the final reduction
a, b = cut(s, "            // Allocate memory for a lazy accumulator (128-bit coefficients)\n",
           "        });\n        // Accumulated products are now stored in t_poly_prod")
original = s[a:b]
# the digit preparation (copy or reduce, then lazy NTT) is kept as is
o1, o2 = cut(original, "                SEAL_ALLOCATE_GET_COEFF_ITER(t_ntt, coeff_count, pool);\n",
             "                // Multiply with keys and modular accumulate products in a lazy fashion\n")
operand = original[o1:o2]

fused = f"""#if defined(SEAL_USE_INTEL_HEXL) && defined(HEXL_RVV_HAS_MULT_ADD_MOD) && SEAL_RVV_HEXL_KEYSWITCH
            // {MARK}: one fused multiply-add per digit and key component, accumulated
            // straight into t_poly_prod (digit in [0, 4q) from the lazy NTT, key and
            // accumulator in [0, q): input_mod_factor 4; the output is fully reduced)
            PolyIter t_poly_prod_iter(t_poly_prod.get() + (I * coeff_count), coeff_count, rns_modulus_size);
            const uint64_t key_q = key_modulus[key_index].value();
            SEAL_ITERATE(iter(size_t(0)), decomp_modulus_size, [&](auto J) {{
{operand}                SEAL_ITERATE(iter(key_vector[J].data(), t_poly_prod_iter), key_component_count, [&](auto K) {{
                    uint64_t *acc = (*get<1>(K)).ptr();
                    intel::hexl::EltwiseMultAddMod(
                        acc, t_operand.ptr(), get<0>(K)[key_index].ptr(), acc, coeff_count, key_q, 4);
                }});
            }});
#else
"""
s = s[:a] + fused + original + "#endif\n" + s[b:]

head = """#include "seal/util/uintarith.h"
"""
if s.count(head) != 1:
    sys.exit(f"[seal] ERROR: {REL}: include anchor not found once; review {__file__}")
s = s.replace(head, head + f"""#ifdef SEAL_USE_INTEL_HEXL
#include "hexl/hexl.hpp"  // {MARK}
#endif
#ifndef SEAL_RVV_HEXL_KEYSWITCH
#define SEAL_RVV_HEXL_KEYSWITCH 1
#endif
""", 1)
p.write_text(s)
print(f"[seal] {REL}: key switching -> EltwiseMultAddMod")
