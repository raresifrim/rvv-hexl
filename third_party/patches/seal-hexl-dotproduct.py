#!/usr/bin/env python3
# Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
# All rights reserved.
#
"""Make SEAL's key switching one EltwiseDotProductMod call per output limb (SEAL v4.1.2).

    seal-hexl-dotproduct.py <SEAL source tree>

Applied by third_party/seal.sh after seal-hexl-keyswitch.py, inside that patch's
rvv-hexl branch. Idempotent; fails loudly if upstream changes shape. Compiled only
with SEAL_USE_INTEL_HEXL and the rvv-hexl extension (HEXL_RVV_HAS_DOT_PRODUCT);
-DSEAL_RVV_HEXL_DOTPRODUCT=0 (seal.sh: SEAL_RVV_HEXL_DOTPRODUCT=0) keeps
seal-hexl-keyswitch.py's EltwiseMultAddMod per (digit, key component), the
comparison baseline.

For each output limb I, Evaluator::switch_key_inplace sums over the decomposition
digits J (lazy NTT output in [0, 4q), or the input limb itself in CKKS / BGV when
I == J) times key_vector[J] component K, limb key_index, into t_poly_prod
component K, limb I. The digits are prepared as before, all of them first (one
buffer per digit instead of one reused), then one EltwiseDotProductMod writes the
M = key_component_count results: input_mod_factor 4, overwriting (t_poly_prod is
zero-initialised and every limb is written once), fully reduced as before.
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
REL = "native/src/seal/evaluator.cpp"
MARK = "rvv-hexl: patches/seal-hexl-dotproduct.py"
KEYSWITCH = "rvv-hexl: patches/seal-hexl-keyswitch.py"

p = ROOT / REL
s = p.read_text()
if MARK in s:
    print(f"[seal] {REL}: dot-product patch already applied")
    sys.exit(0)


def fail(msg):
    sys.exit(f"[seal] ERROR: {REL}: {msg}; review {__file__}")


if KEYSWITCH not in s:
    fail("seal-hexl-keyswitch.py must be applied first")

# seal-hexl-keyswitch.py's branch: "#if ...KEYSWITCH\n" <fused> "#else\n" <original> "#endif\n"
IF = "#if defined(SEAL_USE_INTEL_HEXL) && defined(HEXL_RVV_HAS_MULT_ADD_MOD) && SEAL_RVV_HEXL_KEYSWITCH\n"
if s.count(IF) != 1:
    fail("key-switch branch not found once")
a = s.index(IF) + len(IF)
b = s.index("#else\n", a)
fused = s[a:b]
if "intel::hexl::EltwiseMultAddMod(" not in fused or "#if" in fused:
    fail("unexpected key-switch branch body")

dot = f"""#if defined(HEXL_RVV_HAS_DOT_PRODUCT) && SEAL_RVV_HEXL_DOTPRODUCT
            // {MARK}: all digits of limb I first (lazy NTT output in
            // [0, 4q), or the input limb itself), then one dot product writes limb I of
            // every key component of t_poly_prod: input_mod_factor 4, overwriting the zeros
            {{
                const uint64_t key_q = key_modulus[key_index].value();
                SEAL_ALLOCATE_GET_RNS_ITER(t_digits, coeff_count, decomp_modulus_size, pool);
                std::vector<const uint64_t *> digit(decomp_modulus_size);
                std::vector<const uint64_t *> key(decomp_modulus_size * key_component_count);
                std::vector<uint64_t *> out(key_component_count);
                SEAL_ITERATE(iter(size_t(0)), decomp_modulus_size, [&](auto J) {{
                    // RNS-NTT form exists in input
                    if ((scheme == scheme_type::ckks || scheme == scheme_type::bgv) && (I == J))
                    {{
                        digit[J] = target_iter[J].ptr();
                    }}
                    // Perform RNS-NTT conversion
                    else
                    {{
                        CoeffIter t_ntt = t_digits[J];
                        // No need to perform RNS conversion (modular reduction)
                        if (key_modulus[J] <= key_modulus[key_index])
                        {{
                            set_uint(t_target[J], coeff_count, t_ntt);
                        }}
                        // Perform RNS conversion (modular reduction)
                        else
                        {{
                            modulo_poly_coeffs(t_target[J], coeff_count, key_modulus[key_index], t_ntt);
                        }}
                        // NTT conversion lazy outputs in [0, 4q)
                        ntt_negacyclic_harvey_lazy(t_ntt, key_ntt_tables[key_index]);
                        digit[J] = t_ntt.ptr();
                    }}
                    for (size_t K = 0; K < key_component_count; K++)
                    {{
                        key[J * key_component_count + K] = key_vector[J].data().data(K) + key_index * coeff_count;
                    }}
                }});
                for (size_t K = 0; K < key_component_count; K++)
                {{
                    out[K] = t_poly_prod.get() + (K * rns_modulus_size + I) * coeff_count;
                }}
                intel::hexl::EltwiseDotProductMod(
                    out.data(), key_component_count, digit.data(), decomp_modulus_size, key.data(), nullptr,
                    coeff_count, key_q, 4, false);
            }}
#else
"""
s = s[:a] + dot + fused + "#endif\n" + s[b:]

DEF = "#ifndef SEAL_RVV_HEXL_KEYSWITCH\n#define SEAL_RVV_HEXL_KEYSWITCH 1\n#endif\n"
if s.count(DEF) != 1:
    fail("SEAL_RVV_HEXL_KEYSWITCH default not found once")
s = s.replace(DEF, DEF + f"#ifndef SEAL_RVV_HEXL_DOTPRODUCT  // {MARK}\n#define SEAL_RVV_HEXL_DOTPRODUCT 1\n#endif\n")
if "#include <vector>" not in s:
    s = s.replace('#include "hexl/hexl.hpp"', '#include <vector>\n#include "hexl/hexl.hpp"', 1)
p.write_text(s)
print(f"[seal] {REL}: key switching -> EltwiseDotProductMod")
