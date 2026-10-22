// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// When a modular multiply may run on the FP64 FMA instead of the e64 integer
// multiplier (rvv-hexl extension, shared by the NTT and the eltwise kernels).
// On the K3 vfmul / vfmacc f64 retire 4 elements per cycle on both clusters,
// vmul / vmulhu e64 1 (rvv-hexl/results/ntt-width-2026-10-20).
//
// Modular multiply with doubles (Harvey, "Faster arithmetic for
// number-theoretic transforms", J. Symb. Comput. 2014), a, b exact doubles,
// qd = q, qi = 1/q:
//   h = a * b                  the rounded product (vfmul)
//   l = fma(a, b, -h)          its exact rounding error: h + l = a * b (vfmsub)
//   t = round(h * qi)          quotient estimate (vfmul + vfcvt, or the
//                              2^52 + 2^51 shift trick: vfmadd + vfsub)
//   r = fma(-t, qd, h) + l     a * b - t q, exact while |r| < 2^53 (vfnmsac, vfadd)
//   r in (-q, q): + q when negative (in integer after the convert out:
//                 vadd + vminu, cheaper than vmflt + a masked vfadd)
// With a fixed operand w (NTT twiddle, EltwiseFMAMod's scalar) precompute
// w * qi: t = round(a * (w qi)), one multiply fewer. An addend c < q joins
// exactly: r = fma(-t, qd, h) + l + c, t estimated from h + c.
//
// Conditions (ModMulFp64Applicable):
//   q >= 2^30   below it the RVV32 kernels run e32 integer lanes, 8 elements
//               per cycle and fewer instructions: faster than FP64
//   q <  2^50   the lazy inputs (up to 4q, EltwiseFMAMod 8q < 2^53) and the
//               error terms are exact in a double's 53-bit significand
// FP32 is never worth it: q < 2^22 only, and FP32 runs at the e32 integer
// rate on the X100 (results/ntt-width-2026-10-20/fma-throughput32.txt).

#pragma once

#include <stdint.h>

#include "util/cpu-features.hpp"
#include "util/rvv-config.hpp"  // the switches (empty without RVV)

namespace intel {
namespace hexl {

/// Smallest modulus the FP64 paths take: below it, the e32 integer lanes win.
constexpr uint64_t kFp64MinModulus = 1ULL << 30;
/// Bound on the modulus: the lazy values (< 4q, < 8q for EltwiseFMAMod) must
/// be exact doubles.
constexpr uint64_t kFp64MaxModulus = 1ULL << 50;

/// The arithmetic condition of the FP64 modular multiply (no hardware, no switch).
constexpr bool ModMulFp64Applicable(uint64_t modulus) {
	return modulus >= kFp64MinModulus && modulus < kFp64MaxModulus;
}

#ifdef HEXL_HAS_RVV
/// Whether EltwiseMultMod / EltwiseFMAMod / EltwiseMultAddMod (64-bit storage)
/// run their FP64 kernels: the switch rvv::cfg::kEltwiseFp64, an RVV unit,
/// and the condition.
inline bool UseEltwiseFp64(uint64_t modulus) {
	return rvv::cfg::kEltwiseFp64 && has_rvv && ModMulFp64Applicable(modulus);
}
#endif

}  // namespace hexl
}  // namespace intel
