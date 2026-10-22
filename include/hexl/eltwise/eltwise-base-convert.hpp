// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// rvv-hexl extension, NOT part of Intel HEXL: the approximate RNS base
// conversion (fast base extension) of OpenFHE's DCRTPoly::ApproxSwitchCRTBasis,
// which HYBRID key switching runs twice per key switch: ModUp (each digit to the
// other limbs and P) and ModDown (P back to Q). Guard uses with
// #ifdef HEXL_RVV_HAS_BASE_CONVERT.
// Implementation: src/eltwise/eltwise-base-convert.cpp (dispatch + native),
// eltwise-base-convert-rvv.cpp (RVV).
//
// OpenFHE runs it coefficient by coefficient, with 128-bit products summed and
// one 128-bit Barrett reduction per output limb, all scalar: after rvv-hexl it
// is 15% of the CKKS central node's FHE time on the X100 and 33% on the A100
// (blindloop ckks_openfhe profiles, 2026-10-21). Here the conversion runs on
// whole vectors of coefficients.

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

/// @brief Approximate RNS base conversion from the basis q_0 .. q_{L-1} to the
/// basis p_0 .. p_{K-1} (OpenFHE's ApproxSwitchCRTBasis): for every
/// coefficient k,
///   y_i = operand[i][k] * q_hat_inv_mod_q[i] mod q_i
///   result[j][k] = ( sum_i y_i * q_hat_mod_p[i * K + j] ) mod p_j
/// where, for Q = q_0 ... q_{L-1} and Q_i = Q / q_i, q_hat_inv_mod_q[i] =
/// Q_i^{-1} mod q_i and q_hat_mod_p[i * K + j] = Q_i mod p_j. "Approximate":
/// the result is x + u Q mod p_j for some 0 <= u < L, as in OpenFHE (no
/// correction of the overflow u), so the output equals OpenFHE's bit for bit.
/// @param[out] result K vectors of n elements, in [0, p_j). Must not alias
/// the operands.
/// @param[in] operand L vectors of n elements, operand[i] in [0, q_i).
/// @param[in] n Number of coefficients.
/// @param[in] in_moduli q_0 .. q_{L-1}, each in [2, 2^60); 1 <= L <= 128.
/// @param[in] in_count L.
/// @param[in] out_moduli p_0 .. p_{K-1}, each in [2, 2^60); K >= 1.
/// @param[in] out_count K.
/// @param[in] q_hat_inv_mod_q L values, q_hat_inv_mod_q[i] < q_i.
/// @param[in] q_hat_mod_p L x K values, row-major by input limb:
/// q_hat_mod_p[i * K + j] < p_j.
void BaseConvert(uint64_t* const* result, const uint64_t* const* operand,
		uint64_t n, const uint64_t* in_moduli, uint64_t in_count,
		const uint64_t* out_moduli, uint64_t out_count,
		const uint64_t* q_hat_inv_mod_q, const uint64_t* q_hat_mod_p);

}  // namespace hexl
}  // namespace intel
