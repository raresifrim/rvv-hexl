// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// SPDX-License-Identifier: Apache-2.0
//
// rvv-hexl extension, NOT part of Intel HEXL: element-wise fused multiply-add
// of three vectors. Guard uses with #ifdef HEXL_RVV_HAS_MULT_ADD_MOD.
// Implementation: src/eltwise/eltwise-mult-add-mod.cpp (dispatch + native),
// eltwise-mult-add-mod-rvv.cpp (RVV).
//
// The vector-vector counterpart of EltwiseFMAMod, whose multiplier is a single
// scalar (Shoup, with a precomputed factor); here every element has its own
// multiplier, so the product needs a Barrett reduction like EltwiseMultMod.
// Against EltwiseMultMod into a temporary followed by EltwiseAddMod, one pass
// instead of two: 4 words of memory traffic per element instead of 6, one
// reduction sequence, no temporary. Written for OpenFHE binfhe's external
// product, acc += digit (.) key and acc += tmp (.) monomial (rgsw-acc-*.cpp).

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

/// @brief Computes (\p arg1 * \p arg2 + \p arg3) mod \p modulus element-wise.
/// @param[out] result Stores the result. May alias arg1, arg2 and/or arg3
/// (accumulation: result == arg3).
/// @param[in] arg1 Vector to multiply. Elements in [0, input_mod_factor * modulus).
/// @param[in] arg2 Vector to multiply. Elements in [0, input_mod_factor * modulus).
/// @param[in] arg3 Vector to add, not nullptr (no addend: EltwiseMultMod).
/// Elements in [0, input_mod_factor * modulus).
/// @param[in] n Number of elements in each vector
/// @param[in] modulus Modulus with which to perform modular reduction. Must be
/// in [2, 2^61) (the pre-shift Barrett of EltwiseMultMod).
/// @param[in] input_mod_factor 1, 2 or 4.
/// @details Output is fully reduced, in [0, modulus).
void EltwiseMultAddMod(uint64_t* result, const uint64_t* arg1,
		const uint64_t* arg2, const uint64_t* arg3, uint64_t n,
		uint64_t modulus, uint64_t input_mod_factor);

/// @brief Same, 32-bit storage (OpenFHE NATIVE_SIZE=32): input_mod_factor *
/// modulus <= 2^32. q < 2^30 runs the RVV e32 kernels, larger q the native one.
void EltwiseMultAddMod(uint32_t* result, const uint32_t* arg1,
		const uint32_t* arg2, const uint32_t* arg3, uint64_t n,
		uint64_t modulus, uint64_t input_mod_factor);

}  // namespace hexl
}  // namespace intel
