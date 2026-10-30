// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// rvv-hexl extension, NOT part of Intel HEXL: the external product of OpenFHE
// binfhe's blind rotation in one call (RingGSWAccumulator{CGGI,DM,LMKCDEY}::AddToAcc*).
// Guard uses with #ifdef HEXL_RVV_HAS_EXTERNAL_PRODUCT.
// Implementation: src/eltwise/eltwise-external-product.cpp (dispatch + native),
// eltwise-external-product-rvv.cpp (RVV).
//
// After the digit decomposition and the forward NTTs, every external product of
// the blind rotation multiplies the k digit polynomials by the bootstrapping key
// and sums them, per output polynomial (and, in CGGI, per key and monomial).
// OpenFHE does it as one EltwiseMultMod plus k - 1 EltwiseMultAddMod per sum,
// then one more EltwiseMultAddMod with the monomial: 4(k + 1) passes over memory
// in CGGI, each with a full reduction, the digit polynomials reloaded four times
// and the partial sum stored and reloaded k times (the X100's largest TFHE
// function: 23% of the central node at 32-bit words, 34% at 64-bit; blindloop
// perf record, 2026-10-28). One call reads every digit and key polynomial once.
// OpenFHE STD128 / STD128_3: N = 1024, Q 27-bit, k = 6 digit polynomials.

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

/// @brief External product of a blind-rotation step, for j = 0, 1:
///   acc_j = (accumulate ? acc_j : 0)
///         + mono1 * sum_{i < k} digits[i] * key1[2 i + j]
///         + mono2 * sum_{i < k} digits[i] * key2[2 i + j]     (mod modulus)
/// all products element-wise. A null \p mono1 / \p mono2 stands for the
/// constant 1; a null \p key2 drops the second term (and \p mono2).
///   CGGI:    accumulate, key1 = ek1, mono1 = X^a, key2 = ek2, mono2 = X^-a
///   DM:      overwrite (accumulate = false), key1 = ek, no monomial, no key2
///   LMKCDEY: overwrite, key1 = ek, no monomial, no key2 (its Automorphism:
///            accumulate into acc0 = 0 and acc1, key1 = the automorphism key)
/// @param[in,out] acc0, acc1 The two accumulator polynomials, n elements in
/// [0, modulus); results in [0, modulus). Must not alias any input.
/// @param[in] digits k digit polynomials (EVALUATION format), in [0, modulus).
/// @param[in] num_digits k, >= 1.
/// @param[in] key1 2k polynomials: key1[2 i + j] multiplies digits[i] into acc_j.
/// @param[in] mono1 One polynomial, or nullptr.
/// @param[in] key2 2k polynomials, or nullptr.
/// @param[in] mono2 One polynomial, or nullptr. Ignored without \p key2.
/// @param[in] n Number of elements in each polynomial.
/// @param[in] modulus Q, in [2, 2^62). Every input is in [0, modulus).
/// @param[in] accumulate Add into acc0 / acc1 (true) or overwrite them (false).
void EltwiseExternalProductMod(uint64_t* acc0, uint64_t* acc1,
		const uint64_t* const* digits, uint64_t num_digits,
		const uint64_t* const* key1, const uint64_t* mono1,
		const uint64_t* const* key2, const uint64_t* mono2,
		uint64_t n, uint64_t modulus, bool accumulate);

/// @brief Same, 32-bit storage (OpenFHE NATIVE_SIZE=32): modulus < 2^32.
void EltwiseExternalProductMod(uint32_t* acc0, uint32_t* acc1,
		const uint32_t* const* digits, uint64_t num_digits,
		const uint32_t* const* key1, const uint32_t* mono1,
		const uint32_t* const* key2, const uint32_t* mono2,
		uint64_t n, uint64_t modulus, bool accumulate);

}  // namespace hexl
}  // namespace intel
