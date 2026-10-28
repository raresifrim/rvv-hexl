// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// rvv-hexl extension, NOT part of Intel HEXL: a multi-output modular dot product of
// vectors, the inner loop of key switching (SEAL Evaluator::switch_key_inplace,
// OpenFHE KeySwitchBV) and of binfhe's blind rotation.
// Guard uses with #ifdef HEXL_RVV_HAS_DOT_PRODUCT.
// Implementation: src/eltwise/eltwise-dot-product.cpp (dispatch + native),
// eltwise-dot-product-rvv.cpp (RVV).
//
// Key switching multiplies every decomposition digit by the matching key
// polynomial and sums over the digits, per key component and output limb. With
// rvv-hexl, SEAL and OpenFHE run it as one EltwiseMultAddMod per (digit, key
// component): a full pass over the accumulator and a full reduction per digit,
// the digit reloaded once per key component. One call reads every digit and key
// polynomial once, keeps the sums in registers and reduces once per result (the
// FP64 path can keep them as doubles, converting once). binfhe's CGGI step is two
// calls (one per key, the monomial as \p post); EltwiseExternalProductMod does it
// in one, reading the digits once.

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

/// @brief Multi-output dot product, for m = 0 .. M - 1:
///   result[m] = (accumulate ? result[m] : 0)
///             + post * sum_{i < k} a[i] * b[i M + m]     (mod modulus)
/// all products element-wise; a null \p post stands for the constant 1.
///   SEAL key switching:   M = 2 key components, a = the digits in [0, 4q)
///                         (input_mod_factor 4), accumulate
///   OpenFHE KeySwitchBV:  M = 2, a = the digits, overwrite
///   binfhe CGGI (one key): M = 2 accumulator polynomials, post = the monomial
/// @param[in,out] result M vectors of n elements, in [0, modulus) when
/// accumulating; results in [0, modulus). Must not alias any input or each other.
/// @param[in] num_results M, >= 1.
/// @param[in] a k vectors, in [0, input_mod_factor * modulus).
/// @param[in] num_terms k, >= 1.
/// @param[in] b k M vectors in [0, modulus): b[i M + m] multiplies a[i] into result[m].
/// @param[in] post One vector in [0, modulus), or nullptr.
/// @param[in] n Number of elements in each vector.
/// @param[in] modulus Modulus, in [2, 2^61), with input_mod_factor * modulus < 2^63.
/// @param[in] input_mod_factor Bound factor of \p a: 1, 2 or 4.
/// @param[in] accumulate Add into \p result (true) or overwrite it (false).
void EltwiseDotProductMod(uint64_t* const* result, uint64_t num_results,
		const uint64_t* const* a, uint64_t num_terms, const uint64_t* const* b,
		const uint64_t* post, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor, bool accumulate);

/// @brief Same, 32-bit storage (OpenFHE NATIVE_SIZE=32): input_mod_factor *
/// modulus <= 2^32.
void EltwiseDotProductMod(uint32_t* const* result, uint64_t num_results,
		const uint32_t* const* a, uint64_t num_terms, const uint32_t* const* b,
		const uint32_t* post, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor, bool accumulate);

}  // namespace hexl
}  // namespace intel
