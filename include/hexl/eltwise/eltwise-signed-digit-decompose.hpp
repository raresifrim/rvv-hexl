// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// rvv-hexl extension, NOT part of Intel HEXL: the signed gadget decomposition of
// OpenFHE binfhe's blind rotation (RingGSWAccumulator::SignedDigitDecompose).
// Guard uses with #ifdef HEXL_RVV_HAS_SIGNED_DIGIT_DECOMPOSE.
// Implementation: src/eltwise/eltwise-signed-digit-decompose.cpp (dispatch +
// native), eltwise-signed-digit-decompose-rvv.cpp (RVV).
//
// Every external product of the blind rotation splits each coefficient of the
// two accumulator polynomials into signed base-2^g digits, before the digit
// polynomials are transformed (NTT) and multiplied by the bootstrapping key.
// OpenFHE does it coefficient by coefficient, each digit going to a different
// polynomial, in a loop the compiler does not vectorise (26% of the TFHE
// central node on the X100 with rvv-hexl, up to 36% on the A100; blindloop
// profiles, 2026-10-18). This is the same computation on whole vectors: the
// digits of a strip of coefficients stay in registers, one unit-stride store
// per digit. OpenFHE STD128: Q 27-bit, N = 1024, g = 7, 3 digits kept.

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

/// @brief Approximate signed gadget decomposition (OpenFHE binfhe).
/// For every element x of \p operand, the centred value c = x when
/// x < floor(modulus / 2), else x - modulus, is split into signed digits of
/// base B = 2^log_base, lowest first: c = sum_i r_i B^i, r_i in [-B/2, B/2).
/// The lowest digit r_0 is dropped (approximate decomposition) and
///   result[d][j] = r_{d+1} mod modulus   (r + modulus when r < 0)
/// for d = 0 .. num_digits - 1; the digits above them are dropped too.
/// @param[out] result num_digits vectors of n elements, overwritten (OpenFHE
/// adds into zero-initialised polynomials: the same values). Must not alias
/// \p operand or each other.
/// @param[in] operand n elements in [0, modulus).
/// @param[in] n Number of elements in each vector.
/// @param[in] modulus Q, in [2, 2^62).
/// @param[in] log_base g, in [1, 62]: the gadget base is 2^g.
/// @param[in] num_digits Digits kept (OpenFHE: digitsG - 1), >= 1.
void SignedDigitDecompose(uint64_t* const* result, const uint64_t* operand,
		uint64_t n, uint64_t modulus, uint64_t log_base, uint64_t num_digits);

/// @brief Same, 32-bit storage (OpenFHE NATIVE_SIZE=32): modulus < 2^31,
/// log_base in [1, 30].
void SignedDigitDecompose(uint32_t* const* result, const uint32_t* operand,
		uint64_t n, uint64_t modulus, uint64_t log_base, uint64_t num_digits);

}  // namespace hexl
}  // namespace intel
