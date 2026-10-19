// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// SPDX-License-Identifier: Apache-2.0
//
// Kernels behind SignedDigitDecompose (rvv-hexl extension): the approximate
// signed gadget decomposition of OpenFHE binfhe (see the public header). Per
// element: centre x into (-q/2, q/2], drop the lowest signed digit, then
// num_digits times: r = the low log_base bits read as a signed number,
// c = (c - r) >> log_base, store r (+ q when negative) in result[d].
// Templates on the storage word, like the eltwise kernels:
//   Word = uint64_t  OpenFHE NATIVE_SIZE=64
//   Word = uint32_t  OpenFHE NATIVE_SIZE=32
// The arithmetic is signed: e32 lanes for q < 2^31 (RVV32, either word),
// e64 lanes otherwise (RVV64, 64-bit storage only).

#pragma once

#include <stdint.h>

#include <type_traits>

#include "util/cpu-features.hpp"
#include "util/rvv-config.hpp"  // lane types (empty without RVV)

namespace intel {
namespace hexl {

/// @brief Reference: a plain scalar version, one element at a time with
/// OpenFHE's own shifts (rgsw-acc.cpp). The lane tests compare the kernels
/// against it; it is itself checked against an independent oracle.
template <typename Word>
inline void SignedDigitDecomposeReference(Word* const* result, const Word* operand,
		uint64_t n, uint64_t modulus, uint64_t log_base, uint64_t num_digits) {
	using S = std::conditional_t<std::is_same_v<Word, uint32_t>, int32_t, int64_t>;
	using U = std::make_unsigned_t<S>;
	constexpr int kBits = static_cast<int>(sizeof(S) * 8);
	const int up = kBits - static_cast<int>(log_base);
	const int g = static_cast<int>(log_base);
	const S q = static_cast<S>(modulus);
	const uint64_t q_half = modulus >> 1;
	for (uint64_t j = 0; j < n; ++j) {
		const uint64_t x = operand[j];
		S c = x < q_half ? static_cast<S>(x) : static_cast<S>(x) - q;
		S r = static_cast<S>(static_cast<U>(c) << up) >> up;  // the dropped digit
		c = (c - r) >> g;
		for (uint64_t d = 0; d < num_digits; ++d) {
			r = static_cast<S>(static_cast<U>(c) << up) >> up;
			c = (c - r) >> g;
			result[d][j] = static_cast<Word>(r < 0 ? r + q : r);
		}
	}
}

/// @brief Native (scalar) kernel.
template <typename Word>
void SignedDigitDecomposeNative(Word* const* result, const Word* operand,
		uint64_t n, uint64_t modulus, uint64_t log_base, uint64_t num_digits);

#ifdef HEXL_HAS_RVV
/// @brief RVV, 32-bit signed lanes: modulus < 2^31. Word = uint64_t narrows on
/// load / widens on store (rvv::Load32 / Store32, so at most u32m4); uint32_t
/// loads directly.
template <typename Word,
	 class V = rvv::LaneFor<Word, rvv::cfg::SignedDigitDecompose32U64, rvv::cfg::SignedDigitDecompose32>>
		 void SignedDigitDecomposeRVV32(Word* const* result, const Word* operand,
				 uint64_t n, uint64_t modulus, uint64_t log_base, uint64_t num_digits);

/// @brief RVV, 64-bit signed lanes: every other modulus. 64-bit storage only.
template <class V = rvv::cfg::SignedDigitDecompose64>
void SignedDigitDecomposeRVV64(uint64_t* const* result, const uint64_t* operand,
		uint64_t n, uint64_t modulus, uint64_t log_base, uint64_t num_digits);
#endif

}  // namespace hexl
}  // namespace intel
