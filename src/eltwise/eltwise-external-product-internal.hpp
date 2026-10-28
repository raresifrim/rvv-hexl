// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// Kernels behind EltwiseExternalProductMod (rvv-hexl extension): the external
// product of OpenFHE binfhe's blind rotation (see the public header). Templates
// on the storage word, like the eltwise kernels:
//   Word = uint64_t  OpenFHE NATIVE_SIZE=64
//   Word = uint32_t  OpenFHE NATIVE_SIZE=32
// e32 lanes for q < kMaxModulusRVV32 (RVV32, either word), e64 lanes otherwise
// (RVV64, 64-bit storage only).
//
// The point of the fused kernel is to read each digit and key polynomial once
// per strip and keep the k-term sums in registers. With q < 2^28 (binfhe), a
// Barrett product left in [0, 2q) can be added to a 32-bit accumulator about
// eight times before it needs a correction, so the e32 kernel can stay at e32
// throughout and correct once per sum (on the K3 the e64 lanes multiply one
// element per cycle, the e32 lanes eight).

#pragma once

#include <stdint.h>

#include "hexl/number-theory/number-theory.hpp"
#include "util/cpu-features.hpp"
#include "util/rvv-config.hpp"  // lane types (empty without RVV)

namespace intel {
namespace hexl {

/// @brief Reference: the defining sums with one exact modular product per
/// term (MultiplyMod) and modular additions, element by element. The lane
/// tests compare the kernels against it; it is itself checked against the
/// OpenFHE call sequence (EltwiseMultMod + EltwiseMultAddMod) in test-eltwise.cpp.
template <typename Word>
inline void ExternalProductModReference(Word* acc0, Word* acc1,
		const Word* const* digits, uint64_t num_digits,
		const Word* const* key1, const Word* mono1,
		const Word* const* key2, const Word* mono2,
		uint64_t n, uint64_t modulus, bool accumulate) {
	Word* acc[2] = {acc0, acc1};
	for (uint64_t e = 0; e < n; ++e) {
		for (int j = 0; j < 2; ++j) {
			uint64_t s1 = 0, s2 = 0;
			for (uint64_t i = 0; i < num_digits; ++i) {
				s1 = AddUIntMod(s1, MultiplyMod(digits[i][e], key1[2 * i + j][e], modulus), modulus);
				if (key2) s2 = AddUIntMod(s2, MultiplyMod(digits[i][e], key2[2 * i + j][e], modulus), modulus);
			}
			if (mono1) s1 = MultiplyMod(s1, mono1[e], modulus);
			if (key2 && mono2) s2 = MultiplyMod(s2, mono2[e], modulus);
			uint64_t r = AddUIntMod(s1, s2, modulus);
			if (accumulate) r = AddUIntMod(r, acc[j][e], modulus);
			acc[j][e] = static_cast<Word>(r);
		}
	}
}

/// @brief Native (scalar) kernel.
template <typename Word>
void ExternalProductModNative(Word* acc0, Word* acc1,
		const Word* const* digits, uint64_t num_digits,
		const Word* const* key1, const Word* mono1,
		const Word* const* key2, const Word* mono2,
		uint64_t n, uint64_t modulus, bool accumulate);

#ifdef HEXL_HAS_RVV
/// @brief RVV, 32-bit lanes: modulus < kMaxModulusRVV32 (2^30). Word =
/// uint64_t narrows on load / widens on store (rvv::Load32 / Store32, so at
/// most u32m4); uint32_t loads directly.
template <typename Word,
	 class V = rvv::LaneFor<Word, rvv::cfg::ExternalProduct32U64, rvv::cfg::ExternalProduct32>>
	 void ExternalProductModRVV32(Word* acc0, Word* acc1,
			 const Word* const* digits, uint64_t num_digits,
			 const Word* const* key1, const Word* mono1,
			 const Word* const* key2, const Word* mono2,
			 uint64_t n, uint64_t modulus, bool accumulate);

/// @brief RVV, 64-bit lanes: every other modulus. 64-bit storage only.
template <class V = rvv::cfg::ExternalProduct64>
void ExternalProductModRVV64(uint64_t* acc0, uint64_t* acc1,
		const uint64_t* const* digits, uint64_t num_digits,
		const uint64_t* const* key1, const uint64_t* mono1,
		const uint64_t* const* key2, const uint64_t* mono2,
		uint64_t n, uint64_t modulus, bool accumulate);
#endif

}  // namespace hexl
}  // namespace intel
